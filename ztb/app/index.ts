import dotenv from "dotenv";
dotenv.config();
import { fetchMarketOverview, getAccountBalance, getActiveOrders, cancelOpenOrders } from "../core/market";
import { initExecution } from "../core/execution";
// Migrate away from deprecated adapters/risk-service: use core exports directly
import { getRiskConfig, calculateSma as calculateSma, describeExit } from "../core/risk";
import { loadPositions as getPositions, savePositions as savePositionsToFile } from "../adapters/risk-config";
import { logInfo, logError } from "../utils/logger";
import { getPriceSeries } from "../utils/price-cache";
import { createPrivateApi } from "../api/adapters";
import { PrivateApi } from "../types/private";
import { setNonceBase } from "../utils/signer";
import { loadAppConfig, restoreNonce, loadTradeMode, loadTradeFlow } from "../utils/config";
import * as fsMod from "fs";
import { TradeMode, TradeFlow } from "../types/private";
import { runSellStrategy } from "../core/strategies/sell-strategy";
import { runBuyStrategy } from "../core/strategies/buy-strategy";
import { incBuyEntry, incSellEntry } from "../utils/daily-stats";
import { initMarket } from "../core/market";
import { sleep } from '../utils/toolkit'; // Moved to top-level static import
import { registerAllSubscribers } from '../application/events';

const EX = (process.env.EXCHANGE || 'zaif').toLowerCase();
const privateApi: PrivateApi = createPrivateApi();
logInfo(`[EXCHANGE] ${EX}`);
logInfo(`[BACKEND] ${process.env.USE_PRIVATE_MOCK === "1" ? "MOCK" : "REAL"}`);
initMarket(privateApi);
initExecution(privateApi);
// Register application-wide event subscribers (position/stats/logger/trade-logger)
try { registerAllSubscribers(); } catch {}
const appConfig = loadAppConfig();
if (EX === 'zaif') {
    restoreNonce(appConfig.nonceStorePath);
    if (process.env.ZAIF_STARTING_NONCE) { const n = Number(process.env.ZAIF_STARTING_NONCE); if (!Number.isNaN(n)) setNonceBase(n); }
}
const hasCreds = EX === 'zaif'
    ? (!!process.env.ZAIF_API_KEY && !!process.env.ZAIF_API_SECRET)
    : (!!process.env.COINCHECK_API_KEY && !!process.env.COINCHECK_API_SECRET);
if (!hasCreds) {
    if (EX === 'zaif') logError('ZAIF_API_KEY / ZAIF_API_SECRET が設定されていません。プライベートAPIは利用できません。');
    if (EX === 'coincheck') logError('COINCHECK_API_KEY / COINCHECK_API_SECRET が設定されていません。プライベートAPIは利用できません。');
}

/**
 * Executes a single iteration of the trading strategy for a given trading pair.
 *
 * This asynchronous routine performs market data retrieval, indicator calculations,
 * position evaluation, trailing-stop management, and optionally executes exit actions.
 * It is intended to be run periodically (e.g. on a timer or cron) and is resilient to
 * transient failures by catching and logging errors rather than throwing them.
 *
 * Behavior summary:
 * - Loads risk/config and existing positions from disk.
 * - Fetches market overview (ticker, order book, trades).
 * - Honors a global emergency KILL_SWITCH: if process.env.KILL_SWITCH === '1' it
 *   skips placing new orders and attempts to cancel open orders for the given pair.
 * - Logs market and account information and appends recent trade price samples to a
 *   local time-series store.
 * - Computes indicators (SMA, RSI) as configured and caches a last-run timestamp in
 *   a small stamp file ('.indicator_stamp') to throttle indicator computation to
 *   riskCfg.indicatorIntervalSec.
 * - Evaluates exit signals:
 *   - Uses evaluateExitConditions on active (long) positions.
 *   - Manages trailing stops per position via manageTrailingStop and aggregates exits.
 *   - If execution is enabled and not in dry-run mode, removes exited positions from
 *     the positions file (savePositionsToFile) and updates any related counters (e.g.
 *     incTrailStop, incSellEntry).
 * - Computes short/long SMA and RSI for entry/exit heuristics and records entry counter
 *   when final sell conditions are met.
 *
 * Side effects:
 * - Reads and writes the positions file determined by riskCfg.positionsFile.
 * - Reads and writes a local indicator stamp file named '.indicator_stamp'.
 * - Calls external APIs for market data, account balance, active orders, and order cancellations.
 * - Updates an in-process global marker (global._indicatorLastRun) to avoid over-computing indicators.
 * - Logs informational and error messages via logInfo/logError and may modify on-disk counters via inc* helpers.
 *
 * Environment variables and runtime flags consulted:
 * - API_KEY, API_SECRET: required to proceed with the strategy's active account operations.
 * - KILL_SWITCH: if '1' aborts new order placement and attempts to cancel existing open orders.
 * - DRY_RUN: if '1' prevents actual execution of exit actions even when EXECUTE is true.
 * - SMA_SHORT (default 9), SMA_LONG (default 26), RSI_PERIOD (default 14): override indicator periods.
 *
 * Notes on execution semantics:
 * - The function swallows exceptions internally and logs them; it does not propagate errors to callers.
 * - If EXECUTE is truthy and DRY_RUN is not set to '1', the function will persistently remove positions
 *   that were exited during this run. Otherwise it will only log exit signals.
 * - Trailing-stop exits trigger both a logged exit and a call to incTrailStop per exit.
 *
 * @param {string} pair - Trading pair identifier (e.g., "BTC/JPY") for which to run the strategy.
 * @param {boolean} EXECUTE - When true, perform state-mutating actions (remove positions / cancel orders) subject to DRY_RUN and KILL_SWITCH.
 * @returns A Promise that resolves once the strategy iteration has completed (never rejects; errors are logged).
 */
async function strategyOnce(pair: string, EXECUTE: boolean) {
    const riskCfg = getRiskConfig(); const positionsFile = riskCfg.positionsFile; let positions = getPositions(positionsFile);
    const overview = await fetchMarketOverview(pair);
    if (process.env.KILL_SWITCH === '1') {
        logError('KILL_SWITCH active: skipping new orders');
        try {
            const active = await getActiveOrders(pair);
            const ids = Object.keys(active).map(id => Number(id)).filter(id => !isNaN(id));
            if (ids.length) {
                logInfo('KILL_SWITCH canceling', { count: ids.length });
                await cancelOpenOrders(ids);
            }
        } catch { }
        return;
    }
    const tradeMode: TradeMode = loadTradeMode();
    const tradeFlow: TradeFlow = loadTradeFlow();
    logInfo(`[FLOW] ${tradeFlow}`);
    logInfo(`Ticker:`, overview.ticker); logInfo('Best Ask:', overview.orderBook.asks[0]); logInfo('Best Bid:', overview.orderBook.bids[0]);
    if (!hasCreds) return;
    try {
        const balance = await getAccountBalance();
        logInfo('Funds:', balance.funds);
        await getActiveOrders(pair);
        const now = Date.now();
        const currentPrice = overview.ticker.last;
        const priceSeries = getPriceSeries(Math.max(riskCfg.smaPeriod, 200));
        const stampFile = '.indicator_stamp';
        if (!(global as any)._indicatorLastRun) {
            try {
                const data = fsMod.readFileSync(stampFile, 'utf8');
                if (data) {
                    (global as any)._indicatorLastRun = Number(data);
                } else {
                    (global as any)._indicatorLastRun = 0;
                }
            } catch {
                (global as any)._indicatorLastRun = 0;
            }
        }
        const lastRun = (global as any)._indicatorLastRun || 0;
        const nowSec = Math.floor(Date.now() / 1000);
        let sma: number | null = null;
        if (nowSec - lastRun >= riskCfg.indicatorIntervalSec) {
            sma = calculateSma(priceSeries, riskCfg.smaPeriod);
            (global as any)._indicatorLastRun = nowSec;
            fsMod.writeFile(stampFile, String(nowSec), err => {
                if (err) logError('Failed to write stamp', err?.message || err);
            });
            if (sma) logInfo(`SMA(${riskCfg.smaPeriod}):`, sma);
        }
        // Strategy dispatch (signals and exits)
    const ctx = { positions, positionsFile, currentPrice, trades: overview.trades, nowMs: now, riskCfg, pair } as any;
        const { allExits } = tradeMode === 'SELL' ? await runSellStrategy(ctx) : await runBuyStrategy(ctx);
        const dryRun = process.env.DRY_RUN === '1';
        if (allExits.length) {
            for (const sig of allExits) { logInfo(`[SIGNAL][mode=${tradeMode}] ${describeExit(sig)}`); }
            if (EXECUTE && !dryRun) {
                const toRemove = new Set<string>();
                for (const sig of allExits) { toRemove.add(sig.position.id); }
                positions = positions.filter(p => !toRemove.has(p.id));
                savePositionsToFile(positionsFile, positions);
            }
        }

        // Optional order placement flow (BUY/SELL sets) — also runs in DRY_RUN (simulated immediate fill)
        const qty = Number(process.env.TEST_FLOW_QTY || '0');
        const doFlow = qty > 0;
        if (doFlow && (privateApi as any).trade) {
            const dry = process.env.DRY_RUN === '1';
            const today = new Date().toISOString().slice(0, 10);
            const place = async (action: 'bid' | 'ask', price: number) => {
                if (dry) {
                    const orderId = `DRY-${Date.now()}`;
                    logInfo(`[FLOW][DRY] placed ${action} id=${orderId} price=${price} qty=${qty}`);
                    logInfo(`[FLOW][DRY] filled ${action} id=${orderId} filledQty=${qty} avgPrice=${price}`);
                    return { order_id: orderId, filledQty: qty, avgPrice: price };
                }
                const r: any = await (privateApi as any).trade({ currency_pair: pair, action, price, amount: qty });
                const id = String(r?.return?.order_id || '');
                const hist: any[] = (await (privateApi as any).trade_history({ currency_pair: pair, count: 50 })) || [];
                const fills = hist.filter(h => String(h.order_id) === id);
                const filledQty = fills.reduce((s, f) => s + Number(f.amount || 0), 0);
                const avgPrice = filledQty > 0 ? fills.reduce((s, f) => s + Number(f.amount) * Number(f.price), 0) / filledQty : price;
                logInfo(`[FLOW] placed ${action} id=${id} price=${price} qty=${qty}`);
                logInfo(`[FLOW] filled ${action} id=${id} filledQty=${filledQty} avgPrice=${avgPrice}`);
                return { order_id: id, filledQty, avgPrice };
            };
            const bestBid = Number(overview.orderBook?.bids?.[0]?.[0] || currentPrice * 0.999);
            const bestAsk = Number(overview.orderBook?.asks?.[0]?.[0] || currentPrice * 1.001);
            const flow = tradeFlow;
            const flows: Record<string, () => Promise<void>> = {
                BUY_ONLY: async () => { 
                    await place('bid', bestBid);
                    incBuyEntry(today, pair);
                },
                SELL_ONLY: async () => {
                    await place('ask', bestAsk);
                    incSellEntry(today, pair);
                },
                BUY_SELL: async () => {
                    const b = await place('bid', bestBid);
                    incBuyEntry(today, pair);
                    if (b.filledQty > 0) await place('ask', Math.max(bestAsk, b.avgPrice * 1.001));
                    incSellEntry(today, pair);
                },
                SELL_BUY: async () => {
                    const s = await place('ask', bestAsk);
                    incSellEntry(today, pair);
                    if (s.filledQty > 0) await place('bid', Math.min(bestBid, s.avgPrice * 0.999));
                    incBuyEntry(today, pair);
                }
            };
            if (flows[flow]) {
                await flows[flow]();
            } else {
                logError(`[FLOW] Unknown tradeFlow: ${flow}. No action taken.`);
            }
        }
    } catch (err: unknown) {
        if (err instanceof Error) {
            logError('strategyOnce error', err.message);
        } else {
            logError('strategyOnce error', String(err));
        }
    }
}

// Entry point run loop simplified
if (require.main === module) {
    (async () => {
        const pair = process.env.PAIR || 'btc_jpy';
        const interval = Number(process.env.LOOP_INTERVAL_MS || 15000);
        const EXECUTE = process.env.DRY_RUN !== '1';
        // Simple pre-start health check (optional)
        try {
            try {
                const fn = (privateApi as any).testGetInfo2;
                const hc = typeof fn === 'function' ? await fn() : null;
                if (hc?.ok) logInfo('Private health OK');
                else if (hc) logError('Private health warn', hc);
            } catch (e: any) {
                logError('Private health error', e?.message || String(e));
            }
        } catch (e: any) { logError('Health check exception', e?.message || e); }
        while (true) {
            await strategyOnce(pair, EXECUTE);
            await sleep(interval);
        }
    })();
}

/** 
 * Re-exports strategy helpers from app/index.ts
 * Starts the periodic strategy loop when this file is executed directly (npm start)
 */
export { strategyOnce };