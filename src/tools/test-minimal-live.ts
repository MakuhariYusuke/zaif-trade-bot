import dotenv from 'dotenv';
dotenv.config();
import fs from 'fs';
import path from 'path';
import { createPrivateApi } from '../api/adapters';
import { getOrderBook, getTrades } from '../api/public-router';
import { logInfo, logError, logWarn } from '../utils/logger';
import { incBuyEntry, incSellEntry } from '../utils/daily-stats';
import type { PrivateApi } from '../types/private';
import { logFeatureSample } from '../utils/features-logger';
import { appendPriceSamples, getPriceSeries } from '../utils/price-cache';
import { calculateSma, calculateRsi } from '../core/risk';

type Flow = 'BUY_ONLY' | 'SELL_ONLY' | 'BUY_SELL' | 'SELL_BUY';
const todayStr = () => new Date().toISOString().slice(0, 10);

function baseFromPair(pair: string) { return (pair || 'btc_jpy').split('_')[0]; }

async function fetchBalances(api: PrivateApi) {
    const r = await api.get_info2();
    if (!r.success || !r.return) throw new Error(r.error || 'get_info2 failed');
    return r.return.funds || {};
}

/**
 * Clamp the amount for safety.
 * @param { 'bid' | 'ask' } side The side of the order (bid or ask).
 * @param { number } amount The amount to trade.
 * @param { number } price The price at which to trade.
 * @param { Record<string, number> } funds The available funds.
 * @param { string } pair The currency pair.
 * @returns { number } The clamped amount.
 */
function clampAmountForSafety(side: 'bid' | 'ask', amount: number, price: number, funds: Record<string, number>, pair: string) {
    if (process.env.SAFETY_MODE !== '1') return amount;
    const base = baseFromPair(pair);
    if (side === 'bid') {
        const jpy = Number(funds.jpy || 0);
        const maxSpend = jpy * 0.10;
        if (maxSpend <= 0 || price <= 0) return amount;
        const maxQty = maxSpend / price;
        if (amount > maxQty) {
            logInfo(`[SAFETY] amount clamped BUY from ${amount} to ${maxQty.toFixed(8)} by 10% JPY`);
            return maxQty;
        }
        return amount;
    } else {
        const bal = Number((funds as any)[base] || 0);
        const maxQty = bal * 0.10;
        if (amount > maxQty) {
            logInfo(`[SAFETY] amount clamped SELL from ${amount} to ${maxQty.toFixed(8)} by 10% ${base.toUpperCase()}`);
            return maxQty;
        }
        return amount;
    }
}

/**
 * Place a limit order and immediately cancel it.
 * @param {PrivateApi} api The private API instance.
 * @param {string} pair The currency pair to trade.
 * @param {'bid'|'ask'} action The action to take (bid or ask).
 * @param {number} price The price at which to place the order.
 * @param {number} qty The quantity to trade.
 * @returns {Promise<{ id: string, filledQty: number, avgPrice: number }>} The order ID and fill information.
 */
async function placeAndCancel(api: PrivateApi, pair: string, action: 'bid' | 'ask', price: number, qty: number, opts?: { market?: boolean; refPx?: number }) {
    const market = !!opts?.market;
    const refPx = Number(opts?.refPx || price || 0);
    const params: any = { currency_pair: pair, action };
    if (market) {
        // Market: buy uses JPY notional; sell uses amount only
        if (action === 'bid') {
            const notional = Math.max(1, Math.round(qty * Math.max(1, refPx)));
            params.market_notional = notional;
        } else {
            params.amount = qty;
        }
    } else {
        params.price = price; params.amount = qty;
    }
    const r: any = await api.trade(params);
    const id = String(r?.return?.order_id || '');
    // For market orders, do not cancel; try to read fills directly
    if (market) {
        let filledQty = 0; let value = 0;
        try {
            const hist: any[] = await api.trade_history({ currency_pair: pair, count: 100 });
            const fills = hist.filter(h => String(h.order_id) === id);
            for (const f of fills) { const a = Number(f.amount || 0); const p = Number(f.price || 0); filledQty += a; value += a * p; }
        } catch {}
        const avgPrice = filledQty > 0 ? value / filledQty : 0;
        logInfo(`[LIVE][MARKET_FILLED] order_id=${id} filledQty=${filledQty} avgPrice=${avgPrice}`);
        return { id, filledQty, avgPrice, status: 'filled' as const };
    }
    // Limit: cancel immediately
    try {
        await api.cancel_order({ order_id: id });
    } catch (e: any) {
        logError('[LIVE][CANCEL_FAIL]', e?.message || e);
        return { id, filledQty: 0, avgPrice: 0, status: 'failed' as const };
    }
    // derive fills after cancel
    let filledQty = 0; let value = 0;
    try {
        const hist: any[] = await api.trade_history({ currency_pair: pair, count: 100 });
        const fills = hist.filter(h => String(h.order_id) === id);
        for (const f of fills) { const a = Number(f.amount || 0); const p = Number(f.price || 0); filledQty += a; value += a * p; }
    } catch {}
    const avgPrice = filledQty > 0 ? value / filledQty : 0;
    logInfo(`[LIVE][CANCELLED] order_id=${id} filledQty=${filledQty} avgPrice=${avgPrice}`);
    return { id, filledQty, avgPrice, status: 'cancelled' as const };
}

(async () => {
    const ex = (process.env.EXCHANGE || 'zaif').toLowerCase();
    const dry = process.env.DRY_RUN === '1';
    const pair = process.env.PAIR || 'btc_jpy';
    const flow = (process.env.TRADE_FLOW || 'BUY_ONLY') as Flow;
    const qtyRaw = Number(process.env.TEST_FLOW_QTY || '0');
    const rateInput = process.env.TEST_FLOW_RATE;
    const isMarket = (rateInput === '' || rateInput == null);
    const rateOverride = isMarket ? 0 : Number(rateInput || '0');
    if (dry) { console.error('DRY_RUN must be 0 for live test'); process.exit(1); }
    if (!(qtyRaw > 0)) { console.error('TEST_FLOW_QTY must be > 0'); process.exit(1); }
    const api = createPrivateApi();
    // default features source to 'live' for files under logs/features/live
    if (!process.env.FEATURES_SOURCE) process.env.FEATURES_SOURCE = 'live';
    const ob = await getOrderBook(pair);
    const bestBid = Number((ob?.bids?.[0]?.[0]) || 0);
    const bestBidSize = Number((ob?.bids?.[0]?.[1]) || 0);
    const bestAsk = Number((ob?.asks?.[0]?.[0]) || 0);
    const bestAskSize = Number((ob?.asks?.[0]?.[1]) || 0);
    // pxBid calculation priority: rateOverride > bestBid > fallback to bestAsk * 0.999
    // pxAsk: Use rateOverride if specified, otherwise use bestAsk if available, else fallback to bestBid * 1.001 or 1
    const pxBid = rateOverride > 0 ? rateOverride : (bestBid > 0 ? bestBid : Math.max(1, bestAsk * 0.999));
    const pxAsk = rateOverride > 0 ? rateOverride : (bestAsk > 0 ? bestAsk : Math.max(1, bestBid * 1.001));
    const funds = await fetchBalances(api);
    let balancesRef: Record<string, number> | null = funds;
    const balancesBefore = { jpy: Number((funds as any).jpy||0), btc: Number((funds as any).btc||0), eth: Number((funds as any).eth||0), xrp: Number((funds as any).xrp||0) };

    const featuresPair = `${ex}_${pair}`;

        async function recordFeatures(side: 'bid'|'ask', fillPrice: number, qty: number, status: 'cancelled'|'failed'){
        const ts = Date.now();
        try {
            // append latest price and compute indicators
            appendPriceSamples([{ ts, price: fillPrice || (side==='bid'? pxBid: pxAsk) }]);
            const rsiPeriod = Number(process.env.RSI_PERIOD || 14);
            const smaShortP = Number(process.env.SMA_SHORT || 9);
            const smaLongP = Number(process.env.SMA_LONG || 26);
            const series = getPriceSeries(Math.max(200, smaLongP+2, rsiPeriod+2));
            const rsi = calculateRsi(series, rsiPeriod);
            const smaS = calculateSma(series, smaShortP) as any;
            const smaL = calculateSma(series, smaLongP) as any;
            let volumeRecent = 0;
            try {
                const trades = await getTrades(pair);
                const nowSec = Math.floor(Date.now()/1000);
                for (const t of (trades||[])){
                    const dt = Number((t as any).date || (t as any).created_at || 0);
                    if (dt && nowSec - dt <= 60){ volumeRecent += Number((t as any).amount || 0); }
                }
            } catch {}
            const funds = balancesRef || {};
            logFeatureSample({ ts, pair: featuresPair, side, rsi: rsi ?? undefined, sma_short: smaS ?? undefined, sma_long: smaL ?? undefined, price: fillPrice || (side==='bid'? pxBid: pxAsk), qty, pnl: 0, win: null as any, balance: { jpy: (funds as any).jpy, btc: (funds as any).btc, eth: (funds as any).eth, xrp: (funds as any).xrp }, bestBid, bestAsk, status, depthBid: bestBidSize, depthAsk: bestAskSize, volumeRecent });
        } catch (e) {
            logError('[FEATURES_LOG_ERROR]', e instanceof Error ? e.message : String(e));
        }
    }

    // SAFETY_MODE はオプション（既定は無効）
    const executed: Array<{ side:'bid'|'ask'; qty:number; price:number; status:string }> = [];

    function warnIfOver5Pct(side:'bid'|'ask', qty:number, price:number){
        try{
            if (side==='bid'){
                const notional = qty * price;
                const jpy = balancesBefore.jpy || 0;
                if (jpy>0 && notional > jpy * 0.05){ logWarn(`[WARN][BALANCE] bid notional ${notional} exceeds 5% of JPY ${jpy}`); }
            } else {
                const base = baseFromPair(pair).toLowerCase();
                const bal = (balancesBefore as any)[base] || 0;
                if (bal>0 && qty > bal * 0.05){ logWarn(`[WARN][BALANCE] ask qty ${qty} exceeds 5% of ${base.toUpperCase()} ${bal}`); }
            }
        } catch {}
    }

    async function runBid() {
        let qty = clampAmountForSafety('bid', qtyRaw, pxBid, funds, pair);
        if (!(qty > 0)) throw new Error('clamped qty <= 0');
        warnIfOver5Pct('bid', qty, pxBid);
            const r = await placeAndCancel(api, pair, 'bid', pxBid, qty, { market: isMarket, refPx: pxAsk });
        incBuyEntry(todayStr(), pair);
        executed.push({ side:'bid', qty, price:pxBid, status: (r as any).status || 'cancelled' });
        try { await recordFeatures('bid', r.avgPrice || pxBid, qty, (r as any).status || 'cancelled'); } catch {}
        return r;
    }
    async function runAsk() {
        let qty = clampAmountForSafety('ask', qtyRaw, pxAsk, funds, pair);
        if (!(qty > 0)) throw new Error('clamped qty <= 0');
    warnIfOver5Pct('ask', qty, pxAsk);
            const r = await placeAndCancel(api, pair, 'ask', pxAsk, qty, { market: isMarket, refPx: pxBid });
        incSellEntry(todayStr(), pair);
    executed.push({ side:'ask', qty, price:pxAsk, status: (r as any).status || 'cancelled' });
    try { await recordFeatures('ask', r.avgPrice || pxAsk, qty, (r as any).status || 'cancelled'); } catch {}
        return r;
    }

    if (flow === 'BUY_ONLY') await runBid();
    else if (flow === 'SELL_ONLY') await runAsk();
    else if (flow === 'BUY_SELL') { await runBid(); await runAsk(); }
    else if (flow === 'SELL_BUY') { await runAsk(); await runBid(); }

    // Archive stats diff to logs/live
    try {
        const { spawnSync } = await import('child_process');
        const npx = process.platform === 'win32' ? 'npx.cmd' : 'npx';
        const r = spawnSync(npx, ['ts-node', 'src/tools/stats-today.ts', '--diff'], { encoding: 'utf8' });
        const out = (r.stdout || '').trim();
        const outDir = path.resolve(process.cwd(), 'logs', 'live');
        if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
        fs.writeFileSync(path.join(outDir, 'stats-diff-live.json'), out || '{}');
            } catch (e){ logError('[ARCHIVE_STATS_DIFF_ERROR]', e instanceof Error ? e.message : String(e)); }

        // Summary JSON
        try {
            const date = todayStr();
            const outDir = path.resolve(process.cwd(), 'logs', 'live');
            if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
            const diffPath = path.join(outDir, 'stats-diff-live.json');
                const after = await fetchBalances(api);
                const balancesAfter = { jpy: Number((after as any).jpy||0), btc: Number((after as any).btc||0), eth: Number((after as any).eth||0), xrp: Number((after as any).xrp||0) };
                const deltas = {
                    jpy: (balancesAfter.jpy - balancesBefore.jpy),
                    btc: (balancesAfter.btc - balancesBefore.btc),
                    eth: (balancesAfter.eth - balancesBefore.eth),
                    xrp: (balancesAfter.xrp - balancesBefore.xrp)
                };
                const warnings: string[] = [];
                // If any executed notional/qty exceeded 5%, we already logged; include summary markers
                for (const exed of executed){
                    if (exed.side==='bid'){
                        const notional = exed.qty * exed.price; if (balancesBefore.jpy>0 && notional > balancesBefore.jpy*0.05) warnings.push('over5pct_jpy');
                    } else {
                        const base = baseFromPair(pair).toLowerCase(); const bal = (balancesBefore as any)[base] || 0; if (bal>0 && exed.qty > bal*0.05) warnings.push('over5pct_base');
                    }
                }
                let summary: any = { env: { EXCHANGE: process.env.EXCHANGE, TRADE_FLOW: process.env.TRADE_FLOW, TEST_FLOW_QTY: process.env.TEST_FLOW_QTY, TEST_FLOW_RATE: process.env.TEST_FLOW_RATE, DRY_RUN: process.env.DRY_RUN }, balancesBefore, balancesAfter, deltas, executed, warnings };
            if (fs.existsSync(diffPath)){
                try {
                    const d = JSON.parse(fs.readFileSync(diffPath,'utf8'));
                    const vals = d?.values || {};
                    const diff = d?.diff || {};
                    summary.stats = {
                        incBuy: (diff.buyEntries||0),
                        incSell: (diff.sellEntries||0),
                        incPnl: (diff.realizedPnl||0),
                        winRate: (vals.trades? (vals.wins||0)/(vals.trades||1): 0),
                        streakWin: vals.streakWin||0,
                        streakLoss: vals.streakLoss||0
                    };
                } catch {}
            }
            // latest feature snapshot
            try {
                const latestPath = path.resolve(process.cwd(), 'logs', 'features', `latest-${featuresPair}.json`);
                if (fs.existsSync(latestPath)) summary.latestFeature = JSON.parse(fs.readFileSync(latestPath,'utf8'));
            } catch {}
            const sumPath = path.join(outDir, `summary-${date}.json`);
            fs.writeFileSync(sumPath, JSON.stringify(summary, null, 2));
        } catch (e){ logError('[SUMMARY_WRITE_ERROR]', e instanceof Error ? e.message : String(e)); }
})();
