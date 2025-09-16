import fs from 'fs';
import path from 'path';
import os from 'os';
import { PrivateApi, GetInfo2Response, ActiveOrder, TradeHistoryRecord, TradeResult, CancelResult } from '../types/private';
import { sleep } from '../utils/toolkit';

interface PaperTrade extends TradeHistoryRecord { currency_pair: string }
interface PaperOrder { 
    id: string;
    currency_pair: string;
    action: 'bid' | 'ask';
    price: number;
    amount: number;
    filled: number;
    ts: number;
    status: 'open' | 'canceled' | 'filled';
}
interface PaperState {
    funds: Record<string, number>;
    orders: PaperOrder[];
    trades: PaperTrade[];
}

const STORE = process.env.PAPER_STORE || path.join(os.tmpdir(), 'paper-trader.json');

/**
 * Loads the paper trading state from the designated file.
 * If the file does not exist or is invalid, returns a default state.
 * @return {PaperState} The loaded paper trading state.
 */
function load(): PaperState {
    try { if (fs.existsSync(STORE)) return JSON.parse(fs.readFileSync(STORE, 'utf8')); } catch { }
    return { funds: { jpy: Number(process.env.PAPER_JPY || 1000000), btc: Number(process.env.PAPER_BTC || 0) }, orders: [], trades: [] };
}
/**
 * Saves the paper trading state to the designated file.
 * @param {PaperState} st - The paper trading state to save.
 */
function save(st: PaperState) { try { fs.writeFileSync(STORE, JSON.stringify(st, null, 2)); } catch { } }

/**
 * Ensures that the funds for both sides of a trading pair are initialized.
 * @param {Record<string, number>} funds - The current funds state.
 * @param {string} pair - The trading pair (e.g., "btc_jpy").
 */
function ensurePairFunds(funds: Record<string, number>, pair: string) {
    const [base, quote] = pair.split('_');
    if (funds[base] == null) funds[base] = 0;
    if (funds[quote] == null) funds[quote] = 0;
}

/**
 * Ensures that the funds for both sides of a trading pair are initialized.
 * @param {Record<string, number>} funds - The current funds state.
 * @param {string} pair - The trading pair (e.g., "btc_jpy").
 * @param {string} action - The action being taken ("bid" or "ask").
 * @param {number} price - The price at which the trade is being executed.
 * @param {number} amount - The amount of the base currency being traded.
 */
function settle(funds: Record<string, number>, pair: string, action: 'bid' | 'ask', price: number, amount: number) {
    const [base, quote] = pair.split('_');
    if (action === 'bid') { // buy base, spend quote
        funds[base] = (funds[base] || 0) + amount;
        funds[quote] = (funds[quote] || 0) - price * amount;
    } else {
        funds[base] = (funds[base] || 0) - amount;
        funds[quote] = (funds[quote] || 0) + price * amount;
    }
}

/**
 * A paper trading implementation of the PrivateApi interface.
 * Simulates order placement, fills, cancellations, and maintains balances.
 * State is persisted to a JSON file specified by the PAPER_STORE environment variable.
 * Balances can be initialized via PAPER_JPY and PAPER_BTC environment variables.
 */
class PaperPrivate implements PrivateApi {
    private async maybeDelayAndError() {
        const lat = Number(process.env.PAPER_LATENCY_MS || 0);
        if (lat > 0) { await sleep(lat); }
        const errRate = Number(process.env.PAPER_ERROR_RATE || 0);
        if (errRate > 0 && Math.random() < errRate) throw new Error('Paper injected error');
    }
    async healthCheck() { return { ok: true, value: await this.get_info2() }; }
    async testGetInfo2() { return { ok: true, value: await this.get_info2() }; }
    async get_info2(): Promise<GetInfo2Response> {
        await this.maybeDelayAndError();
        const st = load();
        return {
            success: 1,
            return: {
                funds: st.funds,
                rights: { info: true, trade: true },
                open_orders: st.orders.filter(o => o.status === 'open').length,
                server_time: Math.floor(Date.now() / 1000)
            }
        };
    }
    async active_orders(p?: any): Promise<ActiveOrder[]> {
        await this.maybeDelayAndError();
        const st = load();
        return st.orders.filter(o => o.status === 'open' && (!p?.currency_pair || p.currency_pair === o.currency_pair)).map(o => ({
            order_id: o.id,
            pair: o.currency_pair,
            side: o.action,
            price: o.price,
            amount: o.amount - o.filled,
            timestamp: Math.floor(o.ts / 1000)
        }));
    }
    async trade_history(p?: any): Promise<TradeHistoryRecord[]> {
        await this.maybeDelayAndError();
        const st = load();
        const count = Math.max(1, Number(p?.count || 100));
        const filtered = Array.isArray(st.trades) ? st.trades.filter(t => !p?.currency_pair || p.currency_pair === t.currency_pair) : [];
        const list = filtered.slice(-count);
        return list as TradeHistoryRecord[];
    }
    async trade(p: any): Promise<TradeResult> {
        await this.maybeDelayAndError();
        const st = load(); ensurePairFunds(st.funds, p.currency_pair);
        const id = typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
            ? crypto.randomUUID()
            : String(Date.now()) + "-" + Math.random().toString(36).slice(2, 10);
        let amt = Number(p.amount);
        const price = Number(p.price);
        if (process.env.SAFETY_MODE === '1') {
            const [base, quote] = String(p.currency_pair).split('_');
            if (p.action === 'bid') {
                const jpy = st.funds[quote] || 0;
                const maxSpend = jpy * 0.10; // spend at most 10%, leaving >=90%
                const minAmt = Math.max(1e-8, maxSpend / Math.max(1, price));
                amt = Math.min(amt, minAmt);
            } else {
                const bal = st.funds[base] || 0;
                const maxSell = bal * 0.10; // sell at most 10%
                const minAmt = Math.max(1e-8, maxSell);
                amt = Math.min(amt, minAmt);
            }
        }
        const order: PaperOrder = {
            id,
            currency_pair: p.currency_pair,
            action: p.action,
            price,
            amount: amt,
            filled: 0,
            ts: Date.now(),
            status: 'open'
        };
        // immediate simulated partial/complete fill depending on PAPER_FILL_RATIO
        const base = Math.min(Math.max(Number(process.env.PAPER_FILL_RATIO || 1), 0), 1);
        const mode = String(process.env.PAPER_FILL_MODE || 'fixed').toLowerCase();
        function gaussianSample(mean: number, std: number) {
            // Box-Muller transform
            let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random();
            const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            return mean + z * std;
        }
        let ratio = base;
        if (mode === 'random') ratio = Math.random() * base;
        else if (mode === 'gaussian') ratio = Math.max(0, Math.min(1, gaussianSample(base, Math.max(0.01, base / 3))));
        const fillQty = Math.min(order.amount, order.amount * ratio);
        if (fillQty > 0) {
            order.filled += fillQty;
            settle(st.funds, order.currency_pair, order.action, order.price, fillQty);
            const tid = Date.now();
            st.trades.push({
                tid,
                order_id: id,
                side: order.action,
                price: order.price,
                amount: fillQty,
                timestamp: Math.floor(Date.now() / 1000),
                currency_pair: order.currency_pair
            });
            if (order.filled >= order.amount) order.status = 'filled';
        }
        st.orders.push(order);
        try {
            save(st);
        } catch (err) {
            console.error("Failed to save paper trading state:", err);
        }
        return { success: 1, return: { order_id: id } };
    }
    async cancel_order(p: { order_id: string }): Promise<CancelResult> {
        await this.maybeDelayAndError();
        const st = load(); const o = st.orders.find(x => x.id === String(p.order_id));
        if (o && o.status === 'open') { o.status = 'canceled'; save(st); }
        return { success: 1, return: { order_id: String(p.order_id) } };
    }
}

/** Creates a real private API client. @return {PrivateApi} The real private API client. */
export function createPrivateReal(): PrivateApi { return new PaperPrivate(); }
/** Creates a mock private API client. @return {PrivateApi} The mock private API client. */
export function createPrivateMock(): PrivateApi { return new PaperPrivate(); }
