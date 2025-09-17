import { getTicker, getOrderBook, getTrades } from "../api/public";
import { PrivateApi, CancelOrderParams, TradeResult } from "../types/private";
import type { Side } from "../types/domain";
import { logWarn, log as logCat } from "../utils/logger";
import { cacheHit, cacheMiss, cacheStale } from "../utils/cache-metrics";
import { sleep } from "../utils/toolkit";
import BaseService from "./base-service";
function toPosInt(val: any, def: number){ const n = Number(val); return Number.isFinite(n) && n >= 0 ? Math.floor(n) : def; }

// --- short-lived cache for public endpoints ---
type CacheEntry<T> = { at: number; val: T };
const __IS_TEST__ = (process.env.TEST_MODE === '1') || !!process.env.VITEST_WORKER_ID;
const TTL_MS = (()=>{ const n = toPosInt(process.env.MARKET_CACHE_TTL_MS, 1500); const base = n >= 0 ? n : 1500; return __IS_TEST__ ? 0 : base; })();

class MarketService extends BaseService {
  private cacheTicker = new Map<string, CacheEntry<Awaited<ReturnType<typeof getTicker>>>>();
  private cacheOrderBook = new Map<string, CacheEntry<Awaited<ReturnType<typeof getOrderBook>>>>();
  private cacheTrades = new Map<string, CacheEntry<Awaited<ReturnType<typeof getTrades>>>>();

  private isFresh(entryAt: number): boolean { return (Date.now() - entryAt) <= TTL_MS; }

  async fetchTicker(pair: string){
    const hit = this.cacheTicker.get(pair);
    if (hit) {
      if (this.isFresh(hit.at)) { try { logCat('DEBUG','CACHE','hit ticker',{ pair }); } catch {} cacheHit('market:ticker'); return hit.val; }
      cacheStale('market:ticker');
    } else {
      cacheMiss('market:ticker');
    }
    const t0 = Date.now();
    const v = await this.withRetry(()=>getTicker(pair), 'getTicker', undefined, undefined, { category: 'API-PUBLIC', requestId: undefined, pair, side: undefined, amount: undefined, price: undefined });
    const dt = Date.now() - t0; if (dt > 800) { this.clog('API-PUBLIC','WARN','slow public API',{ pair, op: 'getTicker', elapsedMs: dt }); }
    this.cacheTicker.set(pair, { at: Date.now(), val: v });
    return v;
  }

  async fetchBoard(pair: string){
    const hit = this.cacheOrderBook.get(pair);
    if (hit) {
      if (this.isFresh(hit.at)) { try { logCat('DEBUG','CACHE','hit orderbook',{ pair }); } catch {} cacheHit('market:orderbook'); return hit.val; }
      cacheStale('market:orderbook');
    } else {
      cacheMiss('market:orderbook');
    }
    const t0 = Date.now();
    const v = await this.withRetry(()=>getOrderBook(pair), 'getOrderBook', undefined, undefined, { category: 'API-PUBLIC', requestId: undefined, pair, side: undefined, amount: undefined, price: undefined });
    const dt = Date.now() - t0; if (dt > 800) { this.clog('API-PUBLIC','WARN','slow public API',{ pair, op: 'getOrderBook', elapsedMs: dt }); }
    this.cacheOrderBook.set(pair, { at: Date.now(), val: v });
    return v;
  }

  async fetchRecentTrades(pair: string){
    const hit = this.cacheTrades.get(pair);
    if (hit) {
      if (this.isFresh(hit.at)) { try { logCat('DEBUG','CACHE','hit trades',{ pair }); } catch {} cacheHit('market:trades'); return hit.val; }
      cacheStale('market:trades');
    } else {
      cacheMiss('market:trades');
    }
    const t0 = Date.now();
    const v = await this.withRetry(()=>getTrades(pair), 'getTrades', undefined, undefined, { category: 'API-PUBLIC', requestId: undefined, pair, side: undefined, amount: undefined, price: undefined });
    const dt = Date.now() - t0; if (dt > 800) { this.clog('API-PUBLIC','WARN','slow public API',{ pair, op: 'getTrades', elapsedMs: dt }); }
    this.cacheTrades.set(pair, { at: Date.now(), val: v });
    return v;
  }

  async fetchBalance() {
    if (!this.privateApi) throw new Error('Private API not initialized');
    const res = await this.privateApi.get_info2();
    if (!res.success) throw new Error(res.error || "Unknown balance error");
    return res.return!;
  }

  async placeLimitOrder(pair: string, side: Side, price: number, amount: number) {
    if (!this.privateApi) throw new Error('Private API not initialized');
    const action = side === 'BUY' ? 'bid' : 'ask';
    let lastErr: any;
    for (let i=0;i<3;i++){
      try {
        const res: TradeResult = await this.privateApi.trade({ currency_pair: pair, action, price, amount });
    return res.return; // { order_id }
      } catch (e: any) {
        lastErr = e;
        const msg = String(e?.message || e?.error || '').toLowerCase();
        const isNonce = msg.includes('nonce') || msg.includes('invalid nonce');
        const isRateLimit = msg.includes('429') || msg.includes('too many requests') || msg.includes('rate limit');
    // API retry WARN log with required meta
  this.clog('API-PRIVATE', 'WARN', 'retry', { requestId: undefined, pair, side: action, amount, price, retries: i+1, cause: { code: e?.code ?? e?.cause?.code ?? null, message: e?.message || e?.error } });
        // Non-retryable error or last attempt: break to emit final ERROR once below
        if (!(isNonce || isRateLimit) || i === 2) break;
        await sleep(100 * (i+1));
        continue;
      }
    }
  // final ERROR log with required meta
  this.clog('API-PRIVATE', 'ERROR', 'failed', { requestId: undefined, pair, side: action, amount, price, retries: 3, cause: { code: lastErr?.code ?? lastErr?.cause?.code ?? null, message: lastErr?.message || lastErr?.error } });
    throw lastErr;
  }

  async listActiveOrders(currency_pair?: string) {
    if (!this.privateApi) throw new Error('Private API not initialized');
    const arr = await this.privateApi.active_orders({ currency_pair });
    return arr.map((o: any) => ({
      order_id: o.order_id,
      currency_pair: o.pair,
      action: o.side,
      amount: o.amount,
      price: o.price,
      timestamp: o.timestamp
    }));
  }

  async cancelOrder(params: CancelOrderParams) {
    if (!this.privateApi) throw new Error('Private API not initialized');
  const meta = { category: 'API-PRIVATE', requestId: undefined, pair: undefined, side: undefined, amount: undefined, price: undefined };
    const fn = async () => this.privateApi!.cancel_order({ order_id: String(params.order_id) });
    const r = await this.withRetry(fn, 'cancelOrder', undefined, undefined, meta);
    return r.return;
  }

  async cancelOrders(orderIds: number[]) {
    for (const id of orderIds) {
      try {
        await this.cancelOrder({ order_id: id });
      } catch {/* ignore individual fails */ }
    }
  }

  async fetchTradeHistory(pair: string, params: { since?: number; from_id?: number; count?: number } = {}) {
    if (!this.privateApi) throw new Error('Private API not initialized');
    return await this.privateApi.trade_history({ currency_pair: pair, ...params });
  }
}

export interface MarketOverview {
  ticker?: Awaited<ReturnType<typeof getTicker>>;
  orderBook: Awaited<ReturnType<typeof getOrderBook>>;
  trades: Awaited<ReturnType<typeof getTrades>>;
}

/** 
 * Fetches market overview including ticker, order book, and recent trades for a given currency pair.
 * @param {string} pair The currency pair to fetch market overview for.
 * @returns {Promise<MarketOverview>} The market overview data.
 */
export async function fetchMarketOverview(pair: string): Promise<MarketOverview> {
  const [tRes, obRes, trRes] = await Promise.allSettled([
    fetchTicker(pair),
    fetchBoard(pair),
    fetchRecentTrades(pair),
  ]);
  const ticker = tRes.status === 'fulfilled' ? tRes.value : undefined;
  const orderBook = obRes.status === 'fulfilled' ? obRes.value : { bids: [], asks: [] };
  const trades = trRes.status === 'fulfilled' ? trRes.value : [];
  if (tRes.status === 'rejected') logWarn(`[MARKET] getTicker failed for ${pair}: ${tRes.reason?.message || String(tRes.reason)}`);
  if (obRes.status === 'rejected') logWarn(`[MARKET] getOrderBook failed for ${pair}: ${obRes.reason?.message || String(obRes.reason)}`);
  if (trRes.status === 'rejected') logWarn(`[MARKET] getTrades failed for ${pair}: ${trRes.reason?.message || String(trRes.reason)}`);
  return { ticker, orderBook, trades };
}

// Singleton instance to preserve legacy function API
const marketSvc = new MarketService();
let priv: PrivateApi;
export function init(privateApi: PrivateApi) { priv = privateApi; marketSvc.init(privateApi); }

export async function fetchTicker(pair: string){ return marketSvc.fetchTicker(pair); }
export async function fetchBoard(pair: string){ return marketSvc.fetchBoard(pair); }
export async function fetchRecentTrades(pair: string){ return marketSvc.fetchRecentTrades(pair); }
export async function fetchBalance() { return marketSvc.fetchBalance(); }
export async function placeLimitOrder(pair: string, side: Side, price: number, amount: number) { return marketSvc.placeLimitOrder(pair, side, price, amount); }

export async function listActiveOrders(currency_pair?: string) { return marketSvc.listActiveOrders(currency_pair); }

export async function cancelOrder(params: CancelOrderParams) { return marketSvc.cancelOrder(params); }

export async function cancelOrders(orderIds: number[]) { return marketSvc.cancelOrders(orderIds); }

export async function fetchTradeHistory(pair: string, params: { since?: number; from_id?: number; count?: number } = {}) { return marketSvc.fetchTradeHistory(pair, params); }

export interface RealizedPnLResult {
  realized: number; // in quote currency (e.g. JPY) simplistic
  trades: number;
}

// Very naive PnL calc: sums (sell_amount*price) - (buy_amount*price) matched FIFO until flat.
/**
 * Calculates realized PnL from trade history.
 * @param history Array of trade objects, each with:
 *   - trade_type: "bid" for buy, "ask" for sell
 *   - price: trade price
 *   - amount: trade amount
 */
export function calcRealizedPnL(
  history: Array<{ trade_type: string; price: number; amount: number }>
): RealizedPnLResult {
  const buys: Array<{ amount: number; price: number }> = [];
  let head = 0;
  let realized = 0;
  const EPS = 1e-12;

  for (const h of history) {
    if (h.trade_type === "bid") {
      buys.push({ amount: h.amount, price: h.price });
      continue;
    }

    let remain = h.amount;
    while (remain > 0 && head < buys.length) {
      const lot = buys[head];
      const used = Math.min(remain, lot.amount);
      realized += used * (h.price - lot.price);
      lot.amount -= used;
      remain -= used;
      if (lot.amount <= EPS) head++;
    }
  }

  return { realized, trades: history.length };
}
