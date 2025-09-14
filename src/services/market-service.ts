import { getTicker, getOrderBook, getTrades } from "../api/public";
import { PrivateApi, CancelOrderParams, TradeResult } from "../types/private";
import type { Side } from "../types/domain";
import { logWarn } from "../utils/logger";
import { sleep } from "../utils/toolkit";
function toPosInt(val: any, def: number){ const n = Number(val); return Number.isFinite(n) && n >= 0 ? Math.floor(n) : def; }
const RETRY_ATTEMPTS = (()=>{ const n = toPosInt(process.env.RETRY_ATTEMPTS, 3); return n > 0 ? n : 3; })();
const RETRY_BACKOFF_MS = (()=>{ const n = toPosInt(process.env.RETRY_BACKOFF_MS, 50); return n >= 0 ? n : 50; })();
async function withRetry<T>(fn: ()=>Promise<T>, label: string, attempts = RETRY_ATTEMPTS, backoffMs = RETRY_BACKOFF_MS): Promise<T> {
  let err: any;
  for (let i=0;i<attempts;i++){
    try { return await fn(); } catch (e:any) {
      err = e;
  // If it's a hard network reset, don't keep trying too long
  const code = e?.code || e?.cause?.code;
  const isConnReset = code === 'ECONNRESET' || /ECONNRESET/i.test(String(e?.message||''));
  if (isConnReset && i>=0) break;
  if (i<attempts-1) {
  const delay = backoffMs * Math.pow(2, i); // 50 -> 100 -> 200 ...
  await sleep(delay);
      }
    }
  }
  throw Object.assign(new Error(`${label} failed: ${err?.message||String(err)}`), { cause: err });
}
let priv: PrivateApi;
export function init(privateApi: PrivateApi) { priv = privateApi; }

// --- short-lived cache for public endpoints ---
type CacheEntry<T> = { at: number; val: T };
const __IS_TEST__ = (process.env.TEST_MODE === '1') || !!process.env.VITEST_WORKER_ID;
const TTL_MS = (()=>{ const n = toPosInt(process.env.MARKET_CACHE_TTL_MS, 1500); const base = n >= 0 ? n : 1500; return __IS_TEST__ ? 0 : base; })();
const cacheTicker = new Map<string, CacheEntry<Awaited<ReturnType<typeof getTicker>>>>();
const cacheOrderBook = new Map<string, CacheEntry<Awaited<ReturnType<typeof getOrderBook>>>>();
const cacheTrades = new Map<string, CacheEntry<Awaited<ReturnType<typeof getTrades>>>>();

function isFresh(entryAt: number): boolean { return (Date.now() - entryAt) <= TTL_MS; }

export async function fetchTicker(pair: string){
  const hit = cacheTicker.get(pair);
  if (hit && isFresh(hit.at)) { console.log('[CACHE] ticker hit'); return hit.val; }
  try {
    const v = await withRetry(()=>getTicker(pair), 'getTicker');
    cacheTicker.set(pair, { at: Date.now(), val: v });
    return v;
  } catch (e:any) {
    const msg = String(e?.message || e);
    if (/401|unauthorized|429|too many/i.test(msg)) throw e;
    throw e;
  }
}

export async function fetchBoard(pair: string){
  const hit = cacheOrderBook.get(pair);
  if (hit && isFresh(hit.at)) { console.log('[CACHE] orderbook hit'); return hit.val; }
  const v = await withRetry(()=>getOrderBook(pair), 'getOrderBook');
  cacheOrderBook.set(pair, { at: Date.now(), val: v });
  return v;
}

export async function fetchRecentTrades(pair: string){
  const hit = cacheTrades.get(pair);
  if (hit && isFresh(hit.at)) { console.log('[CACHE] trades hit'); return hit.val; }
  const v = await withRetry(()=>getTrades(pair), 'getTrades');
  cacheTrades.set(pair, { at: Date.now(), val: v });
  return v;
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

export async function fetchBalance() {
  const res = await priv.get_info2();
  if (!res.success) throw new Error(res.error || "Unknown balance error");
  return res.return!;
}

export async function placeLimitOrder(pair: string, side: Side, price: number, amount: number) {
  const action = side === 'BUY' ? 'bid' : 'ask';
  let lastErr: any;
  for (let i=0;i<3;i++){
    try {
      const res: TradeResult = await priv.trade({ currency_pair: pair, action, price, amount });
      return res.return; // { order_id }
    } catch (e: any) {
      lastErr = e;
      const msg = String(e?.message || e?.error || '').toLowerCase();
  const isNonce = msg.includes('nonce') || msg.includes('invalid nonce');
  const isRateLimit = msg.includes('429') || msg.includes('too many requests') || msg.includes('rate limit');
  if (!(isNonce || isRateLimit) || i === 2) throw e;
  await sleep(100 * (i+1));
      continue;
    }
  }
  throw lastErr;
}

export async function listActiveOrders(currency_pair?: string) {
  const arr = await priv.active_orders({ currency_pair });
  return arr.map((o: any) => ({
    order_id: o.order_id,
    currency_pair: o.pair,
    action: o.side,
    amount: o.amount,
    price: o.price,
    timestamp: o.timestamp
  }));
}

export async function cancelOrder(params: CancelOrderParams) {
  const res = await priv.cancel_order({ order_id: String(params.order_id) });
  return res.return;
}

export async function cancelOrders(orderIds: number[]) {
  for (const id of orderIds) {
    try {
      await cancelOrder({ order_id: id });
    } catch {/* ignore individual fails */ }
  }
}

export async function fetchTradeHistory(pair: string, params: { since?: number; from_id?: number; count?: number } = {}) {
  return await priv.trade_history({ currency_pair: pair, ...params });
}

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