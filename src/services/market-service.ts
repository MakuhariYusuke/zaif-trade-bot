import { getTicker, getOrderBook, getTrades } from "../api/public";
import { PrivateApi, CancelOrderParams, TradeResult } from "../types/private";
import type { Side } from "../types/domain";
import { logWarn } from "../utils/logger";
const RETRY_ATTEMPTS = Number(process.env.RETRY_ATTEMPTS || 2);
const RETRY_BACKOFF_MS = Number(process.env.RETRY_BACKOFF_MS || 150);
async function withRetry<T>(fn: ()=>Promise<T>, label: string, attempts = RETRY_ATTEMPTS, backoffMs = RETRY_BACKOFF_MS): Promise<T> {
  let err: any;
  for (let i=0;i<attempts;i++){
    try { return await fn(); } catch (e:any) { err = e; if (i<attempts-1) await new Promise(r=>setTimeout(r, backoffMs)); }
  }
  throw Object.assign(new Error(`${label} failed: ${err?.message||String(err)}`), { cause: err });
}
let priv: PrivateApi;
export function init(privateApi: PrivateApi) { priv = privateApi; }

/** 
 * Fetches market overview including ticker, order book, and recent trades for a given currency pair.
 * @param {string} pair The currency pair to fetch market overview for.
 * @returns {Promise<Object>} The market overview data.
 */
export async function fetchMarketOverview(pair: string) {
  const [tRes, obRes, trRes] = await Promise.allSettled([
    withRetry(()=>getTicker(pair), 'getTicker'),
    withRetry(()=>getOrderBook(pair), 'getOrderBook'),
    withRetry(()=>getTrades(pair), 'getTrades'),
  ]);
  const ticker = tRes.status === 'fulfilled' ? tRes.value : undefined;
  const orderBook = obRes.status === 'fulfilled' ? obRes.value : { bids: [], asks: [] };
  const trades = trRes.status === 'fulfilled' ? trRes.value : [];
  if (tRes.status === 'rejected') logWarn(`[MARKET] getTicker failed for ${pair}: ${tRes.reason?.message || String(tRes.reason)}`);
  if (obRes.status === 'rejected') logWarn(`[MARKET] getOrderBook failed for ${pair}: ${obRes.reason?.message || String(obRes.reason)}`);
  if (trRes.status === 'rejected') logWarn(`[MARKET] getTrades failed for ${pair}: ${trRes.reason?.message || String(trRes.reason)}`);
  return { ticker, orderBook, trades } as any;
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
      if (!isNonce || i === 2) throw e;
      await new Promise(r=>setTimeout(r, 100 * (i+1)));
      continue;
    }
  }
  throw lastErr;
}

export async function listActiveOrders(currency_pair?: string) {
  const arr = await priv.active_orders({ currency_pair });
  const map: Record<string, any> = {};
  for (const o of arr) map[o.order_id] = { 
    currency_pair: o.pair, 
    action: o.side, 
    amount: o.amount, 
    price: o.price, 
    timestamp: o.timestamp 
  };
  return map;
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