import { getTicker, getOrderBook, getTrades } from "../api/public";
import { PrivateApi, CancelOrderParams, TradeResult } from "../types/private";
import type { Side } from "../types/domain";
let priv: PrivateApi;
export function init(privateApi: PrivateApi) { priv = privateApi; }

/** 
 * Fetches market overview including ticker, order book, and recent trades for a given currency pair.
 * @param {string} pair The currency pair to fetch market overview for.
 * @returns {Promise<Object>} The market overview data.
 */
export async function fetchMarketOverview(pair: string) {
  const [ticker, orderBook, trades] = await Promise.all([
    getTicker(pair),
    getOrderBook(pair),
    getTrades(pair),
  ]);

  return {
    ticker,
    orderBook,
    trades,
  };
}

export async function fetchBalance() {
  const res = await priv.get_info2();
  if (!res.success) throw new Error(res.error || "Unknown balance error");
  return res.return!;
}

export async function placeLimitOrder(pair: string, side: Side, price: number, amount: number) {
  const action = side === 'BUY' ? 'bid' : 'ask';
  const res: TradeResult = await priv.trade({ currency_pair: pair, action, price, amount });
  return res.return; // { order_id }
}

export async function listActiveOrders(currency_pair?: string) {
  const arr = await priv.active_orders({ currency_pair });
  const map: Record<string, any> = {};
  for (const o of arr) map[o.order_id] = { currency_pair: o.pair, action: o.side, amount: o.amount, price: o.price, timestamp: o.timestamp };
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