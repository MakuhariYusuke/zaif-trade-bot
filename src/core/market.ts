import { getTicker, getOrderBook, getTrades } from "../api/public";
import { PrivateApi, CancelOrderParams, TradeResult } from "../types/private";
let priv: PrivateApi;
export function initMarket(privateApi: PrivateApi) { priv = privateApi; }

export async function fetchMarketOverview(pair: string) {
	const [ticker, orderBook, trades] = await Promise.all([
		getTicker(pair),
		getOrderBook(pair),
		getTrades(pair),
	]);
	const marketOverview = { ticker, orderBook, trades };
	return marketOverview;
}

export async function getAccountBalance() {
	const res = await priv.get_info2();
	if (!res.success) throw new Error(res.error || "Unknown balance error");
	return res.return!;
}

export async function placeLimitOrder(pair: string, side: "BUY"|"SELL", price: number, amount: number) {
	const action = side === 'BUY' ? 'bid' : 'ask';
	const res: TradeResult = await priv.trade({ currency_pair: pair, action, price, amount });
	return res.return;
}

export async function getActiveOrders(currency_pair?: string) {
	const arr = await priv.active_orders({ currency_pair });
	const map: Record<string, any> = {};
	for (const o of arr) map[o.order_id] = { currency_pair: o.pair, action: o.side, amount: o.amount, price: o.price, timestamp: o.timestamp };
	return map;
}

export async function cancelOrder(params: CancelOrderParams) {
	const res = await priv.cancel_order({ order_id: String(params.order_id) });
	return res.return;
}

export async function cancelOpenOrders(orderIds: number[]) {
	for (const id of orderIds) {
		try { await cancelOrder({ order_id: id }); } catch {}
	}
}

export async function fetchTradeHistory(pair: string, params: { since?: number; from_id?: number; count?: number } = {}) {
	return await priv.trade_history({ currency_pair: pair, ...params });
}

export interface RealizedPnLResult { realized: number; trades: number; }

/**
 * Calculate realized profit and loss (PnL) from a chronological trade history.
 *
 * This function scans the provided `history` array and matches sell trades against
 * prior buy trades using a FIFO (first-in, first-out) method. Each time a sell
 * ("ask") is matched to one or more earlier buys ("bid"), the realized PnL is
 * increased by the matched quantity multiplied by the difference between the
 * sell price and the buy price.
 *
 * Notes and assumptions:
 * - A trade object is expected to have the shape:
 *   `{ trade_type: string; price: number; amount: number }`.
 * - Trades with `trade_type === 'bid'` are treated as buys and are queued for
 *   later matching. Any other `trade_type` value is treated as a sell (i.e. an
 *   attempt to realize PnL).
 * - Sells are matched only against previously seen buys. If a sell exceeds the
 *   total available bought quantity, the excess sell amount is ignored (no
 *   shorting or negative position is modeled).
 * - Partial fills are supported: a single buy may be partially consumed by
 *   multiple later sells, and a single sell may consume multiple prior buys.
 * - A small epsilon (`1e-12`) is used to treat very small remaining amounts as
 *   exhausted to mitigate floating-point rounding issues.
 *
 * Complexity:
 * - Time: O(n), where n is the number of trades in `history`.
 * - Space: O(b), where b is the number of unmatched buys stored (worst-case O(n)).
 *
 * @param {Array<{ trade_type: string; price: number; amount: number }>} history 
 *   - Chronological list of trade records to process. Each record
 *   must include `trade_type` (buy indicated by `'bid'`), `price` (per-unit
 *   price), and `amount` (quantity).
 * @returns {RealizedPnLResult} An object of type `RealizedPnLResult` containing:
 *   - `realized`: total realized PnL as a number (sum of matched quantity * price difference).
 *   - `trades`: the number of trade records processed (i.e., `history.length`).
 *
 * @example
 * - Given buys: buy 10 @ 100, then sell 5 @ 110 -> realized = 5 * (110 - 100) = 50.
 */
export function calcRealizedPnL(history: Array<{ trade_type: string; price: number; amount: number }>): RealizedPnLResult {
	const buys: Array<{ amount: number; price: number }> = []; let head = 0; let realized = 0; const EPS = 1e-12;
	for (const h of history) {
		if (h.trade_type === 'bid') { buys.push({ amount: h.amount, price: h.price }); continue; }
		let remain = h.amount;
		while (remain > 0 && head < buys.length) {
			const lot = buys[head]; const used = Math.min(remain, lot.amount);
			realized += used * (h.price - lot.price);
			lot.amount -= used; remain -= used;
			if (lot.amount <= EPS) head++;
		}
	}
	return { realized, trades: history.length };
}
