
import crypto from 'crypto';
import axios from 'axios';
import { PrivateApi, GetInfo2Response, ActiveOrder, TradeHistoryRecord, TradeResult, CancelResult } from '../types/private';
import { CoincheckBalanceResponse, CoincheckOrderRequest, CoincheckOrderResponse, CoincheckCancelResponse, CoincheckOpenOrdersResponse, CoincheckTradeHistoryResponse } from '../types/exchange-coincheck';

const API_BASE = 'https://coincheck.com';

/**
 * Create a Coincheck private API client implementing the application's PrivateApi interface.
 *
 * This factory returns an object with methods that wrap Coincheck's private REST endpoints,
 * handling authentication headers (ACCESS-KEY, ACCESS-NONCE, ACCESS-SIGNATURE), request/response
 * normalization, and simple error handling.
 *
 * Authentication and signing
 * - Each request uses a nonce based on Date.now().toString().
 * - The HMAC-SHA256 signature is computed over: nonce + url + body using the provided apiSecret.
 * - The apiKey is set on the ACCESS-KEY header.
 *
 * Timeout
 * - HTTP requests use axios and respect the timeout specified by the environment variable
 *   CC_TIMEOUT_MS (milliseconds). If CC_TIMEOUT_MS is not set, a default of 10000 ms is used.
 *
 * Returned API surface
 * - get_info2(): Promise<GetInfo2Response>
 *   - Fetches account balances from /api/accounts/balance.
 *   - Normalizes numeric string fields into a funds map (Record<string, number>).
 *   - Returns a GetInfo2Response-like object containing: funds, rights, open_orders, server_time.
 *   - Throws an Error when the Coincheck response indicates failure or the payload is missing.
 *
 * - active_orders(): Promise<ActiveOrder[]>
 *   - Calls /api/exchange/orders/opens and maps Coincheck orders to the application's ActiveOrder[]
 *     shape.
 *   - Mapping details:
 *     - order_id: stringified Coincheck order id
 *     - pair: 'btc_jpy' (hard-coded)
 *     - side: Coincheck order_type 'buy' => 'bid', 'sell' => 'ask'
 *     - price: Number(o.rate)
 *     - amount: Number(o.pending_amount)
 *     - timestamp: seconds since epoch parsed from o.created_at
 *   - Returns an empty array when the Coincheck response is not successful.
 *
 * - trade_history(): Promise<TradeHistoryRecord[]>
 *   - Calls /api/exchange/orders/transactions and maps Coincheck trade records to TradeHistoryRecord[].
 *   - Mapping details:
 *     - tid: Coincheck trade id
 *     - order_id: stringified Coincheck order_id
 *     - side: 'buy' => 'bid', 'sell' => 'ask'
 *     - price: Number(rate)
 *     - amount: Number(amount)
 *     - timestamp: seconds since epoch parsed from created_at
 *   - Returns an empty array when the Coincheck response is not successful.
 *
 * - trade(p: any): Promise<TradeResult>
 *   - Places an order at POST /api/exchange/orders.
 *   - Accepts an object shaped like the application's trade request (Zaif-style compatibility).
 *   - Validation and mapping:
 *     - p.action must be 'bid' or 'ask' (throws Error if invalid).
 *     - pair defaults to 'btc_jpy' when not provided.
 *     - order_type maps: 'bid' => 'buy', 'ask' => 'sell'.
 *     - rate is taken from p.price or p.limitPrice and converted to Number.
 *     - amount is converted to Number.
 *   - Body is sent as application/x-www-form-urlencoded.
 *   - Returns a TradeResult with success and return.order_id on success.
 *   - Throws an Error when Coincheck reports failure.
 *
 * - cancel_order(p: { order_id: string }): Promise<CancelResult>
 *   - Cancels an order at POST /api/exchange/orders/{id}/cancel.
 *   - Returns a CancelResult with success and return.order_id on success.
 *   - Throws an Error when Coincheck reports failure.
 *
 * Error handling
 * - Network and HTTP errors are propagated as thrown exceptions from axios.
 * - When Coincheck responds with a success: false (or missing success flag), the client methods
 *   either throw (for critical operations like trade/cancel/get_info2) or return an empty array
 *   (for list retrievals like active_orders and trade_history) per the implementation.
 *
 * Notes and caveats
 * - This client assumes JPY/BTC usage (pair set to 'btc_jpy' in list mappings) unless overridden
 *   in the trade() call via p.currency_pair.
 * - All numeric conversions (rate, amount, balances) use Number(...). Be aware of precision and
 *   formatting expectations in upstream Coincheck responses.
 * - The implementation constructs the signature with the full URL string (API_BASE + path) and an
 *   empty body for GET endpoints; for POST endpoints that send a form body, the body string is
 *   included in the signature computation.
 * - Do not log or persist apiSecret; treat credentials as sensitive.
 *
 * @param {string} apiKey - Coincheck API key (ACCESS-KEY).
 * @param {string} apiSecret - Coincheck API secret used to compute HMAC-SHA256 signatures.
 * @returns {PrivateApi} A PrivateApi-compatible client exposing get_info2, active_orders, 
 *          trade_history, trade, and cancel_order methods.
 *
 * @throws {Error} When required inputs are invalid (e.g., invalid trade action) or when Coincheck
 *                 indicates a failure for operations that must return data (trade, cancel, get_info2).
 *
 * @example
 * const client = createCoincheckPrivate(process.env.CC_KEY!, process.env.CC_SECRET!);
 * const info = await client.get_info2();
 */
export function createCoincheckPrivate(apiKey: string, apiSecret: string): PrivateApi {
    async function fetchBalance(): Promise<GetInfo2Response> {
        const path = '/api/accounts/balance';
        const url = API_BASE + path;
        const body = '';
        const nonce = Date.now().toString();
        const message = nonce + url + body;
        const signature = crypto.createHmac('sha256', apiSecret).update(message).digest('hex');
        const headers: Record<string, string> = {
            'ACCESS-KEY': apiKey,
            'ACCESS-NONCE': nonce,
            'ACCESS-SIGNATURE': signature,
            'Content-Type': 'application/x-www-form-urlencoded',
        };
        const res = await axios.get(url, { headers, timeout: Number(process.env.CC_TIMEOUT_MS || 10000) });
        const data: CoincheckBalanceResponse = res.data;
        if (!data || data.success === false) throw new Error('Coincheck balance fetch failed');
        const funds: Record<string, number> = {};
        // Normalize known keys and numeric balance-like fields
        for (const [k, v] of Object.entries(data)) {
            if (typeof v === 'string' || typeof v === 'number') {
                const num = Number(v);
                if (!Number.isNaN(num)) funds[k] = num;
            }
        }
        // 返却値の構造: funds（資産情報）, rights（権限情報）, open_orders（未約定注文数）, server_time（サーバー時刻）
        return { success: 1, return: { funds, rights: { info: true, trade: true }, open_orders: 0, server_time: Math.floor(Date.now() / 1000) } };
    }

    return {
        async get_info2() { return fetchBalance(); },
        async active_orders(): Promise<ActiveOrder[]> {
            const path = '/api/exchange/orders/opens';
            const url = API_BASE + path;
            const body = '';
            const nonce = Date.now().toString();
            const message = nonce + url + body;
            const signature = crypto.createHmac('sha256', apiSecret).update(message).digest('hex');
            const headers: Record<string, string> = { 'ACCESS-KEY': apiKey, 'ACCESS-NONCE': nonce, 'ACCESS-SIGNATURE': signature, 'Content-Type': 'application/x-www-form-urlencoded' };
            const res = await axios.get<CoincheckOpenOrdersResponse>(url, { headers, timeout: Number(process.env.CC_TIMEOUT_MS || 10000) });
            const data = res.data;
            if (!data?.success) return [];
            return (data.orders || []).map(o => ({ order_id: String(o.id), pair: 'btc_jpy', side: o.order_type === 'buy' ? 'bid' : 'ask', price: Number(o.rate), amount: Number(o.pending_amount), timestamp: Math.floor(new Date(o.created_at).getTime() / 1000) }));
        },
        async trade_history(): Promise<TradeHistoryRecord[]> {
            const path = '/api/exchange/orders/transactions';
            const url = API_BASE + path;
            const body = '';
            const nonce = Date.now().toString();
            const message = nonce + url + body;
            const signature = crypto.createHmac('sha256', apiSecret).update(message).digest('hex');
            const headers: Record<string, string> = { 'ACCESS-KEY': apiKey, 'ACCESS-NONCE': nonce, 'ACCESS-SIGNATURE': signature, 'Content-Type': 'application/x-www-form-urlencoded' };
            const res = await axios.get<CoincheckTradeHistoryResponse>(url, { headers, timeout: Number(process.env.CC_TIMEOUT_MS || 10000) });
            const data = res.data;
            if (!data?.success) return [];
            return (data.trades || []).map(t => ({ tid: t.id, order_id: String(t.order_id), side: t.order_type === 'buy' ? 'bid' : 'ask', price: Number(t.rate), amount: Number(t.amount), timestamp: Math.floor(new Date(t.created_at).getTime() / 1000) }));
        },
        async trade(p: any): Promise<TradeResult> {
            // Map from Zaif-style to Coincheck
            if (p.action !== 'bid' && p.action !== 'ask') {
                throw new Error(`Invalid action: ${p.action}. Must be 'bid' or 'ask'.`);
            }
            const pair = String(p.currency_pair || 'btc_jpy');
            const side = (p.action === 'bid' ? 'buy' : 'sell');
            const rateVal = (p.price ?? p.limitPrice);
            const hasRate = !(rateVal == null || rateVal === '' || Number.isNaN(Number(rateVal)));
            const path = '/api/exchange/orders';
            const url = API_BASE + path;
            let body: string;
            if (hasRate) {
                // Limit order
                const req: CoincheckOrderRequest = {
                    pair,
                    order_type: side,
                    rate: Number(rateVal),
                    amount: Number(p.amount)
                };
                body = new URLSearchParams({ pair: req.pair, order_type: req.order_type, rate: String(req.rate), amount: String(req.amount) }).toString();
            } else {
                // Market order
                if (side === 'buy') {
                    // market_buy uses JPY notional as market_buy_amount
                    const notional = Number(p.notional || p.market_notional || 0);
                    if (!(notional > 0)) throw new Error('market_buy requires positive notional (JPY)');
                    body = new URLSearchParams({ pair, order_type: 'market_buy', market_buy_amount: String(notional) }).toString();
                } else {
                    // market_sell uses base amount
                    const amt = Number(p.amount);
                    if (!(amt > 0)) throw new Error('market_sell requires positive amount');
                    body = new URLSearchParams({ pair, order_type: 'market_sell', amount: String(amt) }).toString();
                }
            }
            const nonce = Date.now().toString();
            const message = nonce + url + body;
            const signature = crypto.createHmac('sha256', apiSecret).update(message).digest('hex');
            const headers: Record<string, string> = { 'ACCESS-KEY': apiKey, 'ACCESS-NONCE': nonce, 'ACCESS-SIGNATURE': signature, 'Content-Type': 'application/x-www-form-urlencoded' };
            const res = await axios.post<CoincheckOrderResponse>(url, body, { headers, timeout: Number(process.env.CC_TIMEOUT_MS || 10000) });
            const data = res.data;
            if (!data?.success) throw new Error('Coincheck order failed');
            return { success: 1, return: { order_id: String(data.id) } };
        },
        async cancel_order(p: { order_id: string }): Promise<CancelResult> {
            const id = String(p.order_id);
            const path = `/api/exchange/orders/${id}/cancel`;
            const url = API_BASE + path;
            const body = '';
            const nonce = Date.now().toString();
            const message = nonce + url + body;
            const signature = crypto.createHmac('sha256', apiSecret).update(message).digest('hex');
            const headers: Record<string, string> = { 'ACCESS-KEY': apiKey, 'ACCESS-NONCE': nonce, 'ACCESS-SIGNATURE': signature, 'Content-Type': 'application/x-www-form-urlencoded' };
            const res = await axios.post<CoincheckCancelResponse>(url, body, { headers, timeout: Number(process.env.CC_TIMEOUT_MS || 10000) });
            const data = res.data;
            if (!data?.success) throw new Error('Coincheck cancel failed');
            return { success: 1, return: { order_id: id } };
        },
    };
}
