import fs from "fs";
import path from "path";
import os from "os";
import { PrivateApi, GetInfo2Response, ActiveOrder, TradeHistoryRecord, TradeResult, CancelResult } from "../types/private";

interface MockOrder { id: string; currency_pair: string; side: string; price?: number; amount: number; filled: number; status: "open" | "filled" | "canceled"; created_at: number; fills: Array<{ tid: number; price: number; amount: number; ts: number }>; firstFillDone: boolean; lastSimulateTs: number; }
interface MockState { orders: MockOrder[]; }
const ORDERS_PATH = process.env.MOCK_ORDERS_PATH || path.join(os.tmpdir(), "zaif-orders.json");
const VERBOSE = process.env.DEBUG_MOCK_VERBOSE === '1';
function logMock(...a: any[]) { if (VERBOSE) console.log(...a); }
const balancesEnv = (() => { try { return JSON.parse(process.env.MOCK_BALANCES_JSON || '{}'); } catch { return {}; } })();
const balances: Record<string, number> = Object.keys(balancesEnv).length ? balancesEnv : { jpy: 100000, btc: 0.01 };

/**
 * Ensures that the directory for the orders file exists.
 */
function ensureDir() {
	try { fs.mkdirSync(path.dirname(ORDERS_PATH), { recursive: true }); } catch { }
}

/**
 * Loads the mock state from the designated file.
 * If the file does not exist or is invalid, returns an empty state.
 * @returns {MockState} The loaded mock state.
 */
function loadState(): MockState {
	try {
		if (fs.existsSync(ORDERS_PATH)) {
			const raw = fs.readFileSync(ORDERS_PATH, 'utf8');
			const data = JSON.parse(raw || "{}");
			const orders: MockOrder[] = (data.orders || []).map((o: any) => ({
				...o, id: String(o.id), currency_pair: String(o.currency_pair || o.pair || 'btc_jpy'), side: String(o.side), price: Number(o.price ?? 0), amount: Number(o.amount), filled: Number(o.filled || 0), created_at: Number(o.created_at || Date.now()), firstFillDone: Boolean(o.firstFillDone), lastSimulateTs: Number(o.lastSimulateTs || 0), status: (o.status === 'filled' || o.status === 'canceled') ? o.status : 'open', fills: Array.isArray(o.fills) ? o.fills.map((f: any) => ({
					tid: Number(f.tid), price: Number(f.price), amount: Number(f.amount), ts: Number(f.ts)
				})) : []
			}));
			return { orders };
		}
	} catch { } return { orders: [] };
}

/**
 * Saves the current mock state to the designated file.
 * @param {MockState} st The mock state to save.
 */
function saveState(st: MockState) {
	try {
		ensureDir();
		const tmp = ORDERS_PATH + '.tmp';
		fs.writeFileSync(tmp, JSON.stringify(st, null, 2), 'utf8');
		fs.renameSync(tmp, ORDERS_PATH);
	} catch { }
}

/**
 * Simulates order fills for the current mock state.
 * @param {MockState} st The current mock state containing orders.
 */
function simulateFills(st: MockState) {
	const now = Date.now();
	for (const o of st.orders) {
		if (o.status !== 'open') continue;
		if (o.firstFillDone) continue;
		const exitDelay = Number(process.env.MOCK_EXIT_DELAY_MS || '1000');
		const forceExitDyn = process.env.MOCK_FORCE_EXIT === '1';
		const ratioDyn = Math.min(Number(process.env.MOCK_PARTIAL_FILL_RATIO || '0.4'), 0.9);
		const minPartialDyn = Number(process.env.MOCK_MIN_PARTIAL || '0.00001');
		if (now - o.lastSimulateTs >= exitDelay) {
			o.lastSimulateTs = now;
			const remaining = o.amount - o.filled;
			if (remaining > 0) {
				const force = forceExitDyn && ((o.side === 'ask' && (balances.btc || 0) > 0) || (o.side === 'bid' && (balances.jpy || 0) > 0));
				const ratio = force ? ratioDyn : ratioDyn;
				const fillAmt = Math.min(remaining, Math.max(o.amount * ratio, minPartialDyn));
				if (fillAmt > 0) {
					o.filled += fillAmt;
					const price = o.price || 0;
					const tid = now + Math.floor(Math.random() * 1000);
					o.fills.push({ tid, price, amount: fillAmt, ts: Math.floor(now / 1000) });
					if (o.filled >= o.amount) {
						o.status = 'filled';
						settleBalance(o, price);
					}
				}
			}
		}
	}
}

/**
 * Settles the balance for a completed order.
 * @param {MockOrder} o The mock order that has been filled.
 * @param {number} price The price at which the order was filled.
 */
function settleBalance(o: MockOrder, price: number) {
	if (o.side === 'bid') {
		const cost = price * o.amount;
		balances.jpy = Math.max(0, (balances.jpy || 0) - cost);
		balances.btc = (balances.btc || 0) + o.amount;
	} else {
		balances.btc = Math.max(0, (balances.btc || 0) - o.amount);
		balances.jpy = (balances.jpy || 0) + price * o.amount;
	}
}

/**
 * Creates a mock implementation of the private API used by the trading bot.
 * This mock simulates order placement, fills, cancellations, and maintains balances.
 * It reads and writes state to a JSON file specified by the MOCK_ORDERS_PATH environment variable.
 * Balances can be initialized via the MOCK_BALANCES_JSON environment variable.
 * @example
 * // Initialize mock with specific balances
 * process.env.MOCK_BALANCES_JSON = JSON.stringify({ jpy: 500000, btc: 0.05 });
 */
export function createPrivateMock(): PrivateApi {
	console.log('[MOCK] state path:', ORDERS_PATH); return {
		async get_info2(): Promise<GetInfo2Response> {
			const st = loadState();
			return {
				success: 1, return: {
					funds: balances, rights: { info: true, trade: true }, open_orders: st.orders.filter(o => o.status === 'open').length, server_time: Math.floor(Date.now() / 1000)
				}
			};
		}, async trade(params: any): Promise<TradeResult> {
			logMock('[MOCK] trade', params); const st = loadState();
			let nextNum = 1;
			if (st.orders.length) {
				const nums = st.orders.map(o => Number(o.id)).filter(n => !isNaN(n));
				if (nums.length) nextNum = Math.max(...nums) + 1;
			}
			const id = String(nextNum);
			const order: MockOrder = { id, currency_pair: params.currency_pair, side: params.side || params.action, price: params.limitPrice || params.price, amount: params.amount, filled: 0, status: 'open', created_at: Date.now(), fills: [], firstFillDone: false, lastSimulateTs: 0 };
			const forceImm = process.env.MOCK_FORCE_IMMEDIATE_FILL === '1';
			const forceExit = process.env.MOCK_FORCE_EXIT === '1';
			const ratio = Math.min(Number(process.env.MOCK_PARTIAL_FILL_RATIO || '0.4'), 0.9);
			const minPartial = Number(process.env.MOCK_MIN_PARTIAL || '0.00001');
			const EPS = 1e-12;
			if (forceImm || forceExit) {
				const remaining0 = order.amount - order.filled;
				const target = Math.max(order.amount * ratio, minPartial);
				const fillQty = Math.min(remaining0, target);
				if (fillQty > EPS) {
					order.filled += fillQty;
					const price = order.price ?? 0;
					const tid = Date.now();
					order.fills.push({ tid, price, amount: fillQty, ts: Math.floor(Date.now() / 1000) });
					order.firstFillDone = true;
					if (order.filled >= order.amount - EPS) {
						order.status = 'filled';
						settleBalance(order, price);
					}
				}
			}
			st.orders.push(order); saveState(st);
			return { success: 1, return: { order_id: String(id) } };
		}, async active_orders(): Promise<ActiveOrder[]> {
			const st = loadState(); simulateFills(st);
			const forceExit = process.env.MOCK_FORCE_EXIT === '1';
			const ratio = Math.min(Number(process.env.MOCK_PARTIAL_FILL_RATIO || '0.4'), 0.9);
			const minPartial = Number(process.env.MOCK_MIN_PARTIAL || '0.00001');
			const now = Date.now();
			if (forceExit) {
				for (const o of st.orders) {
					if (o.status !== 'open' || o.firstFillDone) continue;
					const remaining = o.amount - o.filled;
					if (remaining > 0) {
						const fillQty = Math.min(remaining, Math.max(o.amount * ratio, minPartial));
						if (fillQty > 0) {
							o.filled += fillQty;
							const price = o.price || 0;
							const tid = now + Math.floor(Math.random() * 1000);
							o.fills.push({ tid, price, amount: fillQty, ts: Math.floor(now / 1000) });
							o.firstFillDone = true;
							if (o.filled >= o.amount) {
								o.status = 'filled'; settleBalance(o, price);
							}
						}
					}
				}
			} saveState(st);
			return st.orders.filter(o => o.status !== 'canceled' && o.filled < o.amount - 1e-12).map(o => ({
				order_id: String(o.id), pair: o.currency_pair, side: o.side as 'bid' | 'ask', price: Number(o.price || 0), amount: Number(o.amount - o.filled), filled: Number(o.filled), timestamp: Math.floor(o.created_at / 1000)
			}));
		}, async cancel_order(params: any): Promise<CancelResult> {
			const st = loadState();
			const o = st.orders.find(o => o.id == params.order_id);
			if (o && o.status === 'open') o.status = 'canceled';
			saveState(st);
			return { success: 1, return: { order_id: String(params.order_id) } };
		}, async trade_history(): Promise<TradeHistoryRecord[]> {
			const st = loadState();
			const fills = st.orders.flatMap(o => o.fills.map(f => ({ date: f.ts, price: f.price, amount: f.amount, tid: f.tid, currency_pair: o.currency_pair, trade_type: o.side, order_id: o.id })));
			return fills.map(f => ({ tid: f.tid, order_id: String(f.order_id), side: f.trade_type as 'bid' | 'ask', price: f.price, amount: f.amount, timestamp: f.date }));
		}, async healthCheck() {
			return { ok: true };
		}, async testGetInfo2() {
			return { ok: true, httpStatus: 200, successFlag: 1 };
		}
	};
}