import fs from "fs";
import path from "path";
import os from "os";
import { PrivateApi, GetInfo2Response, ActiveOrder, TradeHistoryRecord, TradeResult, CancelResult } from "../types/private";

interface MockOrder {
  id: string;
  currency_pair: string;
  side: string;
  price?: number;
  amount: number;
  filled: number;
  status: "open" | "filled" | "canceled";
  created_at: number; // ms
  fills: Array<{ tid: number; price: number; amount: number; ts: number }>;
  firstFillDone: boolean;
  lastSimulateTs: number;
}

interface MockState {
  orders: MockOrder[];
}

const ORDERS_PATH = process.env.MOCK_ORDERS_PATH || path.join(os.tmpdir(), "zaif-orders.json");
const VERBOSE = process.env.DEBUG_MOCK_VERBOSE === '1';
/** Logs mock API calls */
function logMock(...args: any[]) { if (VERBOSE) console.log(...args); }
const balancesEnv = (() => { try { return JSON.parse(process.env.MOCK_BALANCES_JSON || '{}'); } catch { return {}; } })();
const balances: Record<string, number> = Object.keys(balancesEnv).length ? balancesEnv : { jpy: 100000, btc: 0.01 } ;

/** Ensures that the directory for the orders file exists. */
function ensureDir() { try { fs.mkdirSync(path.dirname(ORDERS_PATH), { recursive: true }); } catch { } }

/**
 * Loads the mock state from the designated file.
 * If the file does not exist or is invalid, returns an empty state.
 * @returns {MockState} The loaded mock state.
 */
function loadState(): MockState {
  try {
    if (fs.existsSync(ORDERS_PATH)) {
      const raw = fs.readFileSync(ORDERS_PATH, "utf8");
      const data = JSON.parse(raw || "{}");
      const orders: MockOrder[] = (data.orders || []).map((o: any) => ({
        ...o,
        id: String(o.id),
        currency_pair: String(o.currency_pair || o.pair || 'btc_jpy'),
        side: String(o.side),
        price: Number(o.price ?? 0),
        amount: Number(o.amount),
        filled: Number(o.filled || 0),
        created_at: Number(o.created_at || Date.now()),
        firstFillDone: Boolean(o.firstFillDone),
        lastSimulateTs: Number(o.lastSimulateTs || 0),
        status: (o.status === 'filled' || o.status === 'canceled') ? o.status : 'open',
        fills: Array.isArray(o.fills) ? o.fills.map((f: any) => ({ tid: Number(f.tid), price: Number(f.price), amount: Number(f.amount), ts: Number(f.ts) })) : []
      }));
      return { orders };
    }
  } catch { }
  return { orders: [] };
}

/**
 * Saves the current mock state to the designated file.
 * @param {MockState} st The current mock state to save.
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
    if (o.firstFillDone) continue; // avoid cascading multiple fills in first tick
    const exitDelay = Number(process.env.MOCK_EXIT_DELAY_MS || '1000');
    const forceExitDyn = process.env.MOCK_FORCE_EXIT === '1';
    const ratioDyn = Math.min(Number(process.env.MOCK_PARTIAL_FILL_RATIO || '0.4'), 0.9);
    const minPartialDyn = Number(process.env.MOCK_MIN_PARTIAL || '0.00001');
    // periodic background progression (non-guaranteed)
    if (now - o.lastSimulateTs >= exitDelay) {
      o.lastSimulateTs = now;
      const remaining = o.amount - o.filled;
      if (remaining > 0) {
        const force = forceExitDyn && ((o.side === 'ask' && (balances.btc || 0) > 0) || (o.side === 'bid' && (balances.jpy || 0) > 0));
        const ratio = force ? ratioDyn : ratioDyn; // keep same but placeholder if later differentiation
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
 * Settles the balances based on the completed order.
 * @param {MockOrder} o - The completed order.
 * @param {number} price - The price at which the order was filled.
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
 *
 * This factory returns an object that implements the same methods as the production
 * PrivateApi, but operates against an in-memory / persisted test state (via loadState/saveState)
 * and simulates order execution and fills. The mock is intended for local development,
 * testing, and integration scenarios where communicating with the real exchange is
 * undesirable or impossible.
 *
 * Behavior summary:
 * - Logs the mock state path on creation (console output).
 * - Persists and loads state using loadState/saveState. The state contains an `orders` list
 *   and balances used to answer info requests and to update on fills.
 * - Generates numeric order IDs (stringified) by finding the current maximum numeric ID
 *   in state and incrementing it.
 * - Simulates immediate partial fills on order creation when the environment variable
 *   MOCK_FORCE_IMMEDIATE_FILL === '1' (or when MOCK_FORCE_EXIT === '1', see below).
 * - Simulates background fills when active_orders() is called (and when MOCK_FORCE_EXIT === '1'
 *   will force at least one partial fill for open orders that have not had a first fill).
 * - Uses configurable partial-fill behavior via the following environment variables:
 *   - MOCK_FORCE_IMMEDIATE_FILL: '1' to attempt an immediate partial fill inside trade().
 *   - MOCK_FORCE_EXIT: '1' to apply immediate-like fills both on trade() and active_orders().
 *   - MOCK_PARTIAL_FILL_RATIO: decimal fraction (default '0.4') used to calculate target partial fill.
 *     The effective ratio is capped at 0.9.
 *   - MOCK_MIN_PARTIAL: minimum quantity to use for partial fills (default '0.00001').
 * - Uses an EPS threshold (1e-12) to avoid tiny floating-point remainder issues when comparing
 *   filled vs amount.
 * - When an order is filled to (or above) its amount (within EPS), settleBalance(order, price)
 *   is invoked to adjust the mocked balances.
 *
 * Returned API (key methods and their behavior):
 * - get_info2(): Promise<GetInfo2Response>
 *   - Returns success=1 and a `return` object containing:
 *     - funds: balances (from the persisted state)
 *     - rights: { info: true, trade: true }
 *     - open_orders: number of orders with status === 'open'
 *     - server_time: current UNIX timestamp (seconds)
 *
 * - trade(params: any): Promise<TradeResult>
 *   - Creates a new MockOrder with fields like: id, currency_pair, side/action, price (limitPrice/price),
 *     amount, filled (initially 0 or partially filled if immediate fill applied), status ('open' or 'filled'),
 *     created_at (ms), fills (array), firstFillDone flag, lastSimulateTs.
 *   - If immediate fill behavior is enabled, performs a partial fill using the ratio/minPartial rules,
 *     appends a fill record (tid, price, amount, ts), marks firstFillDone, and if fully filled, sets status
 *     to 'filled' and calls settleBalance.
 *   - Persists the new order to state and returns { success: 1, return: { order_id: string } }.
 *
 * - active_orders(): Promise<ActiveOrder[]>
 *   - Loads state and runs background progression (simulateFills).
 *   - If MOCK_FORCE_EXIT === '1', additionally attempts to apply a partial fill for any open order
 *     that has not yet had a first fill.
 *   - Persists state and returns an array of active orders filtered to exclude canceled orders
 *     and orders already fully filled (uses a small tolerance).
 *   - Each returned ActiveOrder contains:
 *     - order_id: string
 *     - pair: currency pair string
 *     - side: 'bid'|'ask'
 *     - price: number (filled price or order price)
 *     - amount: remaining amount to fill (amount - filled)
 *     - filled: number (amount filled so far)
 *     - timestamp: order creation time in seconds
 *
 * - cancel_order(params: any): Promise<CancelResult>
 *   - Marks an order with the given params.order_id as status = 'canceled' if currently open,
 *     persists state, and returns { success: 1, return: { order_id: string } }.
 *
 * - trade_history(): Promise<TradeHistoryRecord[]>
 *   - Aggregates all fills across persisted orders and returns them mapped to the expected
 *     TradeHistoryRecord shape: { tid, order_id, side, price, amount, timestamp }.
 *
 * - healthCheck(): Promise<{ ok: true }>
 *   - Simple readiness/health probe returning { ok: true }.
 *
 * - testGetInfo2(): Promise<{ ok: true, httpStatus: 200, successFlag: 1 }>
 *   - Lightweight method intended for diagnostics/tests.
 *
 * Notes and implementation details:
 * - The mock relies on helper functions and state machinery defined elsewhere in the module:
 *   loadState(), saveState(), simulateFills(st), settleBalance(order, price), balances variable,
 *   ORDERS_PATH constant, and logMock() for logging. These are used to persist state and simulate
 *   realistic order lifecycle behavior.
 * - Fill records contain: tid (number), price (number), amount (number), ts (unix seconds).
 * - The mock uses timestamps in two forms: created_at in milliseconds and fill.ts in seconds.
 * - This mock is synchronous from the perspective of state updates, but methods are async
 *   to match the production API and to allow future asynchronous behavior.
 *
 * @returns {PrivateApi} A mocked implementation of PrivateApi exposing the methods described above.
 */
export function createPrivateMock(): PrivateApi {
  logMock("[MOCK] state path:", ORDERS_PATH);
  return {
    async get_info2(): Promise<GetInfo2Response> {
      const st = loadState();
      logMock("[MOCK] get_info2");
      return {
        success: 1,
        return: {
          funds: balances,
          rights: { info: true, trade: true },
          open_orders: st.orders.filter(o => o.status === 'open').length,
          server_time: Math.floor(Date.now() / 1000)
        }
      };
    },
    async trade(params: any): Promise<TradeResult> {
      logMock("[MOCK] trade", params);
      const st = loadState();
      let nextNum = 1;
      if (st.orders.length) {
        const nums = st.orders.map(o => Number(o.id)).filter(n => !isNaN(n));
        if (nums.length) nextNum = Math.max(...nums) + 1;
      }
      const id = String(nextNum);
      const order: MockOrder = { id, currency_pair: params.currency_pair, side: params.side || params.action, price: params.limitPrice || params.price, amount: params.amount, filled: 0, status: 'open', created_at: Date.now(), fills: [], firstFillDone: false, lastSimulateTs: 0 };
      // Optional immediate partial fill
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
          if (order.filled >= order.amount - EPS) { order.status = 'filled'; settleBalance(order, price); }
          logMock('[MOCK] immediate_fill', { 
            id: order.id, 
            fillQty, 
            filled: order.filled, 
            remaining: order.amount - order.filled 
          });
        }
      }
      st.orders.push(order);
      saveState(st); // persist initial
      return { success: 1, return: { order_id: String(id) } };
    },
    async active_orders(): Promise<ActiveOrder[]> {
      const st = loadState();
      // background progression
      simulateFills(st);
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
              if (o.filled >= o.amount) { o.status = 'filled'; settleBalance(o, price); }
            }
          }
        }
      }
      saveState(st);
      logMock("[MOCK] active_orders");
      return st.orders
        .filter(o => o.status !== 'canceled' && o.filled < o.amount - 1e-12)
        .map(o => ({
          order_id: String(o.id),
          pair: o.currency_pair,
          side: o.side as 'bid' | 'ask',
          price: Number(o.price || 0),
          amount: Number(o.amount - o.filled),
          filled: Number(o.filled),
          timestamp: Math.floor(o.created_at / 1000)
        }));
    },
    async cancel_order(params: any): Promise<CancelResult> {
      logMock("[MOCK] cancel_order", params);
      const st = loadState();
      const o = st.orders.find(o => o.id == params.order_id);
      if (o && o.status === 'open') o.status = 'canceled';
      saveState(st);
      return { success: 1, return: { order_id: String(params.order_id) } };
    },
    async trade_history(): Promise<TradeHistoryRecord[]> {
      const st = loadState();
      logMock("[MOCK] trade_history");
      const fills = st.orders.flatMap(o => o.fills.map(f => ({
        date: f.ts,
        price: f.price,
        amount: f.amount,
        tid: f.tid,
        currency_pair: o.currency_pair,
        trade_type: o.side,
        order_id: o.id
      })));
      return fills.map(f => ({ 
        tid: f.tid, 
        order_id: String(f.order_id), 
        side: f.trade_type as 'bid' | 'ask', 
        price: f.price, 
        amount: f.amount, 
        timestamp: f.date 
      }));
    },
    async healthCheck() { return { ok: true }; },
    async testGetInfo2() { return { ok: true, httpStatus: 200, successFlag: 1 }; }
  };
}