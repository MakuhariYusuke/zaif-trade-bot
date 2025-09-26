import { describe, it, beforeEach, vi, expect, afterEach } from 'vitest';
import path from 'path';

// Spy logger
const spies = { warn: vi.fn() };
vi.mock('../../../ztb/utils/logger', () => ({
  logInfo: (..._args: any[]) => {},
  logError: (..._args: any[]) => {},
  logWarn: (...args: any[]) => { spies.warn(...args); },
}));

// Mocks
interface MockApi {
  trade: (p: any) => Promise<{ return: { order_id: string } }>;
  cancel_order: (p: any) => Promise<{ return: { order_id: string } }>;
  trade_history: () => Promise<any[]>;
  get_info2: () => Promise<{ success: number; return: { funds: { jpy: number; eth: number } } }>;
}
const calls: { trade: any[]; cancel: any[]; hist: any[]; get_info2: any[] } = { trade: [], cancel: [], hist: [], get_info2: [] };
const mockApi: MockApi = {
  trade: vi.fn(async (p: any) => { calls.trade.push(p); return { return: { order_id: 'OID2' } }; }),
  cancel_order: vi.fn(async (p: any) => { calls.cancel.push(p); return { return: { order_id: p.order_id } }; }),
  trade_history: vi.fn(async () => { calls.hist.push(1); return []; }),
  get_info2: vi.fn(async () => { calls.get_info2.push(1); return { success: 1, return: { funds: { jpy: 100000, eth: 10 } } }; }),
};
vi.mock('../../../ztb/api/adapters', () => ({ createPrivateApi: () => mockApi }));
vi.mock('../../../ztb/api/public', () => ({
  getOrderBook: vi.fn(async () => ({ bids: [[999, 1]], asks: [[1001, 1]] })),
  getTrades: vi.fn(async () => ([{ price: 1000, amount: 0.1, date: Math.floor(Date.now()/1000) }])),
}));

describe('live minimal safety warn', () => {
  const envBk = { ...process.env };
  const TMP = path.resolve(process.cwd(), 'tmp-live-min-warn');
  beforeEach(()=>{
    vi.resetModules();
    Object.keys(calls).forEach(k=> (calls as any)[k] = []);
    spies.warn.mockClear();
    process.env = { ...envBk };
    process.env.POSITION_STORE_DIR = path.join(TMP, 'positions');
    process.env.STATS_DIR = path.join(TMP, 'logs');
    process.env.FEATURES_LOG_DIR = path.join(TMP, 'features');
    process.env.EXCHANGE = 'zaif';
    process.env.DRY_RUN = '0';
    process.env.PAIR = 'eth_jpy';
    process.env.TRADE_FLOW = 'SELL_ONLY';
    process.env.TEST_FLOW_QTY = '2';
    process.env.ORDER_TYPE = 'limit';
    process.env.TEST_FLOW_RATE = '1000';
    process.env.SAFETY_MODE = '1';
    process.env.EXPOSURE_WARN_PCT = '0.05';
  });
  afterEach(()=>{ process.env = { ...envBk }; });

  it('emits WARN when exposure exceeds threshold (after clamp still >5%)', async () => {
    await import('../../../ztb/tools/live/test-minimal-live');
    // Wait until trade and cancel have been called, or timeout after 1s
    await new Promise((resolve, reject) => {
      const start = Date.now();
      (function waitForCalls() {
        if (calls.trade.length > 0 && calls.cancel.length > 0) return resolve(undefined);
        if (Date.now() - start > 1000) return reject(new Error('Timeout waiting for trade/cancel calls'));
        setTimeout(waitForCalls, 10);
      })();
    });
    expect(calls.trade.length).toBeGreaterThan(0);
    expect(calls.cancel.length).toBeGreaterThan(0);
    // warn should be emitted for ask exposure (10% > 5%) and safety clamp message should appear
    expect(spies.warn).toHaveBeenCalled();
    const msgs = spies.warn.mock.calls.map((c:any[])=> c.join(' '));
    expect(msgs.some((m:string)=> m.includes('[SAFETY] amount clamped'))).toBe(true);
  });
});
