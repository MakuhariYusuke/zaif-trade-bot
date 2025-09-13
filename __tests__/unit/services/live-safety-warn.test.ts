import { describe, it, beforeEach, vi, expect, afterEach } from 'vitest';
import path from 'path';
import fs from 'fs';

// Spy logger
const spies = { warn: vi.fn() };
vi.mock('../../../src/utils/logger', () => ({
  logInfo: (..._args: any[]) => {},
  logError: (..._args: any[]) => {},
  logWarn: (...args: any[]) => { spies.warn(...args); },
}));

// Mocks
const calls: any = { trade: [], cancel: [], hist: [], get_info2: [] };
const mockApi: any = {
  trade: vi.fn(async (p: any) => { calls.trade.push(p); return { return: { order_id: 'OID2' } }; }),
  cancel_order: vi.fn(async (p: any) => { calls.cancel.push(p); return { return: { order_id: p.order_id } }; }),
  trade_history: vi.fn(async () => { calls.hist.push(1); return []; }),
  get_info2: vi.fn(async () => { calls.get_info2.push(1); return { success: 1, return: { funds: { jpy: 100000, eth: 10 } } }; }),
};
vi.mock('../../../src/api/adapters', () => ({ createPrivateApi: () => mockApi }));
vi.mock('../../../src/api/public', () => ({
  getOrderBook: vi.fn(async () => ({ bids: [[999, 1]], asks: [[1001, 1]] })),
  getTrades: vi.fn(async () => ([{ price: 1000, amount: 0.1, date: Math.floor(Date.now()/1000) }])),
}));

describe('live minimal safety warn', () => {
  const envBk = { ...process.env };
  const TMP = path.resolve(process.cwd(), 'tmp-live-min');
  beforeEach(()=>{
    vi.resetModules();
    Object.keys(calls).forEach(k=> (calls as any)[k] = []);
    spies.warn.mockClear();
    process.env = { ...envBk };
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
    process.env.POSITION_STORE_DIR = path.join(TMP, 'positions');
    process.env.STATS_DIR = path.join(TMP, 'logs');
    process.env.FEATURES_LOG_DIR = path.join(TMP, 'features');
    process.env.EXCHANGE = 'zaif';
    process.env.DRY_RUN = '0';
    process.env.PAIR = 'eth_jpy';
    process.env.TRADE_FLOW = 'SELL_ONLY';
    // qty above 10% of base (eth=10) -> 2.0 will be clamped to 1.0
    process.env.TEST_FLOW_QTY = '2.0';
    process.env.ORDER_TYPE = 'limit';
    process.env.TEST_FLOW_RATE = '1000';
    // Enable safety clamp and keep warn threshold at default 5%
    process.env.SAFETY_MODE = '1';
  });
  afterEach(()=>{ process.env = { ...envBk }; });

  it('emits WARN when exposure exceeds threshold (after clamp still >5%)', async () => {
    await import('../../../src/tools/live/test-minimal-live');
    // small wait for async
    await new Promise(r=>setTimeout(r, 10));
    expect(calls.trade.length).toBeGreaterThan(0);
    expect(calls.cancel.length).toBeGreaterThan(0);
  // warn should be emitted for ask exposure (10% > 5%) and safety clamp message should appear
  expect(spies.warn).toHaveBeenCalled();
  const msgs = spies.warn.mock.calls.map((c:any[])=> c.join(' '));
  expect(msgs.some((m:string)=> m.includes('[SAFETY] amount clamped'))).toBe(true);
  });
});
