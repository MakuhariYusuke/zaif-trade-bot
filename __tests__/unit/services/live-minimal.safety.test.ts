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
  trade: vi.fn(async (p: any) => { calls.trade.push(p); return { return: { order_id: 'OID3' } }; }),
  cancel_order: vi.fn(async (p: any) => { calls.cancel.push(p); return { return: { order_id: p.order_id } }; }),
  trade_history: vi.fn(async () => { calls.hist.push(1); return []; }),
  get_info2: vi.fn(async () => { calls.get_info2.push(1); return { success: 1, return: { funds: { jpy: 10000, btc: 0.5 } } }; }),
};
vi.mock('../../../src/api/adapters', () => ({ createPrivateApi: () => mockApi }));
vi.mock('../../../src/api/public', () => ({
  getOrderBook: vi.fn(async () => ({ bids: [[2000, 1]], asks: [[2001, 1]] })),
  getTrades: vi.fn(async () => ([{ price: 2000, amount: 0.1, date: Math.floor(Date.now()/1000) }])),
}));

describe('live minimal safety clamp (bid, JPY-based)', () => {
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
    process.env.PAIR = 'btc_jpy';
    process.env.TRADE_FLOW = 'BUY_ONLY';
    // qty=1 at price 2000 JPY => notional 2000 JPY (20% of 10,000), should clamp to 10%
    process.env.TEST_FLOW_QTY = '1';
    process.env.ORDER_TYPE = 'limit';
    process.env.TEST_FLOW_RATE = '2000';
    process.env.SAFETY_MODE = '1';
  });
  afterEach(()=>{ process.env = { ...envBk }; });

  it('logs [SAFETY] amount clamped for bid with JPY clamp', async () => {
    await import('../../../src/tools/live/test-minimal-live');
    await new Promise(r=>setTimeout(r, 15));
    expect(calls.trade.length).toBeGreaterThan(0);
    const msgs = spies.warn.mock.calls.map((c:any[])=> c.join(' '));
    expect(msgs.some((m:string)=> m.includes('[SAFETY] amount clamped') && m.includes('side=bid'))).toBe(true);
  });
});
