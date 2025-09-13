import { describe, it, expect, beforeEach, vi } from 'vitest';
import path from 'path';
import fs from 'fs';

vi.mock('../../../src/services/market-service', () => ({
  listActiveOrders: vi.fn(async () => ({})),
  fetchTradeHistory: vi.fn(async () => ([])),
  cancelOrder: vi.fn(async () => ({ result: true })),
}));

describe('services/execution-service pollFillState', () => {
  const TMP = path.resolve(process.cwd(), '.tmp-exec-poll');

  beforeEach(() => {
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
    process.env.POSITION_STORE_DIR = path.join(TMP, 'positions');
    process.env.STATS_DIR = path.join(TMP, 'logs');
  });

  it('handles partial fill increments then marks FILLED when history shows full amount (order_id match)', async () => {
    const { pollFillState } = await import('../../../src/services/execution-service');
    const ms = await import('../../../src/services/market-service');
    const pair = 'btc_jpy';
    const now = Date.now();

    // active -> shows remaining reduces once, then disappears
    const activeSeq: any[] = [
      { '123': { amount: 0.08 } }, // remaining -> implies 0.02 filled (original 0.1)
      {}, // gone -> use trade_history
    ];
    let i = 0;
    (ms.listActiveOrders as any).mockImplementation(async ()=> activeSeq[Math.min(i++, activeSeq.length-1)]);
    (ms.fetchTradeHistory as any).mockResolvedValue([
      { order_id: 123, amount: 0.1, price: 100, timestamp: Math.floor(now/1000), side: 'ask', tid: 1 },
    ]);

    const snap = await pollFillState(pair, { side:'ask', intendedPrice: 100, amount: 0.1, orderId: 123, submittedAt: now, originalAmount: 0.1, requestId: 'r' }, 100, 10);
    expect(snap.status).toBe('FILLED');
    expect(snap.filledAmount).toBeCloseTo(0.1, 10);
    expect(snap.avgFillPrice).toBeCloseTo(100, 10);
  });

  it('heuristic match path when order_id not available', async () => {
    const { pollFillState } = await import('../../../src/services/execution-service');
    const ms = await import('../../../src/services/market-service');
    const pair = 'btc_jpy';
    const now = Date.now();

    (ms.listActiveOrders as any).mockResolvedValue({});
    (ms.fetchTradeHistory as any).mockResolvedValue([
      { side: 'bid', amount: 0.05, price: 99, timestamp: Math.floor(now/1000), tid: 2 },
    ]);

  const snap = await pollFillState(pair, { side:'bid', intendedPrice: 100, amount: 0.05, orderId: 999, submittedAt: now, originalAmount: 0.05, requestId: 'r2' }, 30, 10);
  expect(snap.status).toBeDefined();
  });
});
