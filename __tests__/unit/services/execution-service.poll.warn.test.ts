import { describe, it, expect, vi, beforeEach } from 'vitest';
import { setupJsonLogs, captureLogs, expectJsonLog } from '../helpers/logging';

vi.mock('../../../src/adapters/market-service', () => ({
  listActiveOrders: vi.fn(async () => ({ '123': { amount: 0.1 } })),
  fetchTradeHistory: vi.fn(async () => ([])),
  cancelOrder: vi.fn(async () => ({ result: true })),
}));

describe('adapters/execution-service pollFillState slow polling WARN', () => {
  beforeEach(() => { vi.resetModules(); setupJsonLogs(); vi.useFakeTimers(); });

  it('emits EXEC WARN when polling exceeds 30s', async () => {
    const logs = captureLogs();
    const { pollFillState } = await import('../../../src/adapters/execution-service');
    const pair = 'btc_jpy';

    const p = { side: 'bid' as const, intendedPrice: 100, amount: 0.1, orderId: 123, submittedAt: Date.now(), originalAmount: 0.1, requestId: 'r' };
    const pr = pollFillState(pair, p as any, 35000, 1000);
    // 35s 経過させる
    await vi.advanceTimersByTimeAsync(35000);
    const res = await pr;
    expect(res.status).toBe('EXPIRED');
    expectJsonLog(logs, 'EXEC', 'WARN', 'polling slow', ['elapsedMs','pair','side','amount','price']);
  });
});
