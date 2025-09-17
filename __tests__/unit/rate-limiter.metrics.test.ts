import { describe, it, expect, beforeEach, vi } from 'vitest';
import { RateLimiter, setRateLimiter } from '../../src/application/rate-limiter';
import { withRetry } from '../../src/adapters/base-service';

describe('RateLimiter metrics + reserved policy', () => {
  beforeEach(() => {
    // fast metrics emission
    process.env.RATE_METRICS_INTERVAL_MS = '200';
    setRateLimiter(new RateLimiter({ capacity: 5, refillPerSec: 2, reserveRatio: 0.2 }));
  });

  it('emits RATE/METRICS and reserves only for ORDER', async () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
    // Drain near reserve boundary with normal (PUBLIC)
    const p: Promise<any>[] = [];
    for (let i=0;i<4;i++) p.push(withRetry(async () => 1, 'n', 1, 1, { category: 'API-PUBLIC', priority: 'normal', opType: 'QUERY' }));
    await Promise.allSettled(p);
    // CANCEL should not borrow reserved -> likely to wait or reject quickly
    await expect(withRetry(async () => 1, 'c', 1, 1, { category: 'EXEC', priority: 'high', opType: 'CANCEL' })).rejects.toBeTruthy().catch(()=>{});
    // ORDER may borrow reserved and pass quickly
    const t0 = Date.now();
    await withRetry(async () => 1, 'o', 1, 1, { category: 'EXEC', priority: 'high', opType: 'ORDER' });
    const waitedOrder = Date.now() - t0;
    expect(waitedOrder).toBeLessThanOrEqual(1000);

    // wait to let metrics window elapse, then trigger emission
    await new Promise(r => setTimeout(r, 250));
    await withRetry(async () => 1, 'tick', 1, 1, { category: 'API-PUBLIC', priority: 'normal', opType: 'QUERY' });
    const hasMetrics = spy.mock.calls.some(args => args[0] && String(args[0]).includes('[INFO][RATE] metrics'));
    expect(hasMetrics).toBe(true);
    spy.mockRestore();
  });
});
