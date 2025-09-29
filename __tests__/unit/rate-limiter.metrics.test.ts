import { describe, it, expect, beforeEach, vi } from 'vitest';
import { RateLimiter, setRateLimiter } from '../../ztb/application/rate-limiter';
import { withRetry } from '../../ztb/adapters/base-service';

describe('RateLimiter metrics + reserved policy', () => {
  beforeEach(() => {
    // fast metrics emission
    process.env.RATE_METRICS_INTERVAL_MS = '200';
    // clear env that might override buckets from other tests
    delete process.env.RATE_CAPACITY_PUBLIC;
    delete process.env.RATE_REFILL_PUBLIC;
    delete process.env.RATE_CAPACITY_PRIVATE;
    delete process.env.RATE_REFILL_PRIVATE;
    delete process.env.RATE_CAPACITY_EXEC;
    delete process.env.RATE_REFILL_EXEC;
    delete process.env.RATE_CAPACITY;
    delete process.env.RATE_LIMIT_CAPACITY;
    delete process.env.RATE_REFILL;
    delete process.env.RATE_REFILL_PER_SEC;
    delete process.env.RATE_RESERVE_RATIO;
    delete process.env.RATE_PRIORITY_RESERVE;
    // Small refill to keep buckets tight and predictable
    setRateLimiter(new RateLimiter({ capacity: 5, refillPerSec: 1, reserveRatio: 0.2 }));
  });

  it('emits RATE/METRICS and reserves only for ORDER', async () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
    // Drain near reserve boundary with normal (PUBLIC)
    const p: Promise<any>[] = [];
    for (let i=0;i<4;i++) p.push(withRetry(async () => 1, 'n', 1, 1, { category: 'API-PUBLIC', priority: 'normal', opType: 'QUERY', rateMaxWaitMs: 50 }));
    await Promise.allSettled(p);
    // Drain EXEC bucket down to reserved-only (capacity=5, reserved=floor(5*0.2)=1)
    for (let i = 0; i < 4; i++) {
      await withRetry(async () => 1, 'e', 1, 1, { category: 'EXEC', priority: 'normal', opType: 'QUERY', rateMaxWaitMs: 50 });
    }
    // CANCEL should not borrow reserved -> reject quickly
    await expect(withRetry(async () => 1, 'c', 1, 1, { category: 'EXEC', priority: 'high', opType: 'CANCEL', rateMaxWaitMs: 50 })).rejects.toBeTruthy();
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
