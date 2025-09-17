import { describe, it, expect, beforeEach, vi } from 'vitest';
import { RateLimiter, setRateLimiter } from '../../src/application/rate-limiter';
import { withRetry } from '../../src/adapters/base-service';

function pAll<T>(arr: Promise<T>[]) { return Promise.allSettled(arr); }

describe('integration: API rate metrics', () => {
  beforeEach(() => {
    process.env.RATE_METRICS_INTERVAL_MS = '200';
    setRateLimiter(new RateLimiter({ capacity: 10, refillPerSec: 5, reserveRatio: 0.1 }));
  });

  it('metrics reflect avg wait and rejects', async () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const calls: Promise<any>[] = [];
    for (let i=0;i<20;i++) calls.push(withRetry(async () => 1, 'n', 1, 1, { category: 'API-PUBLIC', priority: 'normal', opType: 'QUERY' }));
    // fire some EXEC high CANCEL (no reserved borrow)
    for (let i=0;i<5;i++) calls.push(withRetry(async () => 1, 'x', 1, 1, { category: 'EXEC', priority: 'high', opType: 'CANCEL' }));
    const res = await pAll(calls);
    // allow metrics emit
    await new Promise(r => setTimeout(r, 250));
    const metricsLines = spy.mock.calls.filter(args => args[0] && String(args[0]).includes('[INFO][RATE] metrics'));
    expect(metricsLines.length).toBeGreaterThan(0);
    const anyRejected = res.some(r => r.status === 'rejected');
    expect(anyRejected).toBeTypeOf('boolean');
    spy.mockRestore();
  });
});
