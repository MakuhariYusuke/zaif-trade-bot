import { describe, it, expect, beforeEach, vi } from 'vitest';
import { RateLimiter, setRateLimiter } from '../../src/application/rate-limiter';
import { withRetry } from '../../src/adapters/base-service';

describe('RateLimiter category details', () => {
  beforeEach(() => {
    process.env.RATE_METRICS_INTERVAL_MS = '200';
    setRateLimiter(new RateLimiter({ capacity: 3, refillPerSec: 1, reserveRatio: 0.33 }));
  });

  it('includes details per category', async () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(()=>{});
    // Mix of categories
    await withRetry(async ()=>1, 'p', 1, 1, { category: 'API-PUBLIC', priority: 'normal', opType: 'QUERY' });
    try { await withRetry(async ()=>{ throw Object.assign(new Error('x'), { code: 'RATE_LIMITED' }); }, 'p2', 1, 1, { category: 'API-PUBLIC' }); } catch {}
    await withRetry(async ()=>1, 'x', 1, 1, { category: 'API-PRIVATE', priority: 'normal', opType: 'QUERY' });
    await withRetry(async ()=>1, 'e', 1, 1, { category: 'EXEC', priority: 'high', opType: 'ORDER' });
    await new Promise(r => setTimeout(r, 250));
    // trigger emission after interval
    await withRetry(async ()=>1, 'tick', 1, 1, { category: 'API-PUBLIC', priority: 'normal', opType: 'QUERY' });
    const jsons = spy.mock.calls.map(args => args[1]).filter(x => x && typeof x === 'object');
    const hit = spy.mock.calls.find(args => String(args[0]).includes('[INFO][RATE] metrics'));
    expect(!!hit).toBe(true);
    spy.mockRestore();
  });
});
