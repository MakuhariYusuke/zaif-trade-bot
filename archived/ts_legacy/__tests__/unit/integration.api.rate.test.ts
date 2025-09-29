import { describe, it, expect, beforeEach } from 'vitest';
import { setRateLimiter, RateLimiter } from '../../ztb/application/rate-limiter';
import { withRetry } from '../../ztb/adapters/base-service';

function pAll<T>(arr: Promise<T>[]) { return Promise.allSettled(arr); }

describe('integration: API rate limiting', () => {
  beforeEach(() => {
    setRateLimiter(new RateLimiter({ capacity: 10, refillPerSec: 10, reserveRatio: 0.1 }));
  });

  it('queues within ~1s window; high priority acquires quickly', async () => {
    const start = Date.now();
    const calls: Promise<any>[] = [];
    // 12 normal, 3 high
    for (let i=0;i<12;i++) calls.push(withRetry(async () => 1, 'n', 1, 1, { category: 'API-PUBLIC', priority: 'normal' }));
    for (let i=0;i<3;i++) calls.push(withRetry(async () => 1, 'h', 1, 1, { category: 'EXEC', priority: 'high' }));
    const res = await pAll(calls);
    const took = Date.now() - start;
    // 1秒以内に全完了し、いくつかは待機（WARN相当）していることを想定
    const fulfilled = res.filter(r => r.status === 'fulfilled').length;
    expect(fulfilled).toBe(15);
    expect(took).toBeLessThanOrEqual(1500);
  });
});
