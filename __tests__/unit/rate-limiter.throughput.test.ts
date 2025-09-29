import { describe, it, expect, beforeEach } from 'vitest';
import { RateLimiter, setRateLimiter } from '../../ztb/application/rate-limiter';

const LONG = String(process.env.LONG_TESTS ?? '0') === '1';

function pAll<T>(arr: Promise<T>[]) { return Promise.allSettled(arr); }

if (!LONG) {
  describe.skip('RateLimiter throughput (long) [disabled]', () => {
    it('skipped (set LONG_TESTS=1 to enable)', () => {});
  });
}

LONG && describe('RateLimiter throughput (long)', () => {
  beforeEach(() => {
    setRateLimiter(new RateLimiter({ capacity: 10, refillPerSec: 10, reserveRatio: 0.1 }));
  });

  it('handles ~100 req around 10/sec', async () => {
    const { withRetry } = await import('../../ztb/adapters/base-service');
    const start = Date.now();
    const calls: Promise<any>[] = [];
    for (let i=0;i<100;i++) calls.push(withRetry(async () => 1, 't', 1, 1, { category: 'API-PUBLIC', priority: 'normal' }));
    const res = await pAll(calls);
    const took = (Date.now() - start) / 1000; // seconds
    const ok = res.filter(r => r.status === 'fulfilled').length;
    expect(ok).toBe(100);
    // 10/sec なので理想は ~10s。多少の誤差は許容。
    expect(took).toBeGreaterThanOrEqual(8);
    expect(took).toBeLessThanOrEqual(14);
  }, 20000);
});
