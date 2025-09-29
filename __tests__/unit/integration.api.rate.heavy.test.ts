import { describe, it, expect, beforeEach } from 'vitest';
import { RateLimiter, setRateLimiter } from '../../ztb/application/rate-limiter';

const LONG = String(process.env.LONG_TESTS ?? '0') === '1';

function pAll<T>(arr: Promise<T>[]) { return Promise.allSettled(arr); }

if (!LONG) {
  describe.skip('integration: heavy API rate limiting (long) [disabled]', () => {
    it('skipped (set LONG_TESTS=1 to enable)', () => {});
  });
}

LONG && describe('integration: heavy API rate limiting (long)', () => {
  beforeEach(() => {
    setRateLimiter(new RateLimiter({ capacity: 10, refillPerSec: 10, reserveRatio: 0.1 }));
  });

  it('1000 requests stay within ~10/sec window', async () => {
    const { withRetry } = await import('../../ztb/adapters/base-service');
    const start = Date.now();
    const calls: Promise<any>[] = [];
    for (let i=0;i<1000;i++) calls.push(withRetry(async () => 1, 'n', 1, 1, { category: 'API-PUBLIC', priority: 'normal' }));
    const res = await pAll(calls);
    const took = (Date.now() - start) / 1000; // seconds
    const ok = res.filter(r => r.status === 'fulfilled').length;
    expect(ok).toBe(1000);
    // おおよそ 100 秒 ± 20% 以内
    expect(took).toBeGreaterThanOrEqual(80);
    expect(took).toBeLessThanOrEqual(120);
  }, 130000);
});
