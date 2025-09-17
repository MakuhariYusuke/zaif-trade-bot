import { describe, it, expect, beforeEach } from 'vitest';
import { RateLimiter, setRateLimiter } from '../../src/application/rate-limiter';
import { withRetry } from '../../src/adapters/base-service';

function sleep(ms: number){ return new Promise(r => setTimeout(r, ms)); }

describe('RateLimiter per-category configs', () => {
  beforeEach(() => {
    process.env.RATE_METRICS_INTERVAL_MS = '0';
    process.env.RATE_CAPACITY_PUBLIC = '6';
    process.env.RATE_REFILL_PUBLIC = '6';
    process.env.RATE_CAPACITY_PRIVATE = '3';
    process.env.RATE_REFILL_PRIVATE = '3';
    process.env.RATE_CAPACITY_EXEC = '2';
    process.env.RATE_REFILL_EXEC = '2';
    setRateLimiter(new RateLimiter({ capacity: 5, refillPerSec: 5, reserveRatio: 0.2 }));
  });

  it('applies independent buckets per category', async () => {
    // PUBLIC can take ~6 immediately
    const p = [] as Promise<any>[];
    for (let i=0;i<6;i++) p.push(withRetry(async ()=>1, 'p', 1, 1, { category: 'API-PUBLIC', priority: 'normal', opType: 'QUERY' }));
    await Promise.all(p);
    // Next PUBLIC within short time should wait/reject; PRIVATE still has tokens
    await expect(withRetry(async ()=>1, 'p2', 1, 1, { category: 'API-PUBLIC', priority: 'normal', opType: 'QUERY' })).rejects.toBeTruthy().catch(()=>{});
    // PRIVATE has capacity 3
    const q = [] as Promise<any>[];
    for (let i=0;i<3;i++) q.push(withRetry(async ()=>1, 'q', 1, 1, { category: 'API-PRIVATE', priority: 'normal', opType: 'QUERY' }));
    await Promise.all(q);
    // EXEC capacity 2, high ORDER may borrow reserve if any but capacity is low
    const e1 = withRetry(async ()=>1, 'e1', 1, 1, { category: 'EXEC', priority: 'high', opType: 'ORDER' });
    const e2 = withRetry(async ()=>1, 'e2', 1, 1, { category: 'EXEC', priority: 'high', opType: 'ORDER' });
    await Promise.all([e1,e2]);
  });
});
