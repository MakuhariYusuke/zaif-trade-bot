import { describe, it, expect, beforeEach } from 'vitest';
import { RateLimiter } from '../../ztb/application/rate-limiter';

function sleep(ms: number){ return new Promise(r => setTimeout(r, ms)); }

describe('RateLimiter', () => {
  let rl: RateLimiter;
  beforeEach(() => { rl = new RateLimiter({ capacity: 5, refillPerSec: 1, reserveRatio: 0.2 }); });

  it('rejects when bucket near reserve then allows after refill', async () => {
    // capacity=5, reserve=1 => normalは4まで即取得可
    for (let i=0;i<4;i++) {
      const w = await rl.acquire('normal', 50);
      expect(w).toBeLessThan(20);
    }
    // 5本目(normal)は予約分に抵触するため短時間では拒否
    await expect(rl.acquire('normal', 200)).rejects.toHaveProperty('code','RATE_LIMITED');
    await sleep(1000);
    const w2 = await rl.acquire('normal', 300);
    expect(w2).toBeLessThanOrEqual(300);
  });

  it('high priority borrows from reserved capacity', async () => {
    // Drain non-reserved space (capacity=5, reserve=1)
    for (let i=0;i<4;i++) await rl.acquire('normal', 50);
    // normal is now at reserved boundary, should reject quickly
    await expect(rl.acquire('normal', 100)).rejects.toHaveProperty('code','RATE_LIMITED');
    // high priority ORDER should still pass (reserved quota)
    const w = await rl.acquire('high', 100, 'EXEC', 'ORDER');
    expect(w).toBeLessThanOrEqual(100);
  });
});
