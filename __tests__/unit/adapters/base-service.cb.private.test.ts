import { describe, it, expect, beforeEach } from 'vitest';
import { withRetry } from '../../../ztb/adapters/base-service';

function makeFail(times: number, err: any){
  let n = 0;
  return async () => {
    n++;
    throw err;
  };
}

function delay(ms: number){ return new Promise(r => setTimeout(r, ms)); }

describe('BaseService.withRetry CircuitBreaker (API-PRIVATE)', () => {
  beforeEach(() => {
    // reset category-specific CB singletons
    const g: any = global as any;
    g.__cb_private = undefined;
  });

  it('opens circuit after consecutive failures and blocks', async () => {
    const g: any = global as any;
    const { CircuitBreaker } = await import('../../../ztb/application/circuit-breaker');
    g.__cb_private = new CircuitBreaker({ windowSize: 5, maxConsecutiveFailures: 1, failureThreshold: 0.1, cooldownMs: 100 });
    const err = Object.assign(new Error('auth failed'), { code: 'AUTH' });
    const fn = makeFail(10, err);
    await expect(withRetry(fn, 'privateCall', 2, 1, { category: 'API-PRIVATE' })).rejects.toThrowError(/privateCall failed/);
    await expect(withRetry(async () => 1, 'privateCall2', 1, 1, { category: 'API-PRIVATE' })).rejects.toHaveProperty('cause.code', 'CIRCUIT_OPEN');
  });

  it('half-open allows trial after cooldown then closes on success', async () => {
    const g: any = global as any;
    // ensure cb exists and configure short cooldown
    const { CircuitBreaker } = await import('../../../ztb/application/circuit-breaker');
    g.__cb_private = new CircuitBreaker({ windowSize: 5, maxConsecutiveFailures: 1, cooldownMs: 50, halfOpenTrial: 1, latencyThreshold: 999999 });
    const cb = g.__cb_private as InstanceType<typeof CircuitBreaker>;
    cb.recordFailure(); // OPEN
    expect(cb.getState()).toBe('OPEN');

    await delay(60);
    // withRetry should pass allowRequest (HALF_OPEN) and record success, closing it
    const res = await withRetry(async () => 42, 'trial', 1, 1, { category: 'API-PRIVATE' });
    expect(res).toBe(42);
    expect(cb.getState()).toBe('CLOSED');
  });
});
