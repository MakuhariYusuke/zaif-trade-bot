import { describe, it, expect, beforeEach } from 'vitest';
import { withRetry } from '../../../ztb/adapters/base-service';

function delay(ms: number){ return new Promise(r => setTimeout(r, ms)); }

describe('BaseService.withRetry CircuitBreaker (EXEC)', () => {
  beforeEach(() => {
    const g: any = global as any;
    g.__cb_exec = undefined;
  });

  it('blocks immediately when circuit is OPEN', async () => {
    const g: any = global as any;
    const { CircuitBreaker } = await import('../../../ztb/application/circuit-breaker');
    const cb = new CircuitBreaker({ windowSize: 5, maxConsecutiveFailures: 1, failureThreshold: 0.1, cooldownMs: 1000 });
    // force OPEN with a single failure (per config)
    cb.recordFailure();
    g.__cb_exec = cb;
    expect(cb.getState()).toBe('OPEN');

    await expect(withRetry(async () => 1, 'exec', 1, 1, { category: 'EXEC' })).rejects.toHaveProperty('cause.code', 'CIRCUIT_OPEN');
  });

  it('recovers after cooldown and success', async () => {
    const g: any = global as any;
    const { CircuitBreaker } = await import('../../../ztb/application/circuit-breaker');
    const cb = new CircuitBreaker({ windowSize: 5, maxConsecutiveFailures: 1, cooldownMs: 30, halfOpenTrial: 1, latencyThreshold: 999999 });
    cb.recordFailure(); // OPEN
    g.__cb_exec = cb;
    expect(cb.getState()).toBe('OPEN');

    await delay(40);
    // first attempt should be allowed (HALF_OPEN), success should close
    const out = await withRetry(async () => 'ok', 'exec2', 1, 1, { category: 'EXEC' });
    expect(out).toBe('ok');
    expect(cb.getState()).toBe('CLOSED');
  });
});
