import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CircuitBreaker } from '../../ztb/application/circuit-breaker';

describe('CircuitBreaker', () => {
  let cb: CircuitBreaker;
  beforeEach(() => { cb = new CircuitBreaker({ windowSize: 20, cooldownMs: 100, halfOpenTrial: 3, maxConsecutiveFailures: 5, failureThreshold: 0.5, latencyThreshold: 30 }); });

  it('stays CLOSED on successes', () => {
    for (let i=0;i<10;i++) cb.recordSuccess(10);
    expect(cb.getState()).toBe('CLOSED');
  });

  it('opens when failure rate > 50%', () => {
    for (let i=0;i<6;i++) cb.recordFailure();
    expect(cb.getState()).toBe('OPEN');
  });

  it('opens on 6 consecutive failures', () => {
    cb = new CircuitBreaker({ windowSize: 20, cooldownMs: 100, halfOpenTrial: 3, maxConsecutiveFailures: 5 });
    for (let i=0;i<6;i++) cb.recordFailure();
    expect(cb.getState()).toBe('OPEN');
  });

  it('OPEN -> HALF_OPEN after cooldown, then CLOSED on successes', async () => {
    for (let i=0;i<6;i++) cb.recordFailure();
    expect(cb.getState()).toBe('OPEN');
    // cooldown
    await new Promise(r => setTimeout(r, 120));
    expect(cb.allowRequest()).toBe(true); // moves to HALF_OPEN
    expect(cb.getState()).toBe('HALF_OPEN');
    cb.recordSuccess(5);
    cb.recordSuccess(5);
    cb.recordSuccess(5);
    expect(cb.getState()).toBe('CLOSED');
  });

  it('HALF_OPEN failure re-opens', async () => {
    for (let i=0;i<6;i++) cb.recordFailure();
    await new Promise(r => setTimeout(r, 120));
    expect(cb.allowRequest()).toBe(true);
    expect(cb.getState()).toBe('HALF_OPEN');
    cb.recordFailure();
    expect(cb.getState()).toBe('OPEN');
  });
});
