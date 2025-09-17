import { describe, it, expect, beforeEach, vi } from 'vitest';
import { getEventBus } from '../../../src/application/events/bus';
import { registerStatsSubscriber } from '../../../src/application/events/subscribers/stats-subscriber';
import { getCircuitBreaker, setCircuitBreaker, CircuitBreaker } from '../../../src/application/circuit-breaker';

describe('stats-subscriber -> CircuitBreaker', () => {
  beforeEach(() => {
    // fresh bus
    getEventBus().clear();
    // inject fresh CB
    setCircuitBreaker(new CircuitBreaker({ windowSize: 10, maxConsecutiveFailures: 5, cooldownMs: 50, halfOpenTrial: 2, latencyThreshold: 1000 }));
    registerStatsSubscriber();
  });

  it('ORDER_FILLED keeps CB CLOSED', async () => {
    const cb = getCircuitBreaker();
    for (let i=0;i<3;i++) getEventBus().publish({ type: 'ORDER_SUBMITTED', requestId: `r${i}`, orderId: String(i), pair: 'btc_jpy', side: 'buy', amount: 0.01, price: 100 });
    for (let i=0;i<3;i++) getEventBus().publish({ type: 'ORDER_FILLED', requestId: `r${i}`, orderId: String(i), pair: 'btc_jpy', side: 'buy', amount: 0.01, price: 100, filled: 0.01, avgPrice: 100 });
    await new Promise(r => setTimeout(r, 0));
    expect(cb.getState()).toBe('CLOSED');
  });

  it('6 failures -> OPEN', async () => {
    const cb = getCircuitBreaker();
    for (let i=0;i<6;i++) {
      const id = `x${i}`;
      getEventBus().publish({ type: 'ORDER_SUBMITTED', requestId: id, orderId: String(i), pair: 'btc_jpy', side: 'buy', amount: 0.01, price: 100 });
      getEventBus().publish({ type: 'ORDER_EXPIRED', requestId: id, orderId: String(i), pair: 'btc_jpy', side: 'buy', amount: 0.01, price: 100, cause: { code: 'TIMEOUT' } });
    }
    await new Promise(r => setTimeout(r, 0));
    expect(cb.getState()).toBe('OPEN');
  });
});
