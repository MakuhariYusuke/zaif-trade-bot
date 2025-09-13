import { describe, it, expect, beforeEach } from 'vitest';
import { validateOrderPlacement, guardOk } from '../../../src/core/risk/guards';

describe('core/risk/guards.validateOrderPlacement', () => {
  beforeEach(() => {
    process.env.RISK_MIN_TRADE_SIZE = '0.01';
    process.env.MAX_POSITION_SIZE = '0.05';
    process.env.MAX_ORDER_NOTIONAL_JPY = '1000';
    process.env.COOLDOWN_SEC = '10';
    process.env.MAX_OPEN_ORDERS = '2';
  });

  it('blocks when qty below min trade size', () => {
    const r = validateOrderPlacement({
      now: Date.now(),
      side: 'BUY',
      price: 100,
      qty: 0.005,
      baseSymbol: 'btc',
      openOrdersCount: 0,
      positionQty: 0,
    });
    expect(r.isAllowed).toBe(false);
    expect(r.reason).toContain('min trade size');
  });

  it('blocks when position + buy qty exceeds max position size', () => {
    const r = validateOrderPlacement({
      now: Date.now(),
      side: 'BUY',
      price: 100,
      qty: 0.02,
      baseSymbol: 'btc',
      openOrdersCount: 0,
      positionQty: 0.04,
    });
    expect(r.isAllowed).toBe(false);
    expect(r.reason).toContain('max position');
  });

  it('blocks SELL when current position already exceeds max position size', () => {
    const r = validateOrderPlacement({
      now: Date.now(),
      side: 'SELL',
      price: 100,
      qty: 0.02,
      baseSymbol: 'btc',
      openOrdersCount: 0,
      positionQty: 0.2,
    });
    expect(r.isAllowed).toBe(false);
  });

  it('blocks when notional exceeds max', () => {
    const r = validateOrderPlacement({
      now: Date.now(),
      side: 'BUY',
      price: 100000,
      qty: 0.02,
      baseSymbol: 'btc',
      openOrdersCount: 0,
      positionQty: 0,
    });
    expect(r.isAllowed).toBe(false);
    expect(r.reason).toContain('max notional');
  });

  it('blocks when open orders exceed limit', () => {
    const r = validateOrderPlacement({
      now: Date.now(),
      side: 'BUY',
      price: 100,
      qty: 0.02,
      baseSymbol: 'btc',
      openOrdersCount: 3,
      positionQty: 0,
    });
    expect(r.isAllowed).toBe(false);
    expect(r.reason).toContain('max open orders');
  });

  it('blocks within cooldown after last exit', () => {
    const now = Date.now();
    const r = validateOrderPlacement({
      now,
      side: 'BUY',
      price: 100,
      qty: 0.02,
      baseSymbol: 'btc',
      openOrdersCount: 0,
      positionQty: 0,
      lastExitAt: now - 5000, // 5s < 10s cooldown
    });
    expect(r.isAllowed).toBe(false);
    expect(r.reason).toContain('cooldown');
  });

  it('allows when all conditions pass', () => {
    const r = validateOrderPlacement({
      now: Date.now(),
      side: 'BUY',
      price: 100,
      qty: 0.02,
      baseSymbol: 'btc',
      openOrdersCount: 1,
  positionQty: 0.01,
    });
    expect(guardOk(r)).toBe(true);
  });
});
