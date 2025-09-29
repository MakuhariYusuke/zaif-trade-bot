import { describe, it, expect } from 'vitest';
import { normalizeTradeConfig, getOrdersPerDay } from '../../../ztb/config/trade-config';

describe('trade-config', () => {
  it('normalizes and maps phase to ordersPerDay', () => {
    const cfg = normalizeTradeConfig({ pair: 'BTC_JPY', phase: 2, phaseSteps: [ { phase: 1, ordersPerDay: 1 }, { phase: 3, ordersPerDay: 5 } ] });
    expect(cfg.pair).toBe('btc_jpy');
    expect(getOrdersPerDay(cfg, 1)).toBe(1);
    // missing exact phase uses nearest lower or first
    expect(getOrdersPerDay(cfg, 2)).toBe(1);
    expect(getOrdersPerDay(cfg, 3)).toBe(5);
  });
});
