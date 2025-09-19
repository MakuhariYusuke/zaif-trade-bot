import { describe, it, expect } from 'vitest';
import { normalizeTradeConfig, getOrdersPerDay } from '../../../src/config/trade-config';

describe('trade-config mapping', ()=>{
  it('normalizes and maps orders per phase', ()=>{
    const cfg = normalizeTradeConfig({ pair: 'ETH_JPY', phase: 2, phaseSteps: [
      { phase: 1, ordersPerDay: 2 }, { phase: 3, ordersPerDay: 7 }
    ]});
    expect(cfg.pair).toBe('eth_jpy');
    expect(getOrdersPerDay(cfg, 1)).toBe(2);
    // phase=2 not defined -> nearest lower (1)
    expect(getOrdersPerDay(cfg, 2)).toBe(2);
    expect(getOrdersPerDay(cfg, 3)).toBe(7);
  });
});
