import { describe, it, expect, beforeEach, vi } from 'vitest';
import { validateConfig } from '../../../ztb/config/validate-config';
import { TradeConfig } from '../../../ztb/config/trade-config';

function baseCfg(): TradeConfig {
  return {
    pair: 'btc_jpy',
    phase: 1,
    phaseSteps: [ { phase:1, ordersPerDay:1 }, { phase:2, ordersPerDay:3 } ],
    maxOrdersPerDay: 10,
    maxLossPerDay: 1000,
    slippageGuardPct: 0.01,
  };
}

describe('validate-config', () => {
  beforeEach(()=>{
    vi.restoreAllMocks();
  });

  it('valid config passes', () => {
    const r = validateConfig(baseCfg(), {} as any);
    expect(r.valid).toBe(true);
    expect(r.errors).toHaveLength(0);
  });

  it('detects non-ascending phases & zero orders', () => {
    const bad: TradeConfig = { ...baseCfg(), phaseSteps: [ { phase:2, ordersPerDay:2 }, { phase:2, ordersPerDay:0 } ] };
    const r = validateConfig(bad, {} as any);
    expect(r.valid).toBe(false);
    expect(r.errors.some(e=>/ascending/.test(e))).toBe(true);
    expect(r.errors.some(e=>/ordersPerDay/.test(e))).toBe(true);
  });

  it('warns if maxOrdersPerDay < largest step', () => {
    const cfg: TradeConfig = { ...baseCfg(), maxOrdersPerDay: 2 };
    const r = validateConfig(cfg, {} as any);
    expect(r.valid).toBe(true);
    expect(r.warnings.some(w=>/maxOrdersPerDay/.test(w))).toBe(true);
  });

  it('errors on invalid pair format', () => {
    const cfg: TradeConfig = { ...baseCfg(), pair: 'BTC-JPY' } as any;
    const r = validateConfig(cfg, {} as any);
    expect(r.valid).toBe(false);
    expect(r.errors.some(e=>/pair format/.test(e))).toBe(true);
  });

  it('env variable warnings captured', () => {
    const env = { REPRICE_MAX_ATTEMPTS: '-1', RISK_MAX_SLIPPAGE_PCT: '0.5' } as any;
    const r = validateConfig(baseCfg(), env);
    expect(r.valid).toBe(true); // config itself fine
    expect(r.warnings.some(w=>/REPRICE_MAX_ATTEMPTS/.test(w))).toBe(true);
    expect(r.warnings.some(w=>/RISK_MAX_SLIPPAGE_PCT/.test(w))).toBe(true);
  });
});
