import { describe, it, expect, beforeEach } from 'vitest';
import { loadRiskConfig, getQtyDecimals, getQtyEpsilon } from '../../../ztb/adapters/risk-config';

describe('risk-config boundary', () => {
  const origEnv = { ...process.env };
  beforeEach(() => {
    process.env = { ...origEnv };
  });

  it('qtyDecimalsDefault falls back to 8 and clamps negative', () => {
    delete process.env.QTY_DECIMALS_DEFAULT;
    expect(getQtyDecimals('btc_jpy')).toBe(8);
    process.env.QTY_DECIMALS_DEFAULT = '-3';
    expect(getQtyDecimals('btc_jpy')).toBe(0);
  });

  it('pair map overrides default and floors fractional', () => {
    process.env.QTY_DECIMALS_DEFAULT = '6';
    process.env.QTY_DECIMALS_MAP = JSON.stringify({ btc_jpy: 5.9 });
    expect(getQtyDecimals('btc_jpy')).toBe(5);
    expect(getQtyDecimals('eth_jpy')).toBe(6);
  });

  it('epsilon matches decimals', () => {
    process.env.QTY_DECIMALS_DEFAULT = '4';
    expect(getQtyEpsilon('xrp_jpy')).toBeCloseTo(1e-4, 10);
  });

  it('loadRiskConfig returns numeric fields with defaults when env absent', () => {
    delete process.env.RISK_STOP_LOSS_PCT;
    const cfg = loadRiskConfig();
    expect(typeof cfg.stopLossPct).toBe('number');
    expect(cfg.stopLossPct).toBeGreaterThan(0);
  });
});
