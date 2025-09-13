import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { clampAmountForSafety } from '../src/utils/toolkit';

describe('toolkit clampAmountForSafety', () => {
  const envBackup = { ...process.env };
  beforeEach(() => { process.env = { ...envBackup }; });
  afterEach(() => { process.env = { ...envBackup }; });

  it('returns original amount when SAFETY_MODE != 1', () => {
    delete process.env.SAFETY_MODE;
    const funds:any = { jpy: 100000, btc: 2 };
    const amt = clampAmountForSafety('bid', 5, 1000, funds, 'btc_jpy');
    expect(amt).toBe(5);
  });

  it('clamps bid side by SAFETY_CLAMP_PCT of JPY notional', () => {
    process.env.SAFETY_MODE = '1';
    process.env.SAFETY_CLAMP_PCT = '0.2'; // 20%
    const funds:any = { jpy: 100000 };
    // price=1 => maxQty = 100000 * 0.2 / 1 = 20000
    expect(clampAmountForSafety('bid', 50000, 1, funds, 'btc_jpy')).toBe(20000);
    // below cap remains unchanged
    expect(clampAmountForSafety('bid', 10000, 1, funds, 'btc_jpy')).toBe(10000);
  });

  it('clamps ask side by SAFETY_CLAMP_PCT of base balance', () => {
    process.env.SAFETY_MODE = '1';
    process.env.SAFETY_CLAMP_PCT = '0.1'; // 10%
    const funds:any = { jpy: 0, btc: 5 };
    // 10% of 5 BTC = 0.5
    expect(clampAmountForSafety('ask', 1.0, 1000000, funds, 'btc_jpy')).toBeCloseTo(0.5, 10);
    expect(clampAmountForSafety('ask', 0.2, 1000000, funds, 'btc_jpy')).toBeCloseTo(0.2, 10);
  });
});
