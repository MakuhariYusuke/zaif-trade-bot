import { describe, it, expect } from 'vitest';
import { macd, dmiAdx, bollinger, ichimoku, atr, supertrend } from '../../../ztb/utils/indicators';

describe('trend indicators', () => {
  const close = Array.from({ length: 120 }, (_, i) => 100 + Math.sin(i / 8) * 3 + i * 0.05);
  const high = close.map((c, i) => c + (1 + Math.sin(i / 5) * 0.2));
  const low = close.map((c, i) => c - (1 + Math.cos(i / 7) * 0.2));

  it('computes MACD(12,26,9)', () => {
    const res = macd(close, 12, 26, 9);
    expect(res.macd).not.toBeNull();
    expect(res.signal).not.toBeNull();
    expect(res.hist).not.toBeNull();
  });

  it('computes ADX(14)', () => {
    const res = dmiAdx(high, low, close, 14);
    expect(res.adx).not.toBeNull();
    expect(res.plusDi).not.toBeNull();
    expect(res.minusDi).not.toBeNull();
  });

  it('computes Bollinger(20,2)', () => {
    const bb = bollinger(close, 20, 2);
    expect(bb.basis).not.toBeNull();
    expect(bb.upper).not.toBeNull();
    expect(bb.lower).not.toBeNull();
    expect((bb.upper as number) > (bb.basis as number)).toBe(true);
    expect((bb.lower as number) < (bb.basis as number)).toBe(true);
  });

  it('computes Ichimoku(9,26,52)', () => {
    const ich = ichimoku(high, low, close, 9, 26, 52);
    expect(ich.tenkan).not.toBeNull();
    expect(ich.kijun).not.toBeNull();
    expect(ich.spanA).not.toBeNull();
    expect(ich.spanB).not.toBeNull();
    expect(ich.chikou).not.toBeNull();
  });

  it('computes ATR(10) and SuperTrend(10,3)', () => {
    const a = atr(high, low, close, 10);
    expect(a).not.toBeNull();
    const st = supertrend(high, low, close, 10, 3);
    expect(st.value).not.toBeNull();
    expect(st.dir === 1 || st.dir === -1).toBe(true);
  });
});
