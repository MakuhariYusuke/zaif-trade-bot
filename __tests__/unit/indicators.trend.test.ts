import { describe, it, expect } from 'vitest';
import { macd, dmiAdx, bollinger, ichimoku, atr, supertrend } from '../../src/utils/indicators';
import { dmiAdxSeries, ichimokuSeries, supertrendSeries } from '../../src/utils/indicators/trend';

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

  describe('DMI/ADX Series', () => {
    it('computes DMI/ADX series with sufficient data', () => {
      const result = dmiAdxSeries(high, low, close, 14);
      expect(result.plusDi.length).toBe(close.length);
      expect(result.minusDi.length).toBe(close.length);
      expect(result.adx.length).toBe(close.length);

      // Check that we have some non-null values
      const nonNullAdx = result.adx.filter(v => v !== null);
      expect(nonNullAdx.length).toBeGreaterThan(0);
    });

    it('returns null arrays for insufficient data', () => {
      const shortHigh = [100, 101, 102];
      const shortLow = [99, 98, 97];
      const shortClose = [100, 100, 100];
      const result = dmiAdxSeries(shortHigh, shortLow, shortClose, 14);

      expect(result.plusDi.every(v => v === null)).toBe(true);
      expect(result.minusDi.every(v => v === null)).toBe(true);
      expect(result.adx.every(v => v === null)).toBe(true);
    });

    it('handles empty arrays', () => {
      const result = dmiAdxSeries([], [], [], 14);
      expect(result.plusDi).toEqual([]);
      expect(result.minusDi).toEqual([]);
      expect(result.adx).toEqual([]);
    });
  });

  describe('Ichimoku Series', () => {
    it('computes Ichimoku series with sufficient data', () => {
      const result = ichimokuSeries(high, low, close, 9, 26, 52);
      expect(result.tenkan.length).toBe(close.length);
      expect(result.kijun.length).toBe(close.length);
      expect(result.senkouA.length).toBe(close.length);
      expect(result.senkouB.length).toBe(close.length);
      expect(result.chikou.length).toBe(close.length);
      expect(result.priceVsCloud.length).toBe(close.length);

      // Check that we have some non-null values
      const nonNullTenkan = result.tenkan.filter(v => v !== null);
      expect(nonNullTenkan.length).toBeGreaterThan(0);
    });

    it('handles price vs cloud classification', () => {
      const result = ichimokuSeries(high, low, close, 9, 26, 52);
      const classifications = result.priceVsCloud.filter(v => v !== null);
      expect(classifications.length).toBeGreaterThan(0);

      // All classifications should be valid
      classifications.forEach(cls => {
        expect(['above', 'in', 'below']).toContain(cls);
      });
    });

    it('handles short arrays gracefully', () => {
      const shortHigh = [100, 101, 102];
      const shortLow = [99, 98, 97];
      const shortClose = [100, 100, 100];
      const result = ichimokuSeries(shortHigh, shortLow, shortClose, 9, 26, 52);

      // Should still return arrays of correct length
      expect(result.tenkan.length).toBe(3);
      expect(result.kijun.length).toBe(3);
    });
  });

  describe('SuperTrend Series', () => {
    it('computes SuperTrend series with sufficient data', () => {
      const result = supertrendSeries(high, low, close, 10, 3);
      expect(result.line.length).toBe(close.length);
      expect(result.dir.length).toBe(close.length);

      // Check that we have some non-null values after the period
      const nonNullLines = result.line.filter(v => v !== null);
      expect(nonNullLines.length).toBeGreaterThan(0);

      const nonNullDirs = result.dir.filter(v => v !== null);
      expect(nonNullDirs.length).toBeGreaterThan(0);
    });

    it('handles direction changes correctly', () => {
      const result = supertrendSeries(high, low, close, 10, 3);
      const directions = result.dir.filter(v => v !== null);

      // All directions should be valid
      directions.forEach(dir => {
        expect(['up', 'down']).toContain(dir);
      });
    });

    it('handles empty arrays', () => {
      const result = supertrendSeries([], [], [], 10, 3);
      expect(result.line).toEqual([]);
      expect(result.dir).toEqual([]);
    });

    it('handles short arrays', () => {
      const shortHigh = [100, 101, 102];
      const shortLow = [99, 98, 97];
      const shortClose = [100, 100, 100];
      const result = supertrendSeries(shortHigh, shortLow, shortClose, 10, 3);

      // Should return arrays filled with null
      expect(result.line.every(v => v === null)).toBe(true);
      expect(result.dir.every(v => v === null)).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    it('handles arrays with identical values', () => {
      const flatHigh = Array(50).fill(100);
      const flatLow = Array(50).fill(100);
      const flatClose = Array(50).fill(100);

      const adxResult = dmiAdx(flatHigh, flatLow, flatClose, 14);
      expect(adxResult.adx).toBe(0);
      expect(adxResult.plusDi).toBe(0);
      expect(adxResult.minusDi).toBe(0);

      // Use shorter periods for Ichimoku to ensure we get values
      const ichResult = ichimoku(flatHigh, flatLow, flatClose, 9, 26, 30);
      expect(ichResult.tenkan).toBe(100);
      expect(ichResult.kijun).toBe(100);
      expect(ichResult.spanA).toBe(100);
      expect(ichResult.spanB).toBe(100);
    });

    it('handles arrays with extreme volatility', () => {
      const volatileHigh = close.map((c, i) => c + Math.sin(i) * 10);
      const volatileLow = close.map((c, i) => c - Math.cos(i) * 10);

      const adxResult = dmiAdx(volatileHigh, volatileLow, close, 14);
      expect(adxResult.adx).not.toBeNull();
      expect(adxResult.plusDi).not.toBeNull();
      expect(adxResult.minusDi).not.toBeNull();

      const stResult = supertrendSeries(volatileHigh, volatileLow, close, 10, 3);
      expect(stResult.line.length).toBe(close.length);
    });
  });
});
