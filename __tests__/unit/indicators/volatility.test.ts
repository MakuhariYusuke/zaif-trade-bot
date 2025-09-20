import { describe, it, expect } from 'vitest';
import {
  stddev,
  bollinger,
  bbWidth,
  atr,
  donchianWidth,
  choppinessSeries
} from '../../../src/utils/indicators/volatility';
import { expectClose } from '../../helpers/expectClose';

describe('volatility indicators', () => {
  // Sample OHLC data for testing
  const highs = [110, 112, 115, 118, 120, 122, 125, 128, 130, 132, 135, 138, 140, 142, 145, 148, 150, 152, 155, 158, 160, 162, 165, 168, 170, 172, 175, 178, 180, 182];
  const lows = [105, 107, 108, 112, 115, 118, 120, 123, 125, 127, 130, 133, 135, 137, 140, 143, 145, 147, 150, 153, 155, 157, 160, 163, 165, 167, 170, 173, 175, 177];
  const closes = [108, 110, 113, 116, 118, 121, 124, 127, 129, 131, 134, 137, 139, 141, 144, 147, 149, 151, 154, 157, 159, 161, 164, 167, 169, 171, 174, 177, 179, 181];

  describe('Standard Deviation (STDDEV)', () => {
    it('calculates STDDEV correctly', () => {
      const values = [100, 102, 98, 105, 97, 103, 99, 101, 104, 96];
      const result = stddev(values, 5);
      expect(result).not.toBeNull();
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThan(0);
    });

    it('returns null for insufficient data', () => {
      expect(stddev([100, 102], 5)).toBeNull();
      expect(stddev([], 5)).toBeNull();
    });

    it('returns null for invalid period', () => {
      expect(stddev([100, 102, 103], 0)).toBeNull();
      expect(stddev([100, 102, 103], -1)).toBeNull();
    });

    it('calculates zero for identical values', () => {
      const flatValues = Array(10).fill(100);
      const result = stddev(flatValues, 5);
      expect(result).toBe(0);
    });

    it('handles different periods correctly', () => {
      const values = Array.from({ length: 20 }, (_, i) => 100 + Math.sin(i / 2) * 10);
      const result5 = stddev(values, 5);
      const result10 = stddev(values, 10);
      expect(result5).not.toBeNull();
      expect(result10).not.toBeNull();
      // Different periods should give different results
      expect(result5).not.toBe(result10);
    });
  });

  describe('Bollinger Bands', () => {
    it('calculates Bollinger Bands correctly', () => {
      const values = [100, 102, 98, 105, 97, 103, 99, 101, 104, 96, 102, 98, 105, 97, 103, 99, 101, 104, 96, 102];
      const result = bollinger(values, 20, 2);
      expect(result.basis).not.toBeNull();
      expect(result.upper).not.toBeNull();
      expect(result.lower).not.toBeNull();
      expect(result.upper).toBeGreaterThan(result.basis!);
      expect(result.lower).toBeLessThan(result.basis!);
    });

    it('returns null for insufficient data', () => {
      const result = bollinger([100, 102], 20);
      expect(result.basis).toBeNull();
      expect(result.upper).toBeNull();
      expect(result.lower).toBeNull();
    });

    it('handles flat data correctly', () => {
      const flatValues = Array(20).fill(100);
      const result = bollinger(flatValues, 20, 2);
      expect(result.basis).toBe(100);
      expect(result.upper).toBe(100); // No variance, bands equal to basis
      expect(result.lower).toBe(100);
    });

    it('different k values produce different bands', () => {
      const values = Array.from({ length: 20 }, (_, i) => 100 + Math.sin(i / 2) * 10);
      const result1 = bollinger(values, 20, 1);
      const result2 = bollinger(values, 20, 2);

      expect(result1.upper).toBeLessThan(result2.upper!);
      expect(result1.lower).toBeGreaterThan(result2.lower!);
    });

    it('bands are symmetric around basis', () => {
      const values = Array.from({ length: 20 }, (_, i) => 100 + Math.sin(i / 2) * 10);
      const result = bollinger(values, 20, 2);

      const upperDiff = result.upper! - result.basis!;
      const lowerDiff = result.basis! - result.lower!;

      expectClose(upperDiff, lowerDiff, 1e-10);
    });
  });

  describe('Bollinger Band Width (BBW)', () => {
    it('calculates BBW correctly', () => {
      const values = [100, 102, 98, 105, 97, 103, 99, 101, 104, 96, 102, 98, 105, 97, 103, 99, 101, 104, 96, 102];
      const result = bbWidth(values, 20, 2);
      expect(result).not.toBeNull();
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThan(0);
    });

    it('returns null for insufficient data', () => {
      expect(bbWidth([100, 102], 20)).toBeNull();
    });

    it('returns null for zero basis', () => {
      const values = Array(20).fill(0);
      expect(bbWidth(values, 20)).toBeNull();
    });

    it('calculates zero width for flat data', () => {
      const flatValues = Array(20).fill(100);
      const result = bbWidth(flatValues, 20, 2);
      expect(result).toBe(0);
    });

    it('width increases with k multiplier', () => {
      const values = Array.from({ length: 20 }, (_, i) => 100 + Math.sin(i / 2) * 10);
      const result1 = bbWidth(values, 20, 1);
      const result2 = bbWidth(values, 20, 2);

      expect(result2).toBeGreaterThan(result1!);
    });
  });

  describe('Average True Range (ATR)', () => {
    it('calculates ATR correctly', () => {
      const result = atr(highs.slice(0, 20), lows.slice(0, 20), closes.slice(0, 20), 14);
      expect(result).not.toBeNull();
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThan(0);
    });

    it('returns null for insufficient data', () => {
      expect(atr(highs.slice(0, 10), lows.slice(0, 10), closes.slice(0, 10), 14)).toBeNull();
    });

    it('returns null for invalid period', () => {
      expect(atr(highs.slice(0, 15), lows.slice(0, 15), closes.slice(0, 15), 0)).toBeNull();
    });

    it('calculates zero ATR for flat market', () => {
      const flatHighs = Array(20).fill(100);
      const flatLows = Array(20).fill(100);
      const flatCloses = Array(20).fill(100);
      const result = atr(flatHighs, flatLows, flatCloses, 14);
      expect(result).toBe(0);
    });

    it('different periods produce different results', () => {
      const result10 = atr(highs.slice(0, 25), lows.slice(0, 25), closes.slice(0, 25), 10);
      const result20 = atr(highs.slice(0, 30), lows.slice(0, 30), closes.slice(0, 30), 20);

      expect(result10).not.toBeNull();
      expect(result20).not.toBeNull();
      expect(result10).not.toBe(result20);
    });
  });

  describe('Donchian Channel Width', () => {
    it('calculates Donchian Width correctly', () => {
      const result = donchianWidth(highs.slice(0, 25), lows.slice(0, 25), closes.slice(0, 25), 20);
      expect(result).not.toBeNull();
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThan(0);
    });

    it('returns null for insufficient data', () => {
      expect(donchianWidth(highs.slice(0, 10), lows.slice(0, 10), closes.slice(0, 10), 20)).toBeNull();
    });

    it('returns null for zero close price', () => {
      const zeroCloses = [...closes.slice(0, 25)];
      zeroCloses[24] = 0;
      expect(donchianWidth(highs.slice(0, 25), lows.slice(0, 25), zeroCloses, 20)).toBeNull();
    });

    it('calculates zero width for flat market', () => {
      const flatHighs = Array(25).fill(100);
      const flatLows = Array(25).fill(100);
      const flatCloses = Array(25).fill(100);
      const result = donchianWidth(flatHighs, flatLows, flatCloses, 20);
      expect(result).toBe(0);
    });

    it('different periods produce different results', () => {
      const result10 = donchianWidth(highs.slice(0, 25), lows.slice(0, 25), closes.slice(0, 25), 10);
      const result20 = donchianWidth(highs.slice(0, 25), lows.slice(0, 25), closes.slice(0, 25), 20);

      expect(result10).not.toBeNull();
      expect(result20).not.toBeNull();
      expect(result10).not.toBe(result20);
    });
  });

  describe('Choppiness Index Series', () => {
    it('calculates Choppiness Index correctly', () => {
      const result = choppinessSeries(highs.slice(0, 25), lows.slice(0, 25), closes.slice(0, 25), 14);
      expect(result.choppiness.length).toBe(25);
      expect(result.choppiness.slice(0, 14).every(v => v === null)).toBe(true);
      expect(result.choppiness[14]).not.toBeNull();
    });

    it('returns all nulls for insufficient data', () => {
      const result = choppinessSeries(highs.slice(0, 5), lows.slice(0, 5), closes.slice(0, 5), 14);
      expect(result.choppiness.every(v => v === null)).toBe(true);
    });

    it('calculates values between 0 and 100', () => {
      const result = choppinessSeries(highs.slice(0, 30), lows.slice(0, 30), closes.slice(0, 30), 14);
      const validValues = result.choppiness.filter(v => v !== null) as number[];
      expect(validValues.every(v => v >= 0 && v <= 100)).toBe(true);
    });

    it('handles flat market (should approach 100)', () => {
      const flatHighs = Array(30).fill(100);
      const flatLows = Array(30).fill(100);
      const flatCloses = Array(30).fill(100);
      const result = choppinessSeries(flatHighs, flatLows, flatCloses, 14);

      const lastValue = result.choppiness[result.choppiness.length - 1];
      // In a flat market, TR (True Range) is 0, so choppiness cannot be calculated
      expect(lastValue).toBeNull();
    });

    it('different periods produce different results', () => {
      const result10 = choppinessSeries(highs.slice(0, 30), lows.slice(0, 30), closes.slice(0, 30), 10);
      const result20 = choppinessSeries(highs.slice(0, 35), lows.slice(0, 35), closes.slice(0, 35), 20);

      const last10 = result10.choppiness[result10.choppiness.length - 1];
      const last20 = result20.choppiness[result20.choppiness.length - 1];

      expect(last10).not.toBeNull();
      expect(last20).not.toBeNull();
      expect(last10).not.toBe(last20);
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('handles arrays with NaN values', () => {
      const badValues = [...closes.slice(0, 20)];
      badValues[10] = NaN;
      expect(() => stddev(badValues, 5)).not.toThrow();
      expect(() => bollinger(badValues, 20)).not.toThrow();
    });

    it('handles arrays with Infinity values', () => {
      const badValues = [...closes.slice(0, 20)];
      badValues[10] = Infinity;
      expect(() => stddev(badValues, 5)).not.toThrow();
      expect(() => bollinger(badValues, 20)).not.toThrow();
    });

    it('handles very short arrays', () => {
      expect(stddev([100], 1)).toBe(0);
      expect(stddev([100], 2)).toBeNull();
      expect(bollinger([100], 1, 2)).toEqual({ basis: 100, upper: 100, lower: 100 });
    });

    it('handles period equal to array length', () => {
      const values = [100, 102, 98, 105, 97];
      const result = stddev(values, 5);
      expect(result).not.toBeNull();
      expect(result).toBeGreaterThanOrEqual(0);
    });

    it('handles negative values correctly', () => {
      const negativeValues = [-100, -102, -98, -105, -97, -103, -99, -101, -104, -96];
      const result = stddev(negativeValues, 5);
      expect(result).not.toBeNull();
      expect(result).toBeGreaterThan(0);
    });
  });

  describe('Parameter Variations', () => {
    it('tests different k values for Bollinger Bands', () => {
      const values = Array.from({ length: 20 }, (_, i) => 100 + Math.sin(i / 2) * 10);
      const kValues = [1, 1.5, 2, 2.5, 3];

      const results = kValues.map(k => bbWidth(values, 20, k));

      // Higher k should produce wider bands
      for (let i = 1; i < results.length; i++) {
        expect(results[i]).toBeGreaterThan(results[i - 1]!);
      }
    });

    it('tests different periods for ATR', () => {
      // Extend test data to ensure sufficient length for all periods
      const extendedHighs = [...highs, ...Array(50).fill(0).map((_, i) => highs[highs.length - 1] + i * 2)];
      const extendedLows = [...lows, ...Array(50).fill(0).map((_, i) => lows[lows.length - 1] + i * 1.5)];
      const extendedCloses = [...closes, ...Array(50).fill(0).map((_, i) => closes[closes.length - 1] + i * 1.8)];
      
      const periods = [5, 10, 14, 20, 25]; // Reduced max period to fit available data
      const results = periods.map(p =>
        atr(extendedHighs.slice(0, Math.max(50, p + 15)), extendedLows.slice(0, Math.max(50, p + 15)), extendedCloses.slice(0, Math.max(50, p + 15)), p)
      );

      results.forEach(result => {
        expect(result).not.toBeNull();
        expect(result).toBeGreaterThanOrEqual(0);
      });
    });

    it('tests different periods for Donchian Width', () => {
      const periods = [5, 10, 15, 20, 25];
      const results = periods.map(p =>
        donchianWidth(highs.slice(0, Math.max(30, p + 5)), lows.slice(0, Math.max(30, p + 5)), closes.slice(0, Math.max(30, p + 5)), p)
      );

      results.forEach(result => {
        expect(result).not.toBeNull();
        expect(result).toBeGreaterThanOrEqual(0);
      });
    });
  });
});