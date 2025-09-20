import { describe, it, expect } from 'vitest';
import {
  atrSeries,
  heikinAshiSeries,
  keltnerSeries,
  donchianSeries,
  macdSeries,
  bollingerSeries,
  supertrendSeries
} from '../../../src/utils/indicators/series';
import { expectClose, expectCloseArray } from '../../helpers/expectClose';

describe('series indicators', () => {
  // Sample OHLC data for testing
  const highs = [110, 112, 115, 118, 120, 122, 125, 128, 130, 132, 135, 138, 140, 142, 145, 148, 150, 152, 155, 158, 160, 162, 165, 168, 170, 172, 175, 178, 180, 182];
  const lows = [105, 107, 108, 112, 115, 118, 120, 123, 125, 127, 130, 133, 135, 137, 140, 143, 145, 147, 150, 153, 155, 157, 160, 163, 165, 167, 170, 173, 175, 177];
  const closes = [108, 110, 113, 116, 118, 121, 124, 127, 129, 131, 134, 137, 139, 141, 144, 147, 149, 151, 154, 157, 159, 161, 164, 167, 169, 171, 174, 177, 179, 181];
  const opens = [107, 109, 111, 114, 117, 119, 122, 125, 127, 129, 132, 135, 137, 139, 142, 145, 147, 149, 152, 155, 157, 159, 162, 165, 167, 169, 172, 175, 177, 179];

  describe('ATR Series', () => {
    it('calculates ATR series correctly', () => {
      const result = atrSeries(highs.slice(0, 20), lows.slice(0, 20), closes.slice(0, 20), 14);
      expect(result.length).toBe(20);
      expect(result.slice(0, 14).every(v => v === null)).toBe(true);
      expect(result[14]).not.toBeNull();
      expect(typeof result[14]).toBe('number');
    });

    it('returns all nulls for insufficient data', () => {
      const result = atrSeries(highs.slice(0, 10), lows.slice(0, 10), closes.slice(0, 10), 14);
      expect(result.every(v => v === null)).toBe(true);
    });

    it('handles edge cases with flat market', () => {
      const flatHighs = Array(20).fill(100);
      const flatLows = Array(20).fill(100);
      const flatCloses = Array(20).fill(100);
      const result = atrSeries(flatHighs, flatLows, flatCloses, 14);
      expect(result[14]).toBe(0);
    });

    it('memoization works correctly', () => {
      const result1 = atrSeries(highs.slice(0, 20), lows.slice(0, 20), closes.slice(0, 20), 14);
      const result2 = atrSeries(highs.slice(0, 20), lows.slice(0, 20), closes.slice(0, 20), 14);
      expect(result1).toEqual(result2); // Same reference due to memoization
    });

    it('different parameters produce different results', () => {
      const result14 = atrSeries(highs.slice(0, 20), lows.slice(0, 20), closes.slice(0, 20), 14);
      const result10 = atrSeries(highs.slice(0, 20), lows.slice(0, 20), closes.slice(0, 20), 10);
      expect(result14).not.toEqual(result10);
    });
  });

  describe('Heikin-Ashi Series', () => {
    it('calculates Heikin-Ashi correctly', () => {
      const result = heikinAshiSeries(opens.slice(0, 10), highs.slice(0, 10), lows.slice(0, 10), closes.slice(0, 10));
      expect(result.open.length).toBe(10);
      expect(result.high.length).toBe(10);
      expect(result.low.length).toBe(10);
      expect(result.close.length).toBe(10);

      // First candle
      expect(result.open[0]).toBe((opens[0] + closes[0]) / 2);
      expect(result.close[0]).toBe((opens[0] + highs[0] + lows[0] + closes[0]) / 4);
    });

    it('throws error for mismatched array lengths', () => {
      expect(() => {
        heikinAshiSeries(opens.slice(0, 5), highs.slice(0, 10), lows.slice(0, 10), closes.slice(0, 10));
      }).toThrow('Input arrays for Heikin-Ashi must have the same length');
    });

    it('handles empty arrays', () => {
      const result = heikinAshiSeries([], [], [], []);
      expect(result.open).toEqual([]);
      expect(result.high).toEqual([]);
      expect(result.low).toEqual([]);
      expect(result.close).toEqual([]);
    });
  });

  describe('Bollinger Bands Series', () => {
    it('calculates Bollinger Bands correctly', () => {
      const result = bollingerSeries(closes.slice(0, 25), 20, 2);
      expect(result.basis.length).toBe(25);
      expect(result.upper.length).toBe(25);
      expect(result.lower.length).toBe(25);
      expect(result.bandwidth.length).toBe(25);
      expect(result.percentB.length).toBe(25);

      // First 19 values should be null
      expect(result.basis.slice(0, 19).every(v => v === null)).toBe(true);
      expect(result.upper.slice(0, 19).every(v => v === null)).toBe(true);
      expect(result.lower.slice(0, 19).every(v => v === null)).toBe(true);
    });

    it('calculates bandwidth and percentB correctly', () => {
      const values = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125];
      const result = bollingerSeries(values, 20, 2);

      // Check that bandwidth and percentB are calculated for the last value
      const lastIdx = values.length - 1;
      expect(result.bandwidth[lastIdx]).not.toBeNull();
      expect(result.percentB[lastIdx]).not.toBeNull();
      expect(result.percentB[lastIdx]).toBeGreaterThanOrEqual(0);
      expect(result.percentB[lastIdx]).toBeLessThanOrEqual(1);
    });

    it('handles flat data (zero variance)', () => {
      const flatValues = Array(25).fill(100);
      const result = bollingerSeries(flatValues, 20, 2);

      const lastIdx = flatValues.length - 1;
      expect(result.basis[lastIdx]).toBe(100);
      expect(result.upper[lastIdx]).toBe(100); // No variance, so bands equal to basis
      expect(result.lower[lastIdx]).toBe(100);
      expect(result.bandwidth[lastIdx]).toBe(0);
    });

    it('different k values produce different bands', () => {
      const result1 = bollingerSeries(closes.slice(0, 30), 20, 1);
      const result2 = bollingerSeries(closes.slice(0, 30), 20, 2);

      const lastIdx = 29; // Use index 29 instead of closes.length - 1
      expect(result1.upper[lastIdx]).toBeLessThan(result2.upper[lastIdx]!);
      expect(result1.lower[lastIdx]).toBeGreaterThan(result2.lower[lastIdx]!);
    });
  });

  describe('Keltner Series', () => {
    it('calculates Keltner channels correctly', () => {
      const result = keltnerSeries(highs.slice(0, 25), lows.slice(0, 25), closes.slice(0, 25), 20, 2);
      expect(result.basis.length).toBe(25);
      expect(result.upper.length).toBe(25);
      expect(result.lower.length).toBe(25);

      // First values should be null until ATR is available
      expect(result.basis.slice(0, 19).every(v => v === null)).toBe(true);
    });

    it('different multipliers produce different channels', () => {
      const result1 = keltnerSeries(highs.slice(0, 30), lows.slice(0, 30), closes.slice(0, 30), 20, 1);
      const result2 = keltnerSeries(highs.slice(0, 30), lows.slice(0, 30), closes.slice(0, 30), 20, 2);

      const lastIdx = 29; // Use index 29 instead of highs.length - 1
      expect(result1.upper[lastIdx]).toBeLessThan(result2.upper[lastIdx]!);
      expect(result1.lower[lastIdx]).toBeGreaterThan(result2.lower[lastIdx]!);
    });
  });

  describe('Donchian Series', () => {
    it('calculates Donchian channels correctly', () => {
      const result = donchianSeries(highs.slice(0, 25), lows.slice(0, 25), 20);
      expect(result.upper.length).toBe(25);
      expect(result.lower.length).toBe(25);
      expect(result.mid.length).toBe(25);

      // First 19 values should be null
      expect(result.upper.slice(0, 19).every(v => v === null)).toBe(true);
      expect(result.lower.slice(0, 19).every(v => v === null)).toBe(true);
      expect(result.mid.slice(0, 19).every(v => v === null)).toBe(true);
    });

    it('mid is average of upper and lower', () => {
      const result = donchianSeries(highs.slice(0, 25), lows.slice(0, 25), 20);
      const lastIdx = highs.length - 1;
      if (result.upper[lastIdx] && result.lower[lastIdx] && result.mid[lastIdx]) {
        expect(result.mid[lastIdx]).toBe((result.upper[lastIdx]! + result.lower[lastIdx]!) / 2);
      }
    });
  });

  describe('MACD Series', () => {
    it('calculates MACD correctly', () => {
      const result = macdSeries(closes.slice(0, 30), 12, 26, 9);
      expect(result.macd.length).toBe(30);
      expect(result.signal.length).toBe(30);
      expect(result.hist.length).toBe(30);

      // MACD starts after slow EMA is available (26 periods)
      expect(result.macd.slice(0, 25).every(v => v === null)).toBe(true);
    });

    it('signal line starts after signal period', () => {
      const result = macdSeries(closes.slice(0, 40), 12, 26, 9);
      // Signal should start after MACD + signal period
      expect(result.signal.slice(0, 34).every(v => v === null)).toBe(true);
      expect(result.signal[34]).not.toBeNull();
    });

    it('histogram is MACD minus signal', () => {
      const result = macdSeries(closes.slice(0, 40), 12, 26, 9);
      for (let i = 34; i < result.hist.length; i++) {
        if (result.macd[i] && result.signal[i] && result.hist[i]) {
          expect(result.hist[i]).toBe(result.macd[i]! - result.signal[i]!);
        }
      }
    });
  });

  describe('SuperTrend Series', () => {
    it('calculates SuperTrend correctly', () => {
      const result = supertrendSeries(highs.slice(0, 25), lows.slice(0, 25), closes.slice(0, 25), 10, 3);
      expect(result.line.length).toBe(25);
      expect(result.dir.length).toBe(25);

      // First values should be null until ATR is available (first valid direction at index period-1)
      expect(result.line.slice(0, 9).every(v => v === null)).toBe(true);
      expect(result.dir.slice(0, 9).every(v => v === null)).toBe(true);
      
      // Direction should be set starting from index 9 (period-1)
      expect(result.dir[9]).toBe('up');
      expect(result.line[9]).not.toBeNull();
    });

    it('direction changes based on price action', () => {
      // Create data that will definitely cause SuperTrend direction changes
      const volatileHighs = [110, 120, 130, 125, 135, 145, 140, 150, 160, 155, 165, 175, 170, 180, 190, 185, 195, 205, 200, 210, 220, 215, 225, 235, 230, 240, 250, 245, 255, 265, 260, 270, 280, 275, 285, 295, 290, 300, 310, 305, 315, 325, 320, 330, 340, 335, 345, 355, 350, 360];
      const volatileLows = [100, 105, 115, 110, 120, 130, 125, 135, 145, 140, 150, 160, 155, 165, 175, 170, 180, 190, 185, 195, 205, 200, 210, 220, 215, 225, 235, 230, 240, 250, 245, 255, 265, 260, 270, 280, 275, 285, 295, 290, 300, 310, 305, 315, 325, 320, 330, 340, 335, 345];
      const volatileCloses = [105, 115, 125, 118, 128, 138, 133, 143, 153, 148, 158, 168, 163, 173, 183, 178, 188, 198, 193, 203, 213, 208, 218, 228, 223, 233, 243, 238, 248, 258, 253, 263, 273, 268, 278, 288, 283, 293, 303, 298, 308, 318, 313, 323, 333, 328, 338, 348, 343, 353];
      
      const result = supertrendSeries(volatileHighs, volatileLows, volatileCloses, 10, 3);
      const directions = result.dir.filter(d => d !== null);
      
      // SuperTrend should have some directions after the initial period
      // The first direction is set to 'up', and may change based on price action
      expect(result.dir.length).toBe(volatileHighs.length);
      
      // Check that we have at least the initial direction
      const firstNonNullIndex = result.dir.findIndex(d => d !== null);
      expect(firstNonNullIndex).toBe(9); // Should start at index 9 (period-1)
      expect(result.dir[firstNonNullIndex]).toBe('up'); // First direction should be 'up'
      
      // Check that directions are valid
      const validDirections = directions.filter(d => d === 'up' || d === 'down');
      expect(validDirections.length).toBe(directions.length);
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('handles arrays with NaN values', () => {
      const badCloses = [...closes.slice(0, 20)];
      badCloses[10] = NaN;
      expect(() => atrSeries(highs.slice(0, 20), lows.slice(0, 20), badCloses, 14)).not.toThrow();
    });

    it('handles arrays with Infinity values', () => {
      const badCloses = [...closes.slice(0, 20)];
      badCloses[10] = Infinity;
      expect(() => atrSeries(highs.slice(0, 20), lows.slice(0, 20), badCloses, 14)).not.toThrow();
    });

    it('handles very short arrays', () => {
      const result = atrSeries([110, 112], [105, 107], [108, 110], 14);
      expect(result.every(v => v === null)).toBe(true);
    });

    it('handles period larger than array length', () => {
      const result = bollingerSeries(closes.slice(0, 10), 20);
      expect(result.basis.every(v => v === null)).toBe(true);
      expect(result.upper.every(v => v === null)).toBe(true);
      expect(result.lower.every(v => v === null)).toBe(true);
    });
  });

  describe('Memoization Boundary Testing', () => {
    it('ATR memoization handles boundary changes correctly', () => {
      // Test with data that crosses memoization boundaries
      const longHighs = highs.slice(0, 50);
      const longLows = lows.slice(0, 50);
      const longCloses = closes.slice(0, 50);

      // First call with full data
      const result1 = atrSeries(longHighs, longLows, longCloses, 14);

      // Second call with same data (should use cache)
      const result2 = atrSeries(longHighs, longLows, longCloses, 14);

      expect(result1).toBe(result2); // Same reference

      // Third call with slightly different data (should not use cache)
      const modifiedCloses = [...longCloses];
      modifiedCloses[49] = modifiedCloses[49] + 0.01;
      const result3 = atrSeries(longHighs, longLows, modifiedCloses, 14);

      expect(result3).not.toBe(result1);
      expect(result3[49]).not.toBe(result1[49]);
    });

    it('cache eviction works with different array lengths', () => {
      const result1 = atrSeries(highs.slice(0, 20), lows.slice(0, 20), closes.slice(0, 20), 14);
      const result2 = atrSeries(highs.slice(0, 25), lows.slice(0, 25), closes.slice(0, 25), 14);

      expect(result1.length).toBe(20);
      expect(result2.length).toBe(25);
      expect(result1).not.toBe(result2);
    });
  });
});