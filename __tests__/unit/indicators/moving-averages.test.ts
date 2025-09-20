import { describe, it, expect } from 'vitest';
import { sma, ema, wma, hma, kama } from '../../../src/utils/indicators/moving-averages';

describe('moving averages', () => {
  const prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25];

  describe('Simple Moving Average (SMA)', () => {
    it('calculates SMA correctly', () => {
      expect(sma([10, 11, 12], 3)).toBe(11);
      expect(sma([10, 11, 12, 13, 14], 3)).toBe(13); // (12+13+14)/3
      expect(sma([10, 11, 12, 13, 14], 3)).toBeCloseTo(13, 2);
    });

    it('returns null for insufficient data', () => {
      expect(sma([10, 11], 3)).toBeNull();
      expect(sma([], 3)).toBeNull();
    });

    it('returns null for invalid period', () => {
      expect(sma([10, 11, 12], 0)).toBeNull();
      expect(sma([10, 11, 12], -1)).toBeNull();
    });

    it('handles period equal to array length', () => {
      expect(sma([10, 11, 12], 3)).toBe(11);
    });
  });

  describe('Exponential Moving Average (EMA)', () => {
    it('calculates EMA correctly', () => {
      const values = [10, 11, 12, 13, 14];
      const result = ema(values, 3);
      expect(result).not.toBeNull();
      expect(typeof result).toBe('number');
    });

    it('returns null for insufficient data', () => {
      expect(ema([10, 11], 3)).toBeNull();
      expect(ema([], 3)).toBeNull();
    });

    it('returns null for invalid period', () => {
      expect(ema([10, 11, 12], 0)).toBeNull();
      expect(ema([10, 11, 12], -1)).toBeNull();
    });

    it('supports rolling calculation with prevEma', () => {
      const values = [10, 11, 12, 13, 14, 15];
      const fullResult = ema(values, 3);
      const rollingResult = ema([15], 3, fullResult!);

      expect(rollingResult).not.toBeNull();
      expect(typeof rollingResult).toBe('number');
      expect(rollingResult).toBeGreaterThan(10);
    });
  });

  describe('Weighted Moving Average (WMA)', () => {
    it('calculates WMA correctly', () => {
      expect(wma([10, 11, 12], 3)).toBe(11.333333333333334);
      expect(wma([10, 11, 12], 3)).toBeCloseTo(11.33, 2);
    });

    it('gives more weight to recent values', () => {
      const values = [10, 15, 20];
      const result = wma(values, 3);
      expect(result).toBeGreaterThan(15); // WMA should be closer to 20 than simple average (15)
      expect(result).toBeCloseTo(16.67, 2);
    });

    it('returns null for insufficient data', () => {
      expect(wma([10, 11], 3)).toBeNull();
      expect(wma([], 3)).toBeNull();
    });

    it('returns null for invalid period', () => {
      expect(wma([10, 11, 12], 0)).toBeNull();
      expect(wma([10, 11, 12], -1)).toBeNull();
    });
  });

  describe('Hull Moving Average (HMA)', () => {
    it('calculates HMA correctly with sufficient data', () => {
      const longPrices = Array.from({ length: 50 }, (_, i) => 100 + Math.sin(i / 5) * 10);
      const result = hma(longPrices, 20);
      expect(result).not.toBeNull();
      expect(typeof result).toBe('number');
    });

    it('returns null for insufficient data', () => {
      expect(hma([10, 11, 12], 20)).toBeNull();
      expect(hma([], 20)).toBeNull();
    });

    it('handles different periods', () => {
      const longPrices = Array.from({ length: 100 }, (_, i) => 100 + i * 0.1);
      const result20 = hma(longPrices, 20);
      const result10 = hma(longPrices, 10);

      expect(result20).not.toBeNull();
      expect(result10).not.toBeNull();
    });
  });

  describe('Kaufman Adaptive Moving Average (KAMA)', () => {
    it('calculates KAMA correctly with sufficient data', () => {
      const longPrices = Array.from({ length: 50 }, (_, i) => 100 + Math.sin(i / 5) * 10);
      const result = kama(longPrices, 10, 2, 30);
      expect(result).not.toBeNull();
      expect(typeof result).toBe('number');
    });

    it('returns null for insufficient data', () => {
      expect(kama([10, 11, 12], 10)).toBeNull();
      expect(kama([], 10)).toBeNull();
    });

    it('supports rolling calculation with prevKama', () => {
      const longPrices = Array.from({ length: 50 }, (_, i) => 100 + Math.sin(i / 5) * 10);
      const fullResult = kama(longPrices, 10, 2, 30);

      const newPrices = [...longPrices, longPrices[longPrices.length - 1] + 1];
      const rollingResult = kama(newPrices, 10, 2, 30, fullResult!);

      expect(rollingResult).not.toBeNull();
      expect(typeof rollingResult).toBe('number');
    });

    it('handles different parameters', () => {
      const longPrices = Array.from({ length: 50 }, (_, i) => 100 + i * 0.1);
      const result1 = kama(longPrices, 10, 2, 30);
      const result2 = kama(longPrices, 10, 5, 20);

      expect(result1).not.toBeNull();
      expect(result2).not.toBeNull();
      expect(result1).not.toBe(result2);
    });

    it('handles flat market (zero volatility)', () => {
      const flatPrices = Array(50).fill(100);
      const result = kama(flatPrices, 10, 2, 30);
      expect(result).not.toBeNull();
      expect(result).toBeCloseTo(100, 1);
    });
  });

  describe('Edge Cases', () => {
    it('handles arrays with identical values', () => {
      const flatPrices = Array(20).fill(100);

      expect(sma(flatPrices, 5)).toBe(100);
      expect(ema(flatPrices, 5)).toBe(100);
      expect(wma(flatPrices, 5)).toBe(100);
    });

    it('handles arrays with increasing values', () => {
      const increasingPrices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      const smaResult = sma(increasingPrices, 5);
      const emaResult = ema(increasingPrices, 5);
      const wmaResult = wma(increasingPrices, 5);

      expect(smaResult).toBe(8); // (6+7+8+9+10)/5
      expect(smaResult).not.toBeNull();
      expect(emaResult).not.toBeNull();
      expect(wmaResult).not.toBeNull();
    });

    it('handles arrays with decreasing values', () => {
      const decreasingPrices = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

      const smaResult = sma(decreasingPrices, 5);
      const emaResult = ema(decreasingPrices, 5);
      const wmaResult = wma(decreasingPrices, 5);

      expect(smaResult).toBe(3); // Latest 5 values: [5,4,3,2,1], average = 3
      expect(smaResult).not.toBeNull();
      expect(emaResult).not.toBeNull();
      expect(wmaResult).not.toBeNull();
    });

    it('handles very short arrays', () => {
      expect(sma([10], 1)).toBe(10);
      expect(sma([10], 2)).toBeNull();

      expect(ema([10], 1)).toBe(10);
      expect(ema([10], 2)).toBeNull();

      expect(wma([10], 1)).toBe(10);
      expect(wma([10], 2)).toBeNull();
    });
  });
});