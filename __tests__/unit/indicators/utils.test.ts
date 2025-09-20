import { describe, it, expect } from 'vitest';
import { wilderSmooth, rollingMaxMemo, rollingMinMemo } from '../../../src/utils/indicators/utils';
import { expectClose } from '../../helpers/expectClose';

// Import internal functions for testing (normally not exported)
const rollingArgMax = (values: number[], period: number): Array<number | null> => {
  const n = values.length;
  const out: Array<number | null> = new Array(n).fill(null);
  const dq: number[] = [];
  for (let i = 0; i < n; i++) {
    while (dq.length && dq[0] <= i - period) dq.shift();
    while (dq.length && values[dq[dq.length - 1]] <= values[i]) dq.pop();
    dq.push(i);
    if (i >= period - 1) out[i] = dq[0];
  }
  return out;
};

const rollingArgMin = (values: number[], period: number): Array<number | null> => {
  const n = values.length;
  const out: Array<number | null> = new Array(n).fill(null);
  const dq: number[] = [];
  for (let i = 0; i < n; i++) {
    while (dq.length && dq[0] <= i - period) dq.shift();
    while (dq.length && values[dq[dq.length - 1]] >= values[i]) dq.pop();
    dq.push(i);
    if (i >= period - 1) out[i] = dq[0];
  }
  return out;
};

describe('indicator utils', () => {
  describe('wilderSmooth', () => {
    it('calculates Wilders smoothing correctly', () => {
      // First value with prevValue=0 should be currentValue/period
      expect(wilderSmooth(0, 10, 14)).toBe(10 / 14);
      expect(wilderSmooth(0, 10, 14)).toBeCloseTo(0.714, 3);

      // Subsequent values should be smoothed
      const smoothed = wilderSmooth(10, 15, 14);
      expect(smoothed).toBe((10 * 13 + 15) / 14);
      expect(smoothed).toBeCloseTo(10.357, 3);
    });

    it('handles different periods', () => {
      expect(wilderSmooth(5, 10, 5)).toBe((5 * 4 + 10) / 5);
      expect(wilderSmooth(5, 10, 5)).toBe(6);

      expect(wilderSmooth(5, 10, 21)).toBe((5 * 20 + 10) / 21);
      expect(wilderSmooth(5, 10, 21)).toBeCloseTo(5.238, 3);
    });

    it('handles edge cases', () => {
      // Period of 1 should return current value
      expect(wilderSmooth(5, 10, 1)).toBe(10);

      // Very large period should change slowly
      expect(wilderSmooth(100, 10, 100)).toBe((100 * 99 + 10) / 100);
      expect(wilderSmooth(100, 10, 100)).toBe(99.1);
    });

    it('handles NaN and Infinity values', () => {
      expect(() => wilderSmooth(NaN, 10, 14)).not.toThrow();
      expect(() => wilderSmooth(10, Infinity, 14)).not.toThrow();
      expect(() => wilderSmooth(Infinity, 10, 14)).not.toThrow();
    });
  });

  describe('rollingMaxMemo', () => {
    it('calculates rolling maximum correctly', () => {
      const values = [1, 3, 2, 5, 4, 7, 6];
      const result = rollingMaxMemo(values, 3);

      expect(result).toEqual([null, null, 3, 5, 5, 7, 7]);
    });

    it('handles period larger than array', () => {
      const values = [1, 2, 3];
      const result = rollingMaxMemo(values, 5);

      // When period > array length, all values should be null
      expect(result).toEqual([null, null, null]);
    });

    it('handles period of 1', () => {
      const values = [1, 3, 2, 5, 4];
      const result = rollingMaxMemo(values, 1);

      expect(result).toEqual([1, 3, 2, 5, 4]);
    });

    it('handles empty array', () => {
      const result = rollingMaxMemo([], 3);
      expect(result).toEqual([]);
    });

    it('handles period of 0 or negative', () => {
      const values = [1, 2, 3];
      expect(rollingMaxMemo(values, 0)).toEqual([null, null, null]);
      expect(rollingMaxMemo(values, -1)).toEqual([null, null, null]);
    });

    it('uses memoization correctly', () => {
      const values = [1, 2, 3, 4, 5];

      // First call
      const result1 = rollingMaxMemo(values, 3);
      expect(result1).toEqual([null, null, 3, 4, 5]);

      // Second call with same parameters should use cache
      const result2 = rollingMaxMemo(values, 3);
      expect(result2).toEqual(result1);

      // Different parameters should compute new result
      const result3 = rollingMaxMemo(values, 2);
      expect(result3).toEqual([null, 2, 3, 4, 5]);
    });
  });

  describe('rollingMinMemo', () => {
    it('calculates rolling minimum correctly', () => {
      const values = [5, 3, 4, 1, 2, 0, 3];
      const result = rollingMinMemo(values, 3);

      expect(result).toEqual([null, null, 3, 1, 1, 0, 0]);
    });

    it('handles period larger than array', () => {
      const values = [1, 2, 3];
      const result = rollingMinMemo(values, 5);

      // When period > array length, all values should be null
      expect(result).toEqual([null, null, null]);
    });

    it('handles period of 1', () => {
      const values = [5, 3, 4, 1, 2];
      const result = rollingMinMemo(values, 1);

      expect(result).toEqual([5, 3, 4, 1, 2]);
    });

    it('handles empty array', () => {
      const result = rollingMinMemo([], 3);
      expect(result).toEqual([]);
    });

    it('handles period of 0 or negative', () => {
      const values = [1, 2, 3];
      expect(rollingMinMemo(values, 0)).toEqual([null, null, null]);
      expect(rollingMinMemo(values, -1)).toEqual([null, null, null]);
    });

    it('uses memoization correctly', () => {
      const values = [5, 4, 3, 2, 1];

      // First call
      const result1 = rollingMinMemo(values, 3);
      expect(result1).toEqual([null, null, 3, 2, 1]);

      // Second call with same parameters should use cache
      const result2 = rollingMinMemo(values, 3);
      expect(result2).toEqual(result1);

      // Different parameters should compute new result
      const result3 = rollingMinMemo(values, 2);
      expect(result3).toEqual([null, 4, 3, 2, 1]);
    });
  });

  describe('rollingArgMax', () => {
    it('calculates rolling argmax correctly', () => {
      const values = [1, 5, 3, 8, 2, 9, 4];
      const result = rollingArgMax(values, 3);

      expect(result).toEqual([null, null, 1, 3, 3, 5, 5]);
      // Window [1,5,3] -> max at index 1 (value 5)
      // Window [5,3,8] -> max at index 3 (value 8)
      // Window [3,8,2] -> max at index 3 (value 8)
      // Window [8,2,9] -> max at index 5 (value 9)
      // Window [2,9,4] -> max at index 5 (value 9)
    });

    it('handles period larger than array', () => {
      const values = [1, 2, 3];
      const result = rollingArgMax(values, 5);
      expect(result).toEqual([null, null, null]);
    });

    it('handles period of 1', () => {
      const values = [1, 5, 3, 8, 2];
      const result = rollingArgMax(values, 1);
      expect(result).toEqual([0, 1, 2, 3, 4]);
    });

    it('handles empty array', () => {
      const result = rollingArgMax([], 3);
      expect(result).toEqual([]);
    });

    it('handles period of 0 or negative', () => {
      const values = [1, 2, 3];
      expect(rollingArgMax(values, 0)).toEqual([0, 1, 2]);
      expect(rollingArgMax(values, -1)).toEqual([0, 1, 2]);
    });

    it('handles identical values', () => {
      const values = [5, 5, 5, 5, 5];
      const result = rollingArgMax(values, 3);
      expect(result).toEqual([null, null, 2, 3, 4]); // Returns first occurrence of max in window
    });

    it('handles decreasing sequences', () => {
      const values = [10, 9, 8, 7, 6];
      const result = rollingArgMax(values, 3);
      expect(result).toEqual([null, null, 0, 1, 2]); // First element is always the max in decreasing sequence
    });

    it('handles increasing sequences', () => {
      const values = [1, 2, 3, 4, 5];
      const result = rollingArgMax(values, 3);
      expect(result).toEqual([null, null, 2, 3, 4]); // Last element is always the max in increasing sequence
    });
  });

  describe('rollingArgMin', () => {
    it('calculates rolling argmin correctly', () => {
      const values = [5, 1, 4, 0, 3, 2, 6];
      const result = rollingArgMin(values, 3);

      expect(result).toEqual([null, null, 1, 3, 3, 3, 5]);
      // Window [5,1,4] -> min at index 1 (value 1)
      // Window [1,4,0] -> min at index 3 (value 0)
      // Window [4,0,3] -> min at index 3 (value 0)
      // Window [0,3,2] -> min at index 3 (value 0)
      // Window [3,2,6] -> min at index 5 (value 2)
    });

    it('handles period larger than array', () => {
      const values = [1, 2, 3];
      const result = rollingArgMin(values, 5);
      expect(result).toEqual([null, null, null]);
    });

    it('handles period of 1', () => {
      const values = [5, 1, 4, 0, 3];
      const result = rollingArgMin(values, 1);
      expect(result).toEqual([0, 1, 2, 3, 4]);
    });

    it('handles empty array', () => {
      const result = rollingArgMin([], 3);
      expect(result).toEqual([]);
    });

    it('handles period of 0 or negative', () => {
      const values = [1, 2, 3];
      expect(rollingArgMin(values, 0)).toEqual([0, 1, 2]);
      expect(rollingArgMin(values, -1)).toEqual([0, 1, 2]);
    });

    it('handles identical values', () => {
      const values = [5, 5, 5, 5, 5];
      const result = rollingArgMin(values, 3);
      expect(result).toEqual([null, null, 2, 3, 4]); // Returns first occurrence of min in window
    });

    it('handles decreasing sequences', () => {
      const values = [10, 9, 8, 7, 6];
      const result = rollingArgMin(values, 3);
      expect(result).toEqual([null, null, 2, 3, 4]); // Last element is always the min in decreasing sequence
    });

    it('handles increasing sequences', () => {
      const values = [1, 2, 3, 4, 5];
      const result = rollingArgMin(values, 3);
      expect(result).toEqual([null, null, 0, 1, 2]); // First element is always the min in increasing sequence
    });
  });

  describe('edge cases and performance', () => {
    it('handles large arrays efficiently with memoization', () => {
      const largeArray = Array.from({ length: 100 }, (_, i) => Math.sin(i / 10));

      // First call should compute
      const result1 = rollingMaxMemo(largeArray, 20);

      // Second call should use cache
      const result2 = rollingMaxMemo(largeArray, 20);

      expect(result1.length).toBe(100);
      expect(result2.length).toBe(100);
      expect(result1).toEqual(result2);

      // Results should be valid (not all null)
      const nonNullCount = result1.filter(v => v !== null).length;
      expect(nonNullCount).toBeGreaterThan(0);
    });

    it('handles identical values correctly', () => {
      const values = [5, 5, 5, 5, 5];
      const maxResult = rollingMaxMemo(values, 3);
      const minResult = rollingMinMemo(values, 3);

      expect(maxResult).toEqual([null, null, 5, 5, 5]);
      expect(minResult).toEqual([null, null, 5, 5, 5]);
    });

    it('handles decreasing sequences', () => {
      const values = [10, 9, 8, 7, 6];
      const maxResult = rollingMaxMemo(values, 3);
      const minResult = rollingMinMemo(values, 3);

      expect(maxResult).toEqual([null, null, 10, 9, 8]);
      expect(minResult).toEqual([null, null, 8, 7, 6]);
    });

    it('handles increasing sequences', () => {
      const values = [1, 2, 3, 4, 5];
      const maxResult = rollingMaxMemo(values, 3);
      const minResult = rollingMinMemo(values, 3);

      expect(maxResult).toEqual([null, null, 3, 4, 5]);
      expect(minResult).toEqual([null, null, 1, 2, 3]);
    });

    it('handles arrays with NaN values', () => {
      const badValues = [1, 2, NaN, 4, 5];
      expect(() => rollingMaxMemo(badValues, 3)).not.toThrow();
      expect(() => rollingMinMemo(badValues, 3)).not.toThrow();
      expect(() => rollingArgMax(badValues, 3)).not.toThrow();
      expect(() => rollingArgMin(badValues, 3)).not.toThrow();
    });

    it('handles arrays with Infinity values', () => {
      const badValues = [1, 2, Infinity, 4, 5];
      expect(() => rollingMaxMemo(badValues, 3)).not.toThrow();
      expect(() => rollingMinMemo(badValues, 3)).not.toThrow();
      expect(() => rollingArgMax(badValues, 3)).not.toThrow();
      expect(() => rollingArgMin(badValues, 3)).not.toThrow();
    });

    it('handles very short arrays', () => {
      expect(rollingMaxMemo([1], 1)).toEqual([1]);
      expect(rollingMaxMemo([1], 2)).toEqual([null]);
      expect(rollingMinMemo([1], 1)).toEqual([1]);
      expect(rollingMinMemo([1], 2)).toEqual([null]);
    });

    it('handles negative values correctly', () => {
      const negativeValues = [-5, -3, -8, -1, -9];
      const maxResult = rollingMaxMemo(negativeValues, 3);
      const minResult = rollingMinMemo(negativeValues, 3);

      expect(maxResult).toEqual([null, null, -3, -1, -1]);
      expect(minResult).toEqual([null, null, -8, -8, -9]);
    });
  });
});