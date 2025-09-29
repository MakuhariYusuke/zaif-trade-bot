import { describe, it, expect, beforeEach } from 'vitest';
import { SmartPreprocessor } from 'ztb/evaluation/preprocess';
import { CommonPreprocessor } from 'ztb/features/base';
import * as pd from 'pandas-js';

describe('SmartPreprocessor', () => {
  let preprocessor: SmartPreprocessor;
  let sampleData: any;

  beforeEach(() => {
    preprocessor = new SmartPreprocessor();

    // Create sample OHLC data
    const dates = pd.date_range('2020-01-01', 100, 'D');
    sampleData = new pd.DataFrame({
      'date': dates.toArray(),
      'open': Array.from({length: 100}, () => 100 + Math.random() * 10),
      'high': Array.from({length: 100}, () => 105 + Math.random() * 10),
      'low': Array.from({length: 100}, () => 95 + Math.random() * 10),
      'close': Array.from({length: 100}, () => 100 + Math.random() * 10),
      'volume': Array.from({length: 100}, () => Math.floor(Math.random() * 10000) + 1000)
    });
    const preprocessedData = CommonPreprocessor.preprocess(sampleData);
    sampleData = preprocessedData;
    sampleData = CommonPreprocessor.preprocess(sampleData);
  });

  describe('preprocess', () => {
    it('should calculate only required ema columns', () => {
      const required = ['ema:12', 'ema:26'];
      const result = preprocessor.preprocess(sampleData, required);

      expect(result.columns).toContain('ema_12');
      expect(result.columns).toContain('ema_26');
      expect(result.columns).not.toContain('ema_5');
      expect(result.columns).not.toContain('ema_50');
    });

    it('should calculate only required rolling columns', () => {
      const required = ['rolling:10:mean', 'rolling:20:std'];
      const result = preprocessor.preprocess(sampleData, required);

      expect(result.columns).toContain('rolling_mean_10');
      expect(result.columns).toContain('rolling_std_20');
      expect(result.columns).not.toContain('rolling_mean_5');
      expect(result.columns).not.toContain('rolling_std_30');
    });

    it('should handle mixed requirements', () => {
      const required = ['ema:12', 'rolling:20:mean', 'close'];
      const result = preprocessor.preprocess(sampleData, required);

      expect(result.columns).toContain('ema_12');
      expect(result.columns).toContain('rolling_mean_20');
      expect(result.columns).toContain('close');
      expect(result.columns).not.toContain('ema_26');
      expect(result.columns).not.toContain('rolling_std_20');
    });

    it('should handle empty requirements', () => {
      const required: string[] = [];
      const result = preprocessor.preprocess(sampleData, required);

      // Should still include basic columns
      expect(result.columns).toContain('close');
      expect(result.columns).toContain('open');
      expect(result.columns).toContain('high');
      expect(result.columns).toContain('low');
      expect(result.columns).toContain('volume');
    });

    it('should cache calculations', () => {
      const required = ['ema:12', 'ema:26'];

      // First call
      const result1 = preprocessor.preprocess(sampleData, required);

      // Second call with same requirements should use cache
      const result2 = preprocessor.preprocess(sampleData, required);
      expect(result1.to_json()).toBe(result2.to_json());
      expect(result1.equals(result2)).toBe(true);
    });
  });
});
