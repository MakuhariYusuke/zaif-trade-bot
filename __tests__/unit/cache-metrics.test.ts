import { describe, it, expect, beforeEach, vi } from 'vitest';
import { __resetCacheMetrics, setCacheMetricsInterval, cacheHit, cacheMiss, cacheStale } from '../../src/utils/cache-metrics';

describe('cache-metrics', () => {
  beforeEach(()=>{ __resetCacheMetrics(); setCacheMetricsInterval(100); delete process.env.TEST_MODE; process.env.LOG_LEVEL = 'INFO'; });

  it('emits CACHE/METRICS with hitRate', async () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(()=>{});
  cacheMiss('market:ticker');
  cacheHit('market:ticker');
  cacheHit('market:ticker');
  cacheStale('market:ticker');
  await new Promise(r => setTimeout(r, 120));
  // trigger emission after interval
  cacheHit('market:ticker');
    const called = spy.mock.calls.some(args => String(args[0]).includes('[INFO][CACHE] metrics'));
    expect(called).toBe(true);
    spy.mockRestore();
  });
});
