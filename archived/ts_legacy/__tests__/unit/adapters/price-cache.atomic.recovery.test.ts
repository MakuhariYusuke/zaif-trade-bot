import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { getEventBus, setEventBus } from '../../../ztb/application/events/bus';

describe('adapters/price-cache atomic recovery', () => {
  const ROOT = process.cwd();
  const TMP = path.resolve(ROOT, `tmp-test-price-cache-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  const file = path.join(TMP, 'price_cache.json');

  beforeEach(() => {
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
    process.env.PRICE_CACHE_FILE = file;
    process.env.PRICE_CACHE_MAX = '10';
    // reset event bus handlers
    setEventBus(new (getEventBus().constructor as any)());
  });

  it('recovers from corrupted JSON and emits single CACHE_ERROR', async () => {
    const { appendPriceSamples, loadPriceCache, resetPriceCache, getPriceSeries } = await import('../../../ztb/utils/price-cache');
    // write valid samples, then corrupt the file
    const base = Date.now();
    appendPriceSamples([{ ts: base - 2, price: 100 }, { ts: base - 1, price: 110 }]);
    // Corrupt the file with partial JSON
    fs.writeFileSync(file, '[{"ts":', 'utf8');

    // capture events
    const bus = getEventBus();
    const errors: any[] = [];
    bus.subscribe('EVENT/ERROR' as any, (ev: any) => {
      if (ev?.code === 'CACHE_ERROR' || ev?.errorCode === 'CACHE_ERROR') errors.push(ev);
    });

    // Force reload path
    resetPriceCache();
    const loaded = loadPriceCache();
    expect(Array.isArray(loaded)).toBe(true);
    // On corrupted file, loader swallows error and initializes empty cache
    expect(loaded.length).toBe(0);
    // Should have emitted one CACHE_ERROR
    if (errors.length !== 1) {
      console.error('[DIAG][price-cache] expected 1 CACHE_ERROR, got', errors.length, errors);
    }
    expect(errors.length).toBe(1);

    // Now append again and verify normal behavior resumes
    appendPriceSamples([{ ts: base + 1, price: 120 }]);
    const series = getPriceSeries(3);
    expect(series[0]).toBe(120);
  });
});
