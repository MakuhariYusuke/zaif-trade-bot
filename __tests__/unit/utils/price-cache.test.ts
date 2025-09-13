import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
const ORIG = { ...process.env };
function setEnv(key: string, val: string){ process.env[key] = val; }

describe('utils/price-cache', () => {
  const TMP = path.resolve(process.cwd(), 'tmp-test-price-cache');
  const file = path.join(TMP, 'price_cache.json');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
    process.env = { ...ORIG };
    setEnv('PRICE_CACHE_FILE', file);
    setEnv('PRICE_CACHE_MAX', '10');
  });

  it('loadPriceCache returns [] when file missing', async ()=>{
    const { loadPriceCache } = await import('../../../src/utils/price-cache');
    expect(loadPriceCache()).toEqual([]);
  });

  it('appendPriceSamples writes and getPriceSeries returns latest first', async ()=>{
    const mod = await import('../../../src/utils/price-cache');
    const now = Date.now();
    mod.appendPriceSamples([{ ts: now-2, price: 100 }, { ts: now-1, price: 110 }]);
    const series = mod.getPriceSeries(2);
    expect(series).toEqual([110, 100]);
  });
});
