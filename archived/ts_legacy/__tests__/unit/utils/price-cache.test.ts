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
    const { loadPriceCache } = await import('../../../ztb/utils/price-cache');
    expect(loadPriceCache()).toEqual([]);
  });

  it('appendPriceSamples writes and getPriceSeries returns latest first', async ()=>{
    const mod = await import('../../../ztb/utils/price-cache');
    const now = Date.now();
    mod.appendPriceSamples([{ ts: now-2, price: 100 }, { ts: now-1, price: 110 }]);
    const series = mod.getPriceSeries(2);
    expect(series).toEqual([110, 100]);
  });

  it('enforces MAX_ENTRIES by trimming oldest', async ()=>{
    const mod = await import('../../../ztb/utils/price-cache');
    const base = Date.now();
    const samples = Array.from({ length: 20 }, (_,i)=> ({ ts: base - (20-i)*1000, price: 100 + i }));
    mod.appendPriceSamples(samples);
    const cache = JSON.parse(fs.readFileSync(file,'utf8'));
    expect(cache.length).toBe(10);
    const series = mod.getPriceSeries(3);
    expect(series[0]).toBe(100 + 19);
  });
});
