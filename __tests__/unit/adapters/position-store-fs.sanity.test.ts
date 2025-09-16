import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { loadPosition, savePosition, removePosition, type StoredPosition } from '../../../src/adapters/position-store';

describe('position-store-fs sanity', () => {
  const DIR = path.resolve(process.cwd(), '.positions-test');
  const PAIR = 'btc_jpy';
  beforeEach(()=>{
    if (fs.existsSync(DIR)) fs.rmSync(DIR, { recursive: true, force: true });
    fs.mkdirSync(DIR, { recursive: true });
    process.env.POSITION_STORE_DIR = DIR;
    process.env.POSITION_STORE_FILE = path.join(DIR, 'store.json');
  });

  it('saves and loads a position', async () => {
    const pos: StoredPosition = { pair: PAIR, qty: 0.01, avgPrice: 100, dcaCount: 0, openOrderIds: [], side: 'long' };
    await savePosition(pos);
    const loaded = await loadPosition(PAIR) as StoredPosition | undefined;
    expect(loaded).toBeTruthy();
    expect(loaded!.pair).toBe(PAIR);
    expect(loaded!.qty).toBeCloseTo(0.01, 8);
  await removePosition(PAIR);
  await new Promise(res=>setTimeout(res,5));
  const removed = await loadPosition(PAIR);
  expect(removed).toBeUndefined();
  });
});
