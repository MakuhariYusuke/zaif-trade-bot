import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';
import { CorePositionStore } from '../../../ztb/adapters/position-store-fs';

describe('position-store-fs atomic writes', () => {
  const TMP_ROOT = path.resolve(process.cwd(), 'tmp-test-position-store');
  let TMP_DIR = '';

  beforeEach(() => {
    TMP_DIR = path.join(TMP_ROOT, `${Date.now()}-${Math.random().toString(36).slice(2)}`);
    fs.mkdirSync(TMP_DIR, { recursive: true });
    process.env.POSITION_STORE_DIR = TMP_DIR;
  });

  afterEach(() => {
    try { fs.rmSync(TMP_DIR, { recursive: true, force: true }); } catch {}
    delete process.env.POSITION_STORE_DIR;
  });

  it('keeps file always parseable across rapid updates', async () => {
    const store = new CorePositionStore();
    const pair = 'btc_jpy';
    const iterations = 50;
    for (let i = 0; i < iterations; i++) {
      const qty = i * 0.001;
      await store.update(pair, { pair, qty, avgPrice: 1000000 + i, dcaCount: i, openOrderIds: [] } as any);
      const file = path.join(TMP_DIR, `${pair}.json`);
      expect(fs.existsSync(file)).toBe(true);
      const txt = fs.readFileSync(file, 'utf8');
      // must be valid JSON at every step
      const obj = JSON.parse(txt);
      expect(obj.qty).toBe(qty);
    }
  });

  it('on rename failure, original file remains intact (no corruption)', async () => {
    const store = new CorePositionStore();
    const pair = 'eth_jpy';
    const file = path.join(TMP_DIR, `${pair}.json`);
    fs.writeFileSync(file, JSON.stringify({ pair, qty: 1, avgPrice: 100, dcaCount: 0, openOrderIds: [] }, null, 2));

    const spy = vi.spyOn(fs, 'renameSync').mockImplementation(() => { throw new Error('inject rename failure'); });
    await store.update(pair, { pair, qty: 2 } as any).catch(()=>{/* ignored by store's logger */});
    spy.mockRestore();
    const txt = fs.readFileSync(file, 'utf8');
    const obj = JSON.parse(txt);
    expect(obj.qty).toBe(1);
  });
});
