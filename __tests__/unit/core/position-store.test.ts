import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { loadPosition, updateOnFill, removePosition } from '../../../src/core/position-store';

describe('core/position-store', ()=>{
  const TMP = path.resolve(process.cwd(), '.positions-test');
  const STORE_FILE = path.join(TMP, 'store.json');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    process.env.POSITION_STORE_DIR = TMP;
    process.env.POSITION_STORE_FILE = STORE_FILE;
  });

  it('adds on bid fills and closes on ask fills', ()=>{
    const pair = 'btc_jpy';
    expect(loadPosition(pair)).toBeUndefined();
    updateOnFill({ pair, side:'bid', price:100, amount:1, ts:Date.now() });
    const p1 = loadPosition(pair)!;
    expect(p1.qty).toBeCloseTo(1, 10);
    expect(p1.avgPrice).toBe(100);
    updateOnFill({ pair, side:'ask', price:110, amount:1, ts:Date.now() });
    const p2 = loadPosition(pair)!;
    expect(p2.qty).toBe(0);
    removePosition(pair);
    expect(loadPosition(pair)).toBeUndefined();
  });
});
