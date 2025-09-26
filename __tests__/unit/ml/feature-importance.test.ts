import { describe, it, beforeEach, expect } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('tools/ml/feature-importance', () => {
  const TMP = path.resolve(process.cwd(), 'tmp-test-ml');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
  });

  it('produces JSON and CSV outputs', async () => {
    const ds = path.join(TMP, 'ml-dataset.jsonl');
    // write minimal dataset with numeric features and labels
    const lines = [
      { ts: 1, price: 100, rsi: 30, sma_short: 10, sma_long: 20, pnl: -1, win: 0 },
      { ts: 2, price: 110, rsi: 70, sma_short: 21, sma_long: 20, pnl: 2, win: 1 },
      { ts: 3, price: 105, rsi: 50, sma_short: 19, sma_long: 20, pnl: 0.5, win: 1 },
    ];
    fs.writeFileSync(ds, lines.map(o=>JSON.stringify(o)).join('\n')+'\n');
    process.argv = ['node','script','--dataset', ds, '--out-json', path.join(TMP,'fi.json'), '--out-csv', path.join(TMP,'fi.csv')];
    await import('../../../ztb/tools/ml/feature-importance');
    // wait a moment for async stream end writes
    const outJ = path.join(TMP,'fi.json');
    const outC = path.join(TMP,'fi.csv');
    for (let i=0;i<10;i++){
      if (fs.existsSync(outJ) && fs.existsSync(outC)) break;
      await new Promise(r=>setTimeout(r, 10));
    }
    const j = JSON.parse(fs.readFileSync(outJ,'utf8'));
    expect(j.top_features && Array.isArray(j.top_features)).toBe(true);
    expect(fs.existsSync(path.join(TMP,'fi.csv'))).toBe(true);
  });
});
