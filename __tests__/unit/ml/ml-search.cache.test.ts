import { describe, it, beforeEach, expect, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('tools/ml/ml-search --use-cache avoids regeneration', () => {
  const TMP = path.resolve(process.cwd(), 'tmp-test-ml');
  beforeEach(()=>{
    vi.resetModules();
    try {
      if (!fs.existsSync(TMP)) fs.mkdirSync(TMP, { recursive: true });
      // remove files only to avoid Windows EPERM on dir handle
      for (const f of fs.readdirSync(TMP)){
        const p = path.join(TMP, f);
        try {
          const st = fs.statSync(p);
          if (st.isDirectory()) fs.rmSync(p, { recursive: true, force: true });
          else fs.unlinkSync(p);
        } catch {}
      }
    } catch {}
  // prepare existing dataset (JSONL)
  fs.writeFileSync(path.join(TMP, 'ml-dataset.jsonl'), '{"pair":"btc_jpy","price":1000000}\n');
  });

  it('uses existing dataset with --use-cache and does not spawn ml-export', async () => {
    const origCwd = process.cwd;
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(TMP);
    const spawnSync = vi.spyOn(require('child_process'), 'spawnSync');
    process.argv = ['node','script','--use-cache'];
    process.env.ML_SEARCH_MODE = 'random';
    process.env.ML_RANDOM_STEPS = '1';
    process.env.PAIR = 'btc_jpy';
    await import('../../../ztb/tools/ml/ml-search');
    // artifacts created
  expect(fs.existsSync(path.join(TMP,'ml-search-results.csv'))).toBe(true);
    expect(fs.existsSync(path.join(TMP,'ml-search-top.json'))).toBe(true);
    // no spawn when cache is used
    expect(spawnSync).not.toHaveBeenCalled();
    cwdSpy.mockRestore();
    (process.cwd as any) = origCwd;
  });
});
