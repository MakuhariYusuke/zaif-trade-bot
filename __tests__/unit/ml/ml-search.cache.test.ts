import { describe, it, beforeEach, expect, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('tools/ml/ml-search --use-cache avoids regeneration', () => {
  const TMP = path.resolve(process.cwd(), 'tmp-test-ml');
  beforeEach(()=>{
    vi.resetModules();
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
    // prepare existing dataset
    fs.writeFileSync(path.join(TMP, 'ml-dataset.csv'), 'pair,price\n');
  });

  it('uses existing dataset with --use-cache and does not spawn ml-export', async () => {
    const origCwd = process.cwd;
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(TMP);
    const spawnSync = vi.spyOn(require('child_process'), 'spawnSync');
    process.argv = ['node','script','--use-cache'];
    process.env.ML_SEARCH_MODE = 'random';
    process.env.ML_RANDOM_STEPS = '1';
    process.env.PAIR = 'btc_jpy';
    await import('../../../src/tools/ml/ml-search');
    // artifacts created
    expect(fs.existsSync(path.join(TMP,'ml-search-results.csv'))).toBe(true);
    expect(fs.existsSync(path.join(TMP,'ml-search-top.json'))).toBe(true);
    // no spawn when cache is used
    expect(spawnSync).not.toHaveBeenCalled();
    cwdSpy.mockRestore();
    (process.cwd as any) = origCwd;
  });
});
