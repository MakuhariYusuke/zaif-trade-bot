import { describe, it, expect, beforeEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';
import { loadDaily, appendFillPnl } from '../../../ztb/utils/daily-stats';

const TMP = path.resolve(process.cwd(), 'tmp-test-stats-edge');

function today(){ return new Date().toISOString().slice(0,10); }

describe('daily-stats edge/error', () => {
  const date = today();
  const pair = 'btc_jpy';
  const statsDir = path.join(TMP, 'logs');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(statsDir, { recursive: true });
    process.env.STATS_DIR = statsDir;
  });

  it('loadDaily returns defaults when file missing', () => {
    const agg = loadDaily(date, pair);
    expect(agg.trades).toBe(0);
    expect(agg.realizedPnl).toBe(0);
  });

  it('loadDaily returns defaults on corrupted JSON', () => {
    const f = path.join(statsDir, 'pairs', pair, `stats-${date}.json`);
    fs.mkdirSync(path.dirname(f), { recursive: true });
    fs.writeFileSync(f, '{ this is not json }');
    const agg = loadDaily(date, pair);
    expect(agg.trades).toBe(0);
    expect(agg.wins).toBe(0);
  });

  it('appendFillPnl swallows write errors gracefully', () => {
    const spy = vi.spyOn(fs, 'writeFileSync').mockImplementationOnce(() => { throw new Error('disk full'); });
    expect(()=> appendFillPnl(date, 1.23, pair)).not.toThrow();
    spy.mockRestore();
  });
});
