import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { appendFillPnl, incBuyEntry, incSellEntry, loadDaily } from '../../../ztb/utils/daily-stats';

const TMP = path.resolve(process.cwd(), 'tmp-test-stats-core');

function today(){ return new Date().toISOString().slice(0,10); }

describe('daily-stats', () => {
  const date = today();
  const pair = 'btc_jpy';
  const statsDir = path.join(TMP, 'logs');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(statsDir, { recursive: true });
    process.env.STATS_DIR = statsDir;
  });

  it('incBuyEntry and appendFillPnl update JSON', () => {
    const before = loadDaily(date, pair);
    incBuyEntry(date, pair);
    appendFillPnl(date, 123.45, pair);
    const after = loadDaily(date, pair);
    expect((after.buyEntries||0)).toBe((before.buyEntries||0)+1);
    expect(after.realizedPnl).toBe((before.realizedPnl||0)+123.45);
  });

  it('incSellEntry twice and negative pnl accumulates', () => {
    const before = loadDaily(date, pair);
    incSellEntry(date, pair);
    incSellEntry(date, pair);
    appendFillPnl(date, -10, pair);
    const after = loadDaily(date, pair);
    expect((after.sellEntries||0)).toBe((before.sellEntries||0)+2);
    expect(after.realizedPnl).toBeCloseTo((before.realizedPnl||0)-10, 10);
  });
});
