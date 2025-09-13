import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { appendFillPnl, appendSummary, loadDaily, summarizeDaily } from '../../../src/utils/daily-stats';

const TMP = path.resolve(process.cwd(), 'tmp-test-stats');
function today(){ return new Date().toISOString().slice(0,10); }

describe('daily-stats metrics', () => {
  const date = today();
  const pair = 'btc_jpy';
  const statsDir = path.join(TMP, 'logs');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(statsDir, { recursive: true });
    process.env.STATS_DIR = statsDir;
  });

  it('maxDrawdown and maxConsecLosses update with mixed pnl', () => {
    appendFillPnl(date, 10, pair);
    appendFillPnl(date, -5, pair);
    appendFillPnl(date, -7, pair);
    const agg = loadDaily(date, pair);
    expect(agg.maxDrawdown).toBeGreaterThan(0);
    expect(agg.maxConsecLosses).toBeGreaterThanOrEqual(2);
  });

  it('avgHoldSec accumulates from durationMs', () => {
    appendSummary(date, { requestId:'r', side:'BUY', intendedQty:1, filledQty:1, avgExpectedPrice:1, avgFillPrice:1, slippagePct:0, durationMs: 3000, submitRetryCount:0, pollRetryCount:0, cancelRetryCount:0, nonceRetryCount:0, totalRetryCount:0, filledCount:1, pair } as any);
    appendSummary(date, { requestId:'r2', side:'BUY', intendedQty:1, filledQty:1, avgExpectedPrice:1, avgFillPrice:1, slippagePct:0, durationMs: 1000, submitRetryCount:0, pollRetryCount:0, cancelRetryCount:0, nonceRetryCount:0, totalRetryCount:0, filledCount:1, pair } as any);
    const sum = summarizeDaily(date, pair);
    expect(sum.avgHoldSec).toBeGreaterThan(0);
  });
});
