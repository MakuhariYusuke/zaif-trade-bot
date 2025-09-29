import { describe, it, expect, beforeEach } from 'vitest';
import path from 'path';
import fs from 'fs';
import { appendSummary, summarizeDaily } from '../../../ztb/utils/daily-stats';
import { logFeatureSample } from '../../../ztb/utils/features-logger';

describe('utils: daily-stats summarizeDaily and features-logger error handling', () => {
  const TMP = path.resolve(process.cwd(), '.tmp-utils-stats');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
    process.env.STATS_DIR = path.join(TMP, 'logs');
  });

  it('summarizeDaily handles zero trades gracefully', () => {
    const today = new Date().toISOString().slice(0,10);
    const s = summarizeDaily(today, 'btc_jpy');
    expect(s.avgSlippage).toBe(0);
    expect(s.avgRetries).toBe(0);
    expect(s.winRate).toBe(0);
  });

  it('summarizeDaily computes averages without division issues', () => {
    const today = new Date().toISOString().slice(0,10);
    appendSummary(today, { requestId:'r', side:'BUY' as any, intendedQty:1, filledQty:1, avgExpectedPrice:100, avgFillPrice:101, slippagePct: 0.01, durationMs:10, submitRetryCount:1, pollRetryCount:2, cancelRetryCount:0, nonceRetryCount:0, totalRetryCount:3, filledCount:1, pnl: 1, win: true, pair:'btc_jpy' });
    appendSummary(today, { requestId:'r2', side:'BUY' as any, intendedQty:1, filledQty:1, avgExpectedPrice:100, avgFillPrice:100.5, slippagePct: 0.005, durationMs:10, submitRetryCount:0, pollRetryCount:0, cancelRetryCount:0, nonceRetryCount:0, totalRetryCount:0, filledCount:1, pnl: 0.5, win: true, pair:'btc_jpy' });
    const s = summarizeDaily(today, 'btc_jpy');
    expect(s.avgSlippage).toBeCloseTo((0.01+0.005)/2, 10);
    expect(s.avgRetries).toBeCloseTo((3+0)/2, 10);
    expect(s.winRate).toBeCloseTo(1, 10);
  });

  it('features-logger swallows IO errors', () => {
    const today = new Date().toISOString().slice(0,10);
    const dir = path.join(TMP, 'no-perm');
    process.env.FEATURES_LOG_DIR = dir;
    // Create a file where a directory is expected to force mkdir failure
    fs.writeFileSync(dir, 'not a dir');
    // Should not throw despite IO problems
    logFeatureSample({ ts: Date.now(), pair: 'btc_jpy', side: 'bid', price: 100, qty: 0 });
    // clean
    delete process.env.FEATURES_LOG_DIR;
  });
});
