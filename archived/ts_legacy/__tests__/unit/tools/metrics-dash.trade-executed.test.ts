import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import { runOnceCollect } from '../../../ztb/tools/metrics-dash';

function writeLog(entries: any[]) {
  const lines = entries.map(e => JSON.stringify(e)).join('\n');
  fs.writeFileSync('logs/test-metrics.log', lines + '\n');
}

describe('metrics-dash trade executed aggregation', () => {
  it('aggregates executed and failed trade events with pnl and guards', () => {
    fs.mkdirSync('logs', { recursive: true });
    const base = { ts: new Date().toISOString(), level: 'INFO', category: 'EVENT', message: 'published' };
    writeLog([
      { ...base, data: [{ type: 'EVENT/TRADE_EXECUTED', success: true, pnl: 10, requestId: '1' }] },
      { ...base, data: [{ type: 'EVENT/TRADE_EXECUTED', success: true, pnl: -5, requestId: '2' }] },
      { ...base, data: [{ type: 'EVENT/TRADE_EXECUTED', success: false, reason: 'SLIPPAGE', requestId: '3' }] },
      { ...base, data: [{ type: 'EVENT/TRADE_EXECUTED', success: false, reason: 'MAX_LOSS', requestId: '4' }] },
    ]);
    const res = runOnceCollect('logs/test-metrics.log', 200);
    expect(res.executedCount).toBe(2);
    expect(res.failCount).toBe(2);
    expect(res.pnlTotal).toBe(5); // 10 + (-5)
    expect(res.guardTrips).toBe(2); // SLIPPAGE + MAX_LOSS
  });
});
