import { describe, it, expect, beforeEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';
import { logSignal, logExecution, logTradeError, logTradeInfo, generateDailyReport } from '../../../src/utils/trade-logger';
import { setLoggerContext, clearLoggerContext, logInfo, logWarn, logError } from '../../../src/utils/logger';

describe('trade-logger snapshots', () => {
  let tmpDir: string;
  beforeEach(()=>{
    tmpDir = fs.mkdtempSync(path.join(process.cwd(), 'tmp-trade-log-'));
    process.env.LOG_DIR = tmpDir;
  });

  it('writes SIGNAL/EXECUTION/ERROR/INFO lines and daily report aggregates', async () => {
    logSignal('sig1', { a: 1 });
    logExecution('exec1', { pnl: 5 });
    logTradeError('err1', { reason: 'x' });
    logTradeInfo('info1', { note: true });
  const today = new Date().toISOString().slice(0,10);
  const file = path.join(tmpDir, `trades-${today}.log`);
  for (let i=0;i<40 && !fs.existsSync(file);i++) await new Promise(r=>setTimeout(r,5));
  expect(fs.existsSync(file)).toBe(true);
  const text = fs.readFileSync(file, 'utf8').trim().split(/\n+/).filter(Boolean).map(l=>{ const o = JSON.parse(l); o.ts='T'; return o; });
    expect(text).toMatchSnapshot();
    const rep = generateDailyReport(today);
    expect(rep.trades).toBe(1);
    expect(rep.signals).toBe(1);
    expect(rep.pnlEstimate).toBe(5);
  });

  it('logger context set/clear affects output', () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(()=>{});
    setLoggerContext({ runId: 'RID', phase: 2 });
    logInfo('hello');
    clearLoggerContext(['phase']);
    logWarn('warned');
    clearLoggerContext();
    logError('boom');
    const calls = spy.mock.calls.map(c=>c[0]);
    spy.mockRestore();
    expect(calls.some(l=>/runId=RID/.test(String(l)))).toBe(true);
    expect(calls.some(l=>/phase=2/.test(String(l)))).toBe(true);
    expect(calls.some(l=>/phase=2/.test(String(l)))).toBe(true);
  });
});
