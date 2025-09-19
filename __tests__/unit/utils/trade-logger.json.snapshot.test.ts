import { describe, it, beforeEach, expect } from 'vitest';
import fs from 'fs';
import path from 'path';
import { logTradeError, logTradeInfo, logSignal, generateDailyReport } from '../../../src/utils/trade-logger';

describe('trade-logger JSON mode snapshot', () => {
  let tmpDir: string;
  beforeEach(()=>{
    tmpDir = fs.mkdtempSync(path.join(process.cwd(), 'tmp-trade-log-json-'));
    process.env.LOG_DIR = tmpDir;
    process.env.LOG_JSON = '1';
  });

  it('writes JSON lines (INFO/ERROR/SIGNAL) snapshot stable', async () => {
    logSignal('strategy_entry', { side: 'buy', strength: 0.8 });
    logTradeInfo('queued_order', { id: 'OID123', qty: 0.01 });
    logTradeError('failed_execution', { reason: 'timeout', retry: true });
    const today = new Date().toISOString().slice(0,10);
    const file = path.join(tmpDir, `trades-${today}.log`);
    for (let i=0;i<40 && !fs.existsSync(file);i++) await new Promise(r=>setTimeout(r,5));
    expect(fs.existsSync(file)).toBe(true);
    const raw = fs.readFileSync(file, 'utf8').trim().split(/\n+/).filter(Boolean);
    const norm = raw.map(l=>{ const o = JSON.parse(l); o.ts = 'T'; return o; });
    expect(norm).toMatchSnapshot();
    const rep = generateDailyReport(today);
    expect(rep.signals).toBe(1);
  });
});
