import { describe, it, expect, beforeEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('report-summary formatted fields', () => {
  const TMP = path.resolve(process.cwd(), 'tmp-test-report');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
  });

  it('includes summaryText and top3Table', async () => {
    const d = new Date().toISOString().slice(0,10);
    fs.writeFileSync(path.join(TMP, 'stats-live.json'), JSON.stringify({ date: d, data: [
      { pair: 'btc_jpy', stats: { realizedPnl: 10, winRate: 0.6 } },
      { pair: 'eth_jpy', stats: { realizedPnl: 5, winRate: 0.55 } },
      { pair: 'xrp_jpy', stats: { realizedPnl: 1, winRate: 0.51 } }
    ] }));
    fs.writeFileSync(path.join(TMP, 'stats-diff-live.json'), JSON.stringify({ values: { trades: 10, wins: 6 }, diff: { realizedPnl: 16 } }));
    const readFileSyncOrig = fs.readFileSync;
    const writeFileSyncOrig = fs.writeFileSync;
    const existsSyncOrig = fs.existsSync;
    (fs.readFileSync as any) = ((p: string, enc?: any) => { const fp = path.isAbsolute(p) ? p : path.join(TMP, p); return readFileSyncOrig(fp, enc); }) as any;
    (fs.writeFileSync as any) = ((p: string, data: any, opts?: any) => { const fp = path.isAbsolute(p) ? p : path.join(TMP, p); return writeFileSyncOrig(fp, data, opts); }) as any;
    (fs.existsSync as any) = ((p: string) => { const fp = path.isAbsolute(p) ? p : path.join(TMP, p); return existsSyncOrig(fp); }) as any;
    process.argv = ['node','script','--source','live'];
    vi.resetModules();
    await import('../../../src/tools/report-summary');
    const outPath = path.join(TMP,'report-summary-live.json');
    await new Promise(res => setTimeout(res, 30));
    const out = JSON.parse(fs.readFileSync(outPath,'utf8'));
    expect(out.summaryText).toContain('Total PnL');
    expect(out.top3Table).toContain('Pair | PnL | WinRate');
  });
});
