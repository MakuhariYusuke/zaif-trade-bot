import { describe, it, expect, beforeEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('tools/report-summary', () => {
  const TMP = path.resolve(process.cwd(), 'tmp-test-report');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
  });

  it('generates summary for paper source', async () => {
    const d = new Date().toISOString().slice(0,10);
    fs.writeFileSync(path.join(TMP, 'stats-paper.json'), JSON.stringify({ date: d, data: [{ pair: 'btc_jpy', stats: { realizedPnl: 10, winRate: 0.5, maxDrawdown: 2, maxConsecLosses: 1, avgHoldSec: 3 } }] }));
    fs.writeFileSync(path.join(TMP, 'stats-diff-paper.json'), JSON.stringify({ values: { trades: 2, wins: 1, maxDrawdown: 2 }, diff: { buyEntries: 1, sellEntries: 0, rsiExits: 1, trailExitTotal: 0, realizedPnl: 10 }, pairsDiff: [{ pair:'btc_jpy', diff: { buyEntries:1, sellEntries:0, rsiExits:1, trailExitTotal:0 } }] }));
    const readFileSyncOrig = fs.readFileSync;
    const writeFileSyncOrig = fs.writeFileSync;
    const existsSyncOrig = fs.existsSync;
    // shim fs methods for this import to resolve files under TMP
    (fs.readFileSync as any) = ((p: string, enc?: any) => { const fp = path.isAbsolute(p) ? p : path.join(TMP, p); return readFileSyncOrig(fp, enc); }) as any;
    fs.writeFileSync = ((p: string | Buffer | URL, data: string | NodeJS.ArrayBufferView, opts?: fs.WriteFileOptions) => {
      const fp = path.isAbsolute(p.toString()) ? p : path.join(TMP, p.toString());
      return writeFileSyncOrig(fp, data, opts);
    }) as typeof fs.writeFileSync;
    (fs.existsSync as any) = ((p: string) => { const fp = path.isAbsolute(p) ? p : path.join(TMP, p); return existsSyncOrig(fp); }) as any;
    process.argv = ['node', 'script', '--source', 'paper'];
  vi.resetModules();
    await import('../../../src/tools/report-summary');
    // wait for file using fs.watch
    const outPath = path.join(TMP,'report-summary-paper.json');
    // Wait briefly to ensure file write completes
    await new Promise(res => setTimeout(res, 50));
    if (!fs.existsSync(outPath)) throw new Error('report-summary-paper.json not found');
    const out = JSON.parse(fs.readFileSync(outPath,'utf8'));
    expect(out.source).toBe('paper');
    expect(out.totals.PnL).toBe(10);
    expect(out.perPair[0].pair).toBe('btc_jpy');
  });

  it('handles first-run with no diff file', async () => {
    const d = new Date().toISOString().slice(0,10);
  fs.writeFileSync(path.join(TMP,'stats-live.json'), JSON.stringify({ date: d, data: [{ pair: 'eth_jpy', stats: { realizedPnl: 0 } }] }));
  const readFileSyncOrig = fs.readFileSync;
  const writeFileSyncOrig = fs.writeFileSync;
  const existsSyncOrig = fs.existsSync;
  (fs.readFileSync as any) = ((p: string, enc?: any) => { const fp = path.isAbsolute(p) ? p : path.join(TMP, p); return readFileSyncOrig(fp, enc); }) as any;
  (fs.writeFileSync as any) = ((p: string, data: any, opts?: any) => { const fp = path.isAbsolute(p) ? p : path.join(TMP, p); return writeFileSyncOrig(fp, data, opts); }) as any;
  (fs.existsSync as any) = ((p: string) => { const fp = path.isAbsolute(p) ? p : path.join(TMP, p); return existsSyncOrig(fp); }) as any;
    process.argv = ['node', 'script', '--source', 'live'];
  vi.resetModules();
    await import('../../../src/tools/report-summary');
    const outPath = path.join(TMP,'report-summary-live.json');
    // Wait briefly to ensure file write completes
    await new Promise(res => setTimeout(res, 50));
    if (!fs.existsSync(outPath)) throw new Error('report-summary-live.json not found');
    const out = JSON.parse(fs.readFileSync(outPath,'utf8'));
    expect(out.totals.PnL).toBe(0);
    expect(out.perPair[0].pair).toBe('eth_jpy');
  });
});
