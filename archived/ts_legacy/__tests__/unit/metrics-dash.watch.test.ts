import { describe, it, expect, vi } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { runDash } from '../../ztb/tools/metrics-dash';

describe('metrics-dash --watch', () => {
  it('refreshes at least twice and stops on SIGINT', async () => {
    const tmpDir = fs.mkdtempSync(path.join(process.cwd(), 'tmp-test-dash-'));
    const logFile = path.join(tmpDir, 'test.log');
    const lines: string[] = [];
    const write = () => fs.writeFileSync(logFile, lines.join('\n'), 'utf8');
    // seed with JSON log entries
    const json = (entry: any) => JSON.stringify(entry);
    const mkRate = (avgWaitMs: number) => json({ ts: new Date().toISOString(), level: 'INFO', category: 'RATE', message: 'metrics', data: [{ window: 5, avgWaitMs, rejectRate: 0, byCategory: { PUBLIC: 5, PRIVATE: 0, EXEC: 0 }, details: { PUBLIC: { count: 5, acquired: 5, rejected: 0, avgWaitMs, rejectRate: 0, capacity: 10, refillPerSec: 5 } } }] });
    const mkCache = (hitRate: number) => json({ ts: new Date().toISOString(), level: 'INFO', category: 'CACHE', message: 'metrics', data: [{ ticker: { hits: 5, misses: 0, stale: 0, hitRate } }] });
    lines.push(mkRate(10));
    lines.push(mkCache(0.8));
    write();

    let loops = 0;
    const spyLog = vi.spyOn(console, 'log').mockImplementation(() => {});
    const spyErr = vi.spyOn(console, 'error').mockImplementation(() => {});
    const spyClear = vi.spyOn(console, 'clear').mockImplementation(() => {});

    const abort = new AbortController();
    const p = runDash({ file: logFile, lines: 200, watch: true, watchMs: 200, abortSignal: abort.signal });

    await new Promise(r => setTimeout(r, 250));
    // update file with new metrics to ensure a second loop sees it
    lines.push(mkRate(20));
    lines.push(mkCache(0.9));
    write();

    await new Promise(r => setTimeout(r, 300));
    abort.abort();
    await p;

    // At least two refresh cycles should have happened
    const combined = spyLog.mock.calls.map(c => String(c[0])).join('\n');
    expect(combined).toContain('RATE/METRICS');
    expect(combined).toContain('CACHE/METRICS');

    spyLog.mockRestore();
    spyErr.mockRestore();
    spyClear.mockRestore();
    try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}
  });
});
