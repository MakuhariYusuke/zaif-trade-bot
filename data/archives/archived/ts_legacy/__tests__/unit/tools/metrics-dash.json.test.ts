import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { runOnceCollect } from '../../../ztb/tools/metrics-dash';

const ROOT = process.cwd();
const TMP = path.join(ROOT, 'tmp-test-dash-qXjivF');

function writeLog(lines: any[]){
  const p = path.join(TMP, 'test.log');
  fs.mkdirSync(TMP, { recursive: true });
  fs.writeFileSync(p, lines.map(l=>JSON.stringify(l)).join('\n')+'\n', 'utf8');
  return p;
}

function runDashJson(file: string){
  return runOnceCollect(file, 4000) as any;
}

describe('metrics-dash --json', ()=>{
  beforeEach(()=>{ if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true }); });
  it('emits keys rate/cache/events/tradePhase', ()=>{
    const file = writeLog([
      { ts: new Date().toISOString(), level: 'INFO', category: 'RATE', message: 'metrics', data: [{ window: 60000, avgWaitMs: 0, rejectRate: 0, byCategory: { PUBLIC: 0, PRIVATE: 0, EXEC: 0 }, details: { PUBLIC: { count: 1, acquired: 1, rejected: 0, avgWaitMs: 0, rejectRate: 0, capacity: 10, refillPerSec: 5 } } }] },
      { ts: new Date().toISOString(), level: 'INFO', category: 'CACHE', message: 'metrics', data: [{ default: { hits: 1, misses: 0, stale: 0, hitRate: 1 } }] },
      { ts: new Date().toISOString(), level: 'INFO', category: 'EVENT', message: 'metrics', data: [{ windowMs: 60000, types: {} }] },
    ]);
    const res = runDashJson(file);
    expect(res).toBeTruthy();
    expect(res).toHaveProperty('rate');
    expect(res).toHaveProperty('cache');
    expect(res).toHaveProperty('events');
    expect(res).toHaveProperty('tradePhase');
  });
});
