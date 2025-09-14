import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import '../../../src/tools/stats/stats-graph';

function today(){ return new Date().toISOString().slice(0,10); }

describe('tools/stats-graph', () => {
  const TMP = path.resolve(process.cwd(), 'tmp-test-stats-core');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
    process.env.STATS_DIR = path.join(TMP, 'logs');
    process.env.PAIRS = 'btc_jpy';
    const f = path.join(process.env.STATS_DIR!, 'pairs', 'btc_jpy', `stats-${today()}.json`);
    fs.mkdirSync(path.dirname(f), { recursive: true });
    fs.writeFileSync(f, JSON.stringify({ date: today(), trades: 1, wins: 1, realizedPnl: 5, maxDrawdown: 2 }));
  });

  it('emits svg containing legend and DD text', async () => {
  const readFileSyncOrig = fs.readFileSync;
  const writeFileSyncOrig = fs.writeFileSync;
  const existsSyncOrig = fs.existsSync;
  (fs.readFileSync as any) = ((p: string, enc?: any) => { const fp = path.isAbsolute(p) ? p : path.join(TMP, p); return readFileSyncOrig(fp, enc); }) as any;
  (fs.writeFileSync as any) = ((p: string, data: any, opts?: any) => { const fp = path.isAbsolute(p) ? p : path.join(TMP, p); return writeFileSyncOrig(fp, data, opts); }) as any;
  (fs.existsSync as any) = ((p: string) => { const fp = path.isAbsolute(p) ? p : path.join(TMP, p); return existsSyncOrig(fp); }) as any;
    const argvOrig = process.argv;
    process.argv = ['node','script','--out', path.join(TMP,'stats.json'),'--svg',path.join(TMP,'stats.svg')];
    try {
      const svgPath = path.join(TMP,'stats.svg');
      for (let i=0;i<20 && !fs.existsSync(svgPath);i++){ await new Promise(r=>setTimeout(r,25)); }
      const svg = fs.readFileSync(svgPath,'utf8');
      expect(svg).toContain('最大DD線');
      expect(svg).toContain('DD=');
    } finally {
      process.argv = argvOrig;
    }
  });
});
