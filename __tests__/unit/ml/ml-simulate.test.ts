import { describe, it, expect, beforeEach, test } from 'vitest';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { sleep } from '../../../ztb/utils/toolkit';

// Use a unique temp dir to avoid cross-file race with ml-search.cache.test
const TMP = path.resolve(process.cwd(), 'tmp-test-ml-simulate');

function todayStr(){ return new Date().toISOString().slice(0,10); }

describe('ml-simulate', ()=>{
  const pair = 'btc_jpy';
  const date = todayStr();
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(path.join(TMP,'features',pair), { recursive: true });
    process.env.FEATURES_LOG_DIR = TMP;
  });
  async function waitForFileNonEmpty(p: string, max = 10, intervalMs = 80) {
    for (let i = 0; i < max; i++) {
      try {
        if (fs.existsSync(p)) {
          const st = fs.statSync(p);
          if (st.size > 0) return;
        }
      } catch {}
      await sleep(intervalMs);
    }
  }
  // Under coverage + Windows FS occasionally needs longer; extend to 15s
  test.sequential('computes winRate and pnl', async ()=>{
    const jsonlPath = path.join(TMP,'features',pair,`features-${date}.jsonl`);
    const now = Date.now();
  const jsonlLines = [
      { ts: now-2000, pair, side:'ask', rsi:70, sma_short:9, sma_long:26, price:100, qty:0.001 },
      { ts: now-1000, pair, side:'ask', rsi:75, sma_short:9, sma_long:26, price:105, qty:0.001, pnl:5, win:1 }
  ].map(o=>JSON.stringify(o)).join('\n');
  fs.writeFileSync(jsonlPath, jsonlLines);
    await waitForFileNonEmpty(jsonlPath);
    const mlPath = path.resolve(process.cwd(), 'src', 'tools', 'ml-simulate.ts').replace(/\\/g,'/');
  const cmd = `node -e "require('ts-node').register(); require('${mlPath}');" -- --pair ${pair}`;
    process.env.QUIET = '1';
    const out = execSync(cmd, { encoding: 'utf8', env: { ...process.env } as any });
    const stdoutLines = out.trim().split(/\r?\n/).filter(Boolean);
    const jsonLine = [...stdoutLines].reverse().find(l => {
      const t = l.trim();
      return t.startsWith('{') && t.endsWith('}');
    }) || stdoutLines[stdoutLines.length-1];
    const res = JSON.parse(jsonLine);
    try {
      expect(res.trades).toBe(1);
      expect(res.winRate).toBe(1);
      expect(res.pnl).toBe(5);
    } catch (e) {
      // Diagnostics on failure
      console.error(`[WARN][DIAG] rowsCount=${res.rowsCount}, scanned=${JSON.stringify(res.scanned)}, filesByDir=${JSON.stringify(res.filesByDir)}`);
      throw e;
    }
  }, 15000);

  test.sequential('counts trade even when only win flag is present', async ()=>{
  const jsonlPath = path.join(TMP,'features',pair,`features-${date}.jsonl`);
  const now = Date.now();
  const jsonlLines2 = [{ ts: now-1000, pair, side:'ask', rsi:70, sma_short:9, sma_long:26, price:100, qty:0.001, win:1 }].map(o=>JSON.stringify(o)).join('\n');
  fs.writeFileSync(jsonlPath, jsonlLines2);
    await waitForFileNonEmpty(jsonlPath);
    // Windows の遅延 flush レース緩和
    await sleep(80);
    const mlPath = path.resolve(process.cwd(), 'src', 'tools', 'ml-simulate.ts').replace(/\\/g,'/');
  const cmd = `node -e "require('ts-node').register(); require('${mlPath}');" -- --pair ${pair}`;
    process.env.QUIET = '1';
    const out = execSync(cmd, { encoding: 'utf8', env: { ...process.env } as any });
  const outLines = out.trim().split(/\r?\n/).filter(Boolean);
  const jsonLine = [...outLines].reverse().find(l => { const t = l.trim(); return t.startsWith('{') && t.endsWith('}'); }) || outLines[outLines.length-1];
  const res = JSON.parse(jsonLine);
    try {
      expect(res.trades).toBe(1);
      expect(res.winRate).toBe(1);
    } catch (e) {
      console.error(`[WARN][DIAG] rowsCount=${res.rowsCount}, scanned=${JSON.stringify(res.scanned)}, filesByDir=${JSON.stringify(res.filesByDir)}`);
      throw e;
    }
  }, 15000);

  test.sequential('returns zeros when no rows', async ()=>{
    const dir = path.join(TMP,'features',pair);
    // ensure empty directory exists
    fs.mkdirSync(dir, { recursive: true });
    await sleep(50);
    const mlPath = path.resolve(process.cwd(), 'src', 'tools', 'ml-simulate.ts').replace(/\\/g,'/');
  const cmd = `node -e "require('ts-node').register(); require('${mlPath}');" -- --pair ${pair}`;
    process.env.QUIET = '1';
    const out = execSync(cmd, { encoding: 'utf8', env: { ...process.env } as any });
    const stdoutLines = out.trim().split(/\r?\n/).filter(Boolean);
    const jsonLine = [...stdoutLines].reverse().find(l => { const t = l.trim(); return t.startsWith('{') && t.endsWith('}'); }) || stdoutLines[stdoutLines.length-1];
    const res = JSON.parse(jsonLine);
    try {
      expect(res.trades).toBe(0);
      expect(res.winRate).toBe(0);
      expect(res.pnl).toBe(0);
    } catch (e) {
      console.error(`[WARN][DIAG] rowsCount=${res.rowsCount}, scanned=${JSON.stringify(res.scanned)}, filesByDir=${JSON.stringify(res.filesByDir)}`);
      throw e;
    }
  }, 15000);
});
