import path from 'path';
import { readFeatureCsvRows, sleep } from '../../utils/toolkit';

function parseArgs(){
  const args = process.argv.slice(2);
  const get = (k: string, def?: string) => { 
    const i = args.indexOf(`--${k}`); 
    return i>=0 ? args[i+1] : def; 
  };
  const pair = get('pair', 'btc_jpy')!;
  const paramsStr = get('params', '{}')!;
  let params: Record<string, any> = {};
  try { params = JSON.parse(paramsStr); } catch {
    // tolerant parsing: replace single quotes and add quotes around unquoted keys
    try {
      let s = String(paramsStr);
      // normalize quotes
      s = s.replace(/'/g, '"');
      // add quotes to unquoted keys: {a:1} -> {"a":1}
      s = s.replace(/([\{,]\s*)([A-Za-z0-9_]+)\s*:/g, '$1"$2":');
      params = JSON.parse(s);
    } catch {
      // silent fallback: params remain empty
      params = {};
    }
  }
  return { pair, params };
}

type Row = ReturnType<typeof readFeatureCsvRows>[number];

function simulate(rows: Row[], params: Record<string, any>){
  const SELL_RSI_OVERBOUGHT = Number(params.SELL_RSI_OVERBOUGHT ?? process.env.SELL_RSI_OVERBOUGHT ?? 70);
  const BUY_RSI_OVERSOLD = Number(params.BUY_RSI_OVERSOLD ?? process.env.BUY_RSI_OVERSOLD ?? 30);
  const SMA_SHORT = Number(params.SMA_SHORT ?? process.env.SMA_SHORT ?? 9);
  const SMA_LONG = Number(params.SMA_LONG ?? process.env.SMA_LONG ?? 26);

  const isWin = (w: any) => w === 1 || w === true || w === '1';
  let wins = 0, trades = 0, pnl = 0; let holdStart = 0; let holds: number[] = [];
  for (const r of rows){
    const rsi = r.rsi ?? 50;
    const sShort = r.sma_short ?? 0;
    const sLong = r.sma_long ?? 0;
    const hasPnl = typeof r.pnl === 'number' && Number.isFinite(r.pnl);
    const hasWin = (r as any).win !== undefined && (r as any).win !== null;

    if (hasPnl || hasWin) {
      trades++;
      if (isWin((r as any).win)) wins++;
      if (hasPnl) pnl += r.pnl as number;
      if (holdStart) { holds.push(r.ts - holdStart); holdStart = 0; }
      continue;
    }

    if (!holdStart && sShort && sLong && ((r.side==='ask' && sShort < sLong && rsi >= SELL_RSI_OVERBOUGHT) || (r.side==='bid' && sShort > sLong && rsi <= BUY_RSI_OVERSOLD))) {
      holdStart = r.ts;
    }
  }
  const winRate = trades ? wins / trades : 0;
  const avgHoldSec = holds.length ? (holds.reduce((a,b)=>a+b,0) / holds.length) / 1000 : 0;
  return { winRate, pnl, trades, avgHoldSec };
}

(async ()=>{
  const { pair, params } = parseArgs();
  const fs = require('fs') as typeof import('fs');
  const base = process.env.FEATURES_LOG_DIR ? path.resolve(process.env.FEATURES_LOG_DIR) : path.resolve(process.cwd(), 'logs');
  const roots = [base];
  const testFallback = path.resolve(process.cwd(), 'tmp-test-ml-simulate');
  if (fs.existsSync(testFallback)) roots.push(testFallback);
  const dirs = Array.from(new Set(roots.flatMap(b => [
    path.join(b, 'features', pair),
    path.join(b, 'features', 'live', pair),
    path.join(b, 'features', 'paper', pair),
  ])));
  const filesByDir: Record<string,string[]> = {};
  try {
    const fs2 = require('fs') as typeof import('fs');
    const path2 = require('path') as typeof import('path');
    for (const d of dirs) {
      try {
        if (fs2.existsSync(d)) filesByDir[d] = fs2.readdirSync(d).filter((f: string)=> f.startsWith('features-'));
        else filesByDir[d] = [];
      } catch { filesByDir[d] = []; }
    }
  } catch {}
  let rows = dirs.flatMap(d => readFeatureCsvRows(d)).sort((a,b)=> a.ts - b.ts);
  if (rows.length === 0) {
  const retries = process.platform === 'win32' ? 10 : 2;
  const delay = process.platform === 'win32' ? 80 : 25;
    for (let i=0;i<retries && rows.length===0;i++) {
      await sleep(delay);
      rows = dirs.flatMap(d => readFeatureCsvRows(d)).sort((a,b)=> a.ts - b.ts);
    }
    if (rows.length === 0) {
      const fs2 = require('fs') as typeof import('fs');
      const path2 = require('path') as typeof import('path');
      const today = new Date().toISOString().slice(0,10);
      const toNum = (v: any): number|undefined => { const n = Number(v); return Number.isFinite(n) ? n : undefined; };
      for (const d of dirs) {
        const f = path2.join(d, `features-${today}.jsonl`);
        try {
          if (fs2.existsSync(f)) {
            const txt = fs2.readFileSync(f, 'utf8');
            const lines = String(txt||'').split(/\r?\n/).filter(Boolean);
            for (const line of lines) {
              try {
                const o = JSON.parse(line);
                rows.push({
                  ts: Number(o.ts), pair: String(o.pair), side: String(o.side),
                  rsi: toNum(o.rsi), sma_short: toNum(o.sma_short), sma_long: toNum(o.sma_long),
                  price: Number(o.price), qty: Number(o.qty), pnl: toNum(o.pnl), win: toNum(o.win)
                } as any);
              } catch {}
            }
          }
        } catch {}
      }
      rows.sort((a,b)=> a.ts - b.ts);
    }
  }
  const res = simulate(rows, params);
  const payload = {
    pair,
    rowsCount: rows.length,
  scanned: dirs,
  filesByDir,
    ...res,
  };
  if (process.env.QUIET !== '1') {
    // optional extra logs for local runs
    console.error('[ML] rows', rows.length);
  }
  console.log(JSON.stringify(payload));
})();
