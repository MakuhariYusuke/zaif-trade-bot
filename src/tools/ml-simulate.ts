// Backward compatibility shim for CI/tests
// Standalone implementation kept here to avoid double execution/logging.
process.env.QUIET = process.env.QUIET ?? '1';
import path from 'path';
import { logError } from '../utils/logger';
import { readFeatureCsvRows, sleep } from '../utils/toolkit';

function parseArgs(){
  const args = process.argv.slice(2);
  const get = (k: string, def?: string) => { const i = args.indexOf(`--${k}`); return i>=0 ? args[i+1] : def; };
  const pair = get('pair', 'btc_jpy')!;
  const paramsStr = get('params', '{}')!;
  let params: Record<string, any> = {};
  try { 
    params = JSON.parse(paramsStr); 
  } catch (e) {
    logError(`[ERROR] Failed to parse params: ${paramsStr}`, e);
    params = {};
  }
  return { pair, params };
}

interface Row { ts: number; pair: string; side: string; rsi?: number; sma_short?: number; sma_long?: number; price: number; qty: number; pnl?: number; win?: number; }

function simulate(rows: Row[], params: Record<string, any>){
  const SELL_RSI_OVERBOUGHT = Number(params.SELL_RSI_OVERBOUGHT ?? process.env.SELL_RSI_OVERBOUGHT ?? 70);
  const BUY_RSI_OVERSOLD = Number(params.BUY_RSI_OVERSOLD ?? process.env.BUY_RSI_OVERSOLD ?? 30);
  const SMA_SHORT = Number(params.SMA_SHORT ?? process.env.SMA_SHORT ?? 9);
  const SMA_LONG = Number(params.SMA_LONG ?? process.env.SMA_LONG ?? 26);
  const isWin = (w: any) => w === 1 || w === true || w === '1';
  let trades = 0, wins = 0, pnl = 0; let holdStart = 0; let holds: number[] = [];
  const rets: number[] = []; // per-trade returns (pnl/notional)
  let equity = 0; let peak = 0; let maxDD = 0; // for Calmar
  for (const r of rows){
    const rsi = r.rsi ?? 50;
    const sShort = r.sma_short ?? 0;
    const sLong = r.sma_long ?? 0;
    const hasPnl = typeof r.pnl === 'number' && Number.isFinite(r.pnl);
    const hasWin = (r as any).win !== undefined && (r as any).win !== null;

    if (hasPnl || hasWin) {
      trades++;
      if (isWin((r as any).win)) wins++;
      if (hasPnl) {
        pnl += r.pnl as number;
        const notional = Math.max(1e-9, (r.price||0) * (r.qty||0));
        const ret = notional > 0 ? ((r.pnl as number) / notional) : 0;
        rets.push(ret);
        equity += ret;
        if (equity > peak) peak = equity;
        const dd = peak - equity; if (dd > maxDD) maxDD = dd;
      }
      if (holdStart) { holds.push(r.ts - holdStart); holdStart = 0; }
      continue;
    }

    // mark holding period start when SMA crossover aligns
    if (
      !holdStart &&
      sShort !== 0 &&
      sLong !== 0 &&
      ((r.side === 'ask' && sShort < sLong && rsi >= SELL_RSI_OVERBOUGHT) ||
       (r.side === 'bid' && sShort > sLong && rsi <= BUY_RSI_OVERSOLD))
    ) {
      holdStart = r.ts;
    }
  }
  const winRate = trades ? wins / trades : 0;
  const avgHoldSec = holds.length ? (holds.reduce((a,b)=>a+b,0) / holds.length) / 1000 : 0;
  // Risk-adjusted metrics (per-trade approximation)
  const mean = rets.length ? rets.reduce((a,b)=>a+b,0) / rets.length : 0;
  const variance = rets.length ? rets.reduce((a,b)=> a + Math.pow(b-mean,2), 0) / rets.length : 0;
  const std = Math.sqrt(variance);
  const downside = rets.filter(r => r < 0);
  const dvar = downside.length ? downside.reduce((a,b)=> a + Math.pow(b-0,2), 0) / downside.length : 0;
  const ddev = Math.sqrt(dvar);
  const sharpe = std > 0 ? (mean / std) * Math.sqrt(Math.max(1, rets.length)) : 0;
  const sortino = ddev > 0 ? (mean / ddev) * Math.sqrt(Math.max(1, rets.length)) : 0;
  const calmar = maxDD > 0 ? (rets.reduce((a,b)=>a+b,0) / maxDD) : 0;
  return { winRate, pnl, trades, avgHoldSec, sharpe, sortino, calmar };
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
  // When FEATURES_LOG_DIR is explicitly set (tests), try direct file path first
  const today = new Date().toISOString().slice(0,10);
  const directCandidates: string[] = [];
  if (process.env.FEATURES_LOG_DIR) {
    const p1 = path.join(base, 'features', pair, `features-${today}.jsonl`);
    directCandidates.push(p1);
    if (fs.existsSync(testFallback)) {
      const p2 = path.join(testFallback, 'features', pair, `features-${today}.jsonl`);
      if (p2 !== p1) directCandidates.push(p2);
    }
  }
  const toNum = (v: any): number|undefined => { const n = Number(v); return Number.isFinite(n) ? n : undefined; };
  const uniqueCandidates = Array.from(new Set(directCandidates));
  let rows = uniqueCandidates.flatMap(p => {
    try {
      if (!fs.existsSync(p)) return [] as Row[];
      const txt = fs.readFileSync(p, 'utf8');
      return String(txt||'').split(/\r?\n/).filter(Boolean).map(line => {
        try {
          const o = JSON.parse(line);
          return {
            ts: Number(o.ts), pair: String(o.pair), side: String(o.side),
            rsi: toNum(o.rsi), sma_short: toNum(o.sma_short), sma_long: toNum(o.sma_long),
            price: Number(o.price), qty: Number(o.qty), pnl: toNum(o.pnl), win: toNum(o.win)
          } as Row;
        } catch { return null as any; }
      }).filter(Boolean);
    } catch { return [] as Row[]; }
  });
  // Windows FS で直後読みが空になるレース緩和: リトライ最大5回
  if (rows.length === 0 && process.platform === 'win32' && uniqueCandidates.length) {
    for (let i=0;i<5 && rows.length===0;i++) {
      try { await sleep(60); } catch {}
      rows = uniqueCandidates.flatMap(p => {
        try {
          if (!fs.existsSync(p)) return [] as Row[];
          const txt = fs.readFileSync(p, 'utf8');
          return String(txt||'').split(/\r?\n/).filter(Boolean).map(line => {
            try {
              const o = JSON.parse(line);
              return {
                ts: Number(o.ts), pair: String(o.pair), side: String(o.side),
                rsi: toNum(o.rsi), sma_short: toNum(o.sma_short), sma_long: toNum(o.sma_long),
                price: Number(o.price), qty: Number(o.qty), pnl: toNum(o.pnl), win: toNum(o.win)
              } as Row;
            } catch { return null as any; }
          }).filter(Boolean);
        } catch { return [] as Row[]; }
      });
    }
  }
  // Retry small delay to avoid race immediately after writer flush in tests/CI (Windows FS can lag)
  if (rows.length === 0) rows = dirs.flatMap(d => readFeatureCsvRows(d)).sort((a: Row, b: Row)=> a.ts - b.ts);
  if (rows.length === 0) {
    const retries = process.platform === 'win32' ? 10 : 2;
    const delay = process.platform === 'win32' ? 80 : 25;
    for (let i=0;i<retries && rows.length===0;i++) {
      await sleep(delay);
      rows = dirs.flatMap(d => readFeatureCsvRows(d)).sort((a: Row, b: Row)=> a.ts - b.ts);
    }
    // Final fallback: directly read today's expected JSONL if present
    if (rows.length === 0) {
      const today = new Date().toISOString().slice(0,10);
      const toNum = (v: any): number|undefined => { const n = Number(v); return Number.isFinite(n) ? n : undefined; };
      for (const d of dirs) {
        const f = path.join(d, `features-${today}.jsonl`);
        try {
          if (fs.existsSync(f)) {
            const txt = fs.readFileSync(f, 'utf8');
            const lines = String(txt||'').split(/\r?\n/).filter(Boolean);
            for (const line of lines) {
              try {
                const o = JSON.parse(line);
                rows.push({
                  ts: Number(o.ts), pair: String(o.pair), side: String(o.side),
                  rsi: toNum(o.rsi), sma_short: toNum(o.sma_short), sma_long: toNum(o.sma_long),
                  price: Number(o.price), qty: Number(o.qty), pnl: toNum(o.pnl), win: toNum(o.win)
                } as Row);
              } catch {}
            }
          }
        } catch {}
      }
      rows.sort((a: Row, b: Row)=> a.ts - b.ts);
    }
    // Last resort: pick most recent features-*.jsonl in any scanned dir
    if (rows.length === 0) {
      try {
        const cand: Array<{file:string, mtime:number}> = [];
        for (const d of dirs) {
          try {
            if (!fs.existsSync(d)) continue;
            const files = fs.readdirSync(d).filter((f: string)=> f.startsWith('features-') && f.endsWith('.jsonl'));
            for (const f of files) {
              const p = path.join(d, f);
              try { const st = fs.statSync(p); cand.push({ file: p, mtime: st.mtimeMs }); } catch {}
            }
          } catch {}
        }
        if (cand.length) {
          cand.sort((a,b)=> b.mtime - a.mtime);
          const p = cand[0].file;
          try {
            const txt = fs.readFileSync(p, 'utf8');
            rows = String(txt||'').split(/\r?\n/).filter(Boolean).map(line => {
              try {
                const o = JSON.parse(line);
                return {
                  ts: Number(o.ts), pair: String(o.pair), side: String(o.side),
                  rsi: toNum(o.rsi), sma_short: toNum(o.sma_short), sma_long: toNum(o.sma_long),
                  price: Number(o.price), qty: Number(o.qty), pnl: toNum(o.pnl), win: toNum(o.win)
                } as Row;
              } catch { return null as any; }
            }).filter(Boolean).sort((a: Row, b: Row)=> a.ts - b.ts);
          } catch {}
        }
      } catch {}
    }
  }
  const res = simulate(rows, params);
  // Include minimal diagnostics for stability debugging; tests ignore extra fields
  const filesByDir: Record<string,string[]> = {};
  try {
    for (const d of dirs) {
      try { filesByDir[d] = fs.existsSync(d) ? fs.readdirSync(d).filter((f: string)=> f.startsWith('features-')) : []; }
      catch { filesByDir[d] = []; }
    }
  } catch {}
  console.log(JSON.stringify({ pair, rowsCount: rows.length, scanned: dirs, filesByDir, ...res }));
})();
