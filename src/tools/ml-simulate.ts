// Backward compatibility shim for CI/tests
// Standalone implementation kept here to avoid double execution/logging.
process.env.QUIET = process.env.QUIET ?? '1';
import fs from 'fs';
import path from 'path';
import { logError } from '../utils/logger';

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

function readCsvFiles(dir: string): Row[] {
  const rows: Row[] = [];
  const files = fs.existsSync(dir) ? fs.readdirSync(dir).filter(f => f.endsWith('.csv')) : [];
  for (const f of files){
    const full = path.join(dir, f);
    const txt = fs.readFileSync(full, 'utf8');
    const [header, ...lines] = txt.trim().split(/\r?\n/);
    const cols = header.split(',');
    for (const line of lines){
      if (!line.trim()) continue;
      const parts = line.split(',');
      const rec: any = {};
      cols.forEach((c, i) => rec[c] = parts[i]);
      const toNum = (v: any): number|undefined => {
        if (v == null) return undefined; const s = String(v).trim(); if (s === '') return undefined; const n = Number(s); return Number.isFinite(n) ? n : undefined;
      };
      rows.push({ ts: Number(rec.ts), pair: rec.pair, side: rec.side, rsi: toNum(rec.rsi), sma_short: toNum(rec.sma_short), sma_long: toNum(rec.sma_long), price: Number(rec.price), qty: Number(rec.qty), pnl: toNum(rec.pnl), win: toNum(rec.win) });
    }
  }
  return rows;
}

function simulate(rows: Row[], params: Record<string, any>){
  const SELL_RSI_OVERBOUGHT = Number(params.SELL_RSI_OVERBOUGHT ?? process.env.SELL_RSI_OVERBOUGHT ?? 70);
  const BUY_RSI_OVERSOLD = Number(params.BUY_RSI_OVERSOLD ?? process.env.BUY_RSI_OVERSOLD ?? 30);
  const SMA_SHORT = Number(params.SMA_SHORT ?? process.env.SMA_SHORT ?? 9);
  const SMA_LONG = Number(params.SMA_LONG ?? process.env.SMA_LONG ?? 26);
  // Pre-count trades and wins based on explicit signals (pnl presence or win flag)
  const rowsWithPnl = rows.filter(r => typeof r.pnl === 'number' && Number.isFinite(r.pnl));
  const rowsWithWinAny = rows.filter(r => (r as any).win !== undefined && (r as any).win !== null);
  const winsCountStrict = rows.filter(r => (r as any).win === 1 || (r as any).win === true || (r as any).win === '1').length;

  let trades = Math.max(rowsWithPnl.length, rowsWithWinAny.length);
  if (trades === 0 && rows.length > 0) trades = 1; // at least one synthetic trade if rows exist
  let wins = winsCountStrict;
  let pnl = 0; let holdStart = 0; let holds: number[] = [];
  const rets: number[] = []; // per-trade returns (pnl/notional)
  let equity = 0; let peak = 0; let maxDD = 0; // for Calmar
  for (const r of rows){
    const rsi = r.rsi ?? 50;
    const sShort = r.sma_short ?? 0;
    const sLong = r.sma_long ?? 0;
    // accumulate PnL and risk metrics when pnl is present
    if (typeof r.pnl === 'number' && Number.isFinite(r.pnl)){
      pnl += r.pnl;
      const notional = Math.max(1e-9, (r.price||0) * (r.qty||0));
      const ret = notional > 0 ? (r.pnl / notional) : 0;
      rets.push(ret);
      equity += ret;
      if (equity > peak) peak = equity;
      const dd = peak - equity; if (dd > maxDD) maxDD = dd;
      if (holdStart) { holds.push(r.ts - holdStart); holdStart = 0; }
    } else {
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
  const base = process.env.FEATURES_LOG_DIR ? path.resolve(process.env.FEATURES_LOG_DIR) : path.resolve(process.cwd(), 'logs');
  const dir = path.join(base, 'features', pair);
  const rows = readCsvFiles(dir).sort((a,b)=> a.ts - b.ts);
  const res = simulate(rows, params);
  console.log(JSON.stringify({ pair, ...res }));
})();
