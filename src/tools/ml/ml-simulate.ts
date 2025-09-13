import path from 'path';
import { readFeatureCsvRows } from '../../utils/toolkit';

function parseArgs(){
  const args = process.argv.slice(2);
  const get = (k: string, def?: string) => { 
    const i = args.indexOf(`--${k}`); 
    return i>=0 ? args[i+1] : def; 
  };
  const pair = get('pair', 'btc_jpy')!;
  const paramsStr = get('params', '{}')!;
  let params: Record<string, any> = {};
  try { params = JSON.parse(paramsStr); } catch {}
  return { pair, params };
}

type Row = ReturnType<typeof readFeatureCsvRows>[number];

function simulate(rows: Row[], params: Record<string, any>){
  const SELL_RSI_OVERBOUGHT = Number(params.SELL_RSI_OVERBOUGHT ?? process.env.SELL_RSI_OVERBOUGHT ?? 70);
  const BUY_RSI_OVERSOLD = Number(params.BUY_RSI_OVERSOLD ?? process.env.BUY_RSI_OVERSOLD ?? 30);
  const SMA_SHORT = Number(params.SMA_SHORT ?? process.env.SMA_SHORT ?? 9);
  const SMA_LONG = Number(params.SMA_LONG ?? process.env.SMA_LONG ?? 26);

  let wins = 0, trades = 0, pnl = 0; let holdStart = 0; let holds: number[] = [];
  for (const r of rows){
    const rsi = r.rsi ?? 50;
    const sShort = r.sma_short ?? 0;
    const sLong = r.sma_long ?? 0;
    if (typeof r.pnl === 'number' && Number.isFinite(r.pnl)){
      trades++;
      pnl += r.pnl;
      if (r.win === 1) wins++;
      if (holdStart) { holds.push(r.ts - holdStart); holdStart = 0; }
    } else if ((r as any).win !== undefined && (r as any).win !== null){ // fallback: win-only rows
      trades++;
      const w = (r as any).win;
      if (w === 1 || w === true || w === '1') wins++;
      if (holdStart) { holds.push(r.ts - holdStart); holdStart = 0; }
    } else {
      if (!holdStart && sShort && sLong && ((r.side==='ask' && sShort < sLong && rsi >= SELL_RSI_OVERBOUGHT) || (r.side==='bid' && sShort > sLong && rsi <= BUY_RSI_OVERSOLD))) {
        holdStart = r.ts;
      }
    }
  }
  const winRate = trades ? wins / trades : 0;
  const avgHoldSec = holds.length ? (holds.reduce((a,b)=>a+b,0) / holds.length) / 1000 : 0;
  return { winRate, pnl, trades, avgHoldSec };
}

(async ()=>{
  const { pair, params } = parseArgs();
  const base = process.env.FEATURES_LOG_DIR ? path.resolve(process.env.FEATURES_LOG_DIR) : path.resolve(process.cwd(), 'logs');
  const dir = path.join(base, 'features', pair);
  const rows = readFeatureCsvRows(dir).sort((a,b)=> a.ts - b.ts);
  const res = simulate(rows, params);
  const payload = { pair, ...res };
  if (process.env.QUIET !== '1') {
    // optional extra logs for local runs
    console.error('[ML] rows', rows.length);
  }
  console.log(JSON.stringify(payload));
})();
