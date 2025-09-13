// Backward compatibility shim for CI/tests
// The implementation was moved to src/tools/ml/ml-simulate.ts
// Importing it will execute the script and write JSON to stdout.
import './ml/ml-simulate';
import fs from 'fs';
import path from 'path';

function parseArgs(){
  const args = process.argv.slice(2);
  const get = (k: string, def?: string) => { const i = args.indexOf(`--${k}`); return i>=0 ? args[i+1] : def; };
  const pair = get('pair', 'btc_jpy')!;
  const paramsStr = get('params', '{}')!;
  let params: Record<string, any> = {};
  try { params = JSON.parse(paramsStr); } catch {}
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
      rows.push({ ts: Number(rec.ts), pair: rec.pair, side: rec.side, rsi: rec.rsi? Number(rec.rsi): undefined, sma_short: rec.sma_short? Number(rec.sma_short): undefined, sma_long: rec.sma_long? Number(rec.sma_long): undefined, price: Number(rec.price), qty: Number(rec.qty), pnl: rec.pnl? Number(rec.pnl): undefined, win: rec.win? Number(rec.win): undefined });
    }
  }
  return rows;
}

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
    // Simple rule: count a trade when RSI or SMA condition holds at an exit event row (pnl defined)
    if (typeof r.pnl === 'number'){
      trades++;
      pnl += r.pnl;
      if (r.win === 1) wins++;
      if (holdStart) { holds.push(r.ts - holdStart); holdStart = 0; }
    } else {
      // mark holding period start when SMA crossover aligns
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
  const rows = readCsvFiles(dir).sort((a,b)=> a.ts - b.ts);
  const res = simulate(rows, params);
  console.log(JSON.stringify({ pair, ...res }));
})();
