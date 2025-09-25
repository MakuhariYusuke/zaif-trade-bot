import { readFeatureCsvRows } from '../../utils/toolkit';
import path from 'path';

export interface McResult { mean: number; std: number; var5: number; es5: number; maxddMean: number; runs: number; len: number }

function parseArgs(){
  const args = process.argv.slice(2);
  const get = (k: string, def?: string) => { const i = args.indexOf(`--${k}`); return i>=0 ? args[i+1] : def; };
  const pair = get('pair', process.env.PAIR || 'btc_jpy')!;
  const runs = Number(get('runs', process.env.MC_RUNS || '500'));
  const len = get('len');
  return { pair, runs, len: len? Number(len): undefined };
}

export function monteCarloFromReturns(rets: number[], runs: number, len?: number): McResult {
  if (!rets.length) return { mean: 0, std: 0, var5: 0, es5: 0, maxddMean: 0, runs: 0, len: 0 };
  const L = len || rets.length;
  const finals: number[] = [];
  const dds: number[] = [];
  for (let r=0;r<runs;r++){
    let eq = 0; let peak = 0; let maxdd = 0;
    for (let i=0;i<L;i++){
      const x = rets[Math.floor(Math.random() * rets.length)] || 0;
      eq += x;
      if (eq > peak) peak = eq;
      const dd = peak - eq; if (dd > maxdd) maxdd = dd;
    }
    finals.push(eq);
    dds.push(maxdd);
  }
  const mean = finals.reduce((a,b)=>a+b,0)/finals.length;
  const variance = finals.reduce((a,b)=> a + Math.pow(b-mean,2), 0) / finals.length;
  const std = Math.sqrt(variance);
  const sorted = finals.slice().sort((a,b)=> a-b);
  const k = Math.max(0, Math.floor(0.05 * (sorted.length-1)));
  const var5 = sorted[k];
  const es5 = sorted.slice(0, k+1).reduce((a,b)=>a+b,0) / (k+1);
  const maxddMean = dds.reduce((a,b)=>a+b,0)/dds.length;
  return { mean, std, var5, es5, maxddMean, runs: finals.length, len: L };
}

if (require.main === module){
  const { pair, runs, len } = parseArgs();
  const base = process.env.FEATURES_LOG_DIR ? path.resolve(process.env.FEATURES_LOG_DIR) : path.resolve(process.cwd(), 'logs');
  const dir = path.join(base, 'features', pair);
  const rows = readFeatureCsvRows(dir).sort((a,b)=> a.ts - b.ts);
  const rets: number[] = [];
  for (const r of rows){
    if (typeof r.pnl === 'number' && Number.isFinite(r.pnl)){
      const notional = Math.max(1e-9, (r.price||0) * (r.qty||0));
      rets.push(notional>0 ? (r.pnl / notional) : 0);
    }
  }
  const res = monteCarloFromReturns(rets, runs, len);
  console.log(JSON.stringify({ pair, ...res }));
}
