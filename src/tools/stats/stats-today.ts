import dotenv from 'dotenv';
dotenv.config();
import { loadDaily } from '../../utils/daily-stats';
import { loadPairs } from '../../utils/config';

function today(){ return new Date().toISOString().slice(0,10); }

const args = process.argv.slice(2);
const flagDiff = args.includes('--diff');
const pairArgIdx = args.indexOf('--pair');
const pairArg = pairArgIdx>=0 ? args[pairArgIdx+1] : undefined;

(async ()=>{
  const d = today();
  const pairs = pairArg ? [pairArg] : loadPairs();
  if (!pairs.length){
    const v = loadDaily(d);
    if (flagDiff){
      const s = (global as any).__stats_snapshot__ || {};
      const diff = Object.keys(v).reduce((acc:any, k)=>{ const prev = (s as any)[k] || 0; const cur = (v as any)[k] || 0; if (typeof cur==='number') acc[k] = cur - prev; return acc; }, {});
      console.log(JSON.stringify({ values: v, diff }));
      ;(global as any).__stats_snapshot__ = v;
    } else {
      console.log(JSON.stringify(v));
    }
    return;
  }
  const values: Record<string, any> = {};
  const diffs: any[] = [];
  for (const p of pairs){
    const v = loadDaily(d, p);
    values[p] = v;
  }
  if (flagDiff){
    const s = (global as any).__stats_snapshot_pairs__ || {};
    for (const p of pairs){
      const v = values[p];
      const prev = (s as any)[p] || {};
      const diff = Object.keys(v).reduce((acc:any, k)=>{ const prevV = (prev as any)[k] || 0; const cur = (v as any)[k] || 0; if (typeof cur==='number') acc[k] = cur - prevV; return acc; }, {});
      diffs.push({ pair: p, diff });
    }
    console.log(JSON.stringify({ pairs: values, pairsDiff: diffs }));
    ;(global as any).__stats_snapshot_pairs__ = values;
  } else {
    console.log(JSON.stringify({ pairs: values }));
  }
})();
