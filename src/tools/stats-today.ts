import dotenv from 'dotenv';
dotenv.config();
import path from 'path';
import fs from 'fs';
import { loadDaily } from '../utils/daily-stats';
import { loadPairs } from '../utils/config';

function todayStr(){ return new Date().toISOString().slice(0,10); }
const KEYS = ['buyEntries','sellEntries','rsiExits','trailExitTotal','trailStops','realizedPnl','wins','trades','streakWin','streakLoss'] as const;
type Keys = typeof KEYS[number];

function pick(obj: Record<string, any>){
  const out: Record<string, any> = {};
  for (const k of KEYS) out[k] = obj[k] ?? 0;
  return out;
}

(async ()=>{
  const args = process.argv.slice(2);
  const wantDiff = args.includes('--diff');
  const date = todayStr();
  const pairs = loadPairs();
  const stats = loadDaily(date) as any;
  const logsDir = process.env.STATS_DIR || path.resolve(process.cwd(), 'logs');
  const snapPath = path.join(logsDir, `.stats-snapshot-${date}.json`);

  let diff: Record<string, number> | null = null;
  if (wantDiff && fs.existsSync(snapPath)){
    const before = JSON.parse(fs.readFileSync(snapPath,'utf8')) as Record<string, any>;
    diff = {};
    for (const k of KEYS){ (diff as any)[k] = (stats[k]||0) - (before[k]||0); }
  }

  // per-pair values and diffs
  const pairsData = pairs.map(pair => {
    const s = loadDaily(date, pair) as any;
    return { pair, values: pick(s) };
  });
  let pairsDiff: Array<{pair:string; diff: Record<string, number>}> | null = null;
  if (wantDiff && fs.existsSync(snapPath)){
    try {
      const before = JSON.parse(fs.readFileSync(snapPath,'utf8')) as any;
      const beforePairs: any[] = before.pairs || [];
      pairsDiff = pairs.map(pair => {
        const prev = beforePairs.find(p=>p.pair===pair)?.values || {};
        const now = loadDaily(date, pair) as any;
        const d: Record<string, number> = {};
        for (const k of KEYS){ d[k] = (now[k]||0) - (prev[k]||0); }
        return { pair, diff: d };
      });
    } catch {}
  }

  const result = { date, values: pick(stats), diff, pairs: pairsData, pairsDiff } as const;
  console.log(JSON.stringify(result));

  // Update snapshot to current (after printing)
  try{ if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true }); fs.writeFileSync(snapPath, JSON.stringify({ ...stats, pairs: pairsData }, null, 2)); } catch {}
})();
