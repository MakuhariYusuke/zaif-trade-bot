import dotenv from 'dotenv';
dotenv.config();
import fs from 'fs';
import path from 'path';

function getArg(name: string, def?: string){
  const args = process.argv.slice(2);
  const i = args.indexOf(`--${name}`);
  return i>=0 ? args[i+1] : def;
}

(async ()=>{
  const source = getArg('source', 'live')!;
  // try scenario-specific files first
  const statsDiff = (()=>{
    const fname = `stats-diff-${source}.json`;
    if (fs.existsSync(fname)) return JSON.parse(fs.readFileSync(fname,'utf8'));
    if (fs.existsSync('stats-diff.json')) return JSON.parse(fs.readFileSync('stats-diff.json','utf8'));
    return {};
  })();
  const statsJson = (()=>{
    const fname = `stats-${source}.json`;
    if (fs.existsSync(fname)) return JSON.parse(fs.readFileSync(fname,'utf8'));
    if (fs.existsSync('stats.json')) return JSON.parse(fs.readFileSync('stats.json','utf8'));
    return {};
  })();
  const pairs: Array<any> = statsJson.data || [];
  const diffPairs: Record<string, any> = {};
  for (const d of (statsDiff.pairsDiff || [])) diffPairs[d.pair] = d.diff;
  // Compute 7-day win rate trend (moving window) across all pairs
  function fmt(d: Date){ const y=d.getFullYear(); const m=String(d.getMonth()+1).padStart(2,'0'); const da=String(d.getDate()).padStart(2,'0'); return `${y}-${m}-${da}`; }
  function getStatsDir(){ return process.env.STATS_DIR || path.resolve(process.cwd(), 'logs'); }
  function loadDailyFor(date: string, pair?: string){
    try {
      const dir = getStatsDir();
      const p = pair ? path.join(dir, 'pairs', pair, `stats-${date}.json`) : path.join(dir, `stats-${date}.json`);
      if (!fs.existsSync(p)) return null;
      return JSON.parse(fs.readFileSync(p,'utf8'));
    } catch { return null; }
  }
  function computeTrend7dWinRate(): number {
    const today = new Date();
    let totalWins = 0; let totalTrades = 0;
    const pairNames = pairs.map((p:any)=>p.pair);
    for (let i=0;i<7;i++){
      const d = new Date(today.getFullYear(), today.getMonth(), today.getDate() - i);
      const key = fmt(d);
      if (pairNames.length){
        for (const pn of pairNames){
          const v = loadDailyFor(key, pn);
          if (v){ totalWins += Number(v.wins||0); totalTrades += Number(v.trades||0); }
        }
      } else {
        const v = loadDailyFor(key);
        if (v){ totalWins += Number(v.wins||0); totalTrades += Number(v.trades||0); }
      }
    }
    if (!(totalTrades>0)) return 0;
    return totalWins / totalTrades;
  }
  const totals = (()=>{
    const d = statsDiff.diff || {};
    const vals = statsDiff.values || {};
    const winRate = vals.trades ? (vals.wins||0)/(vals.trades||1) : 0;
    return {
      buy: d.buyEntries||0,
      sell: d.sellEntries||0,
      rsi: d.rsiExits||0,
      trail: d.trailExitTotal||0,
      PnL: d.realizedPnl||0,
      winRate,
  trend7dWinRate: computeTrend7dWinRate(),
      maxDrawdown: vals.maxDrawdown||0,
      maxConsecLosses: vals.maxConsecLosses||0,
      avgHoldSec: vals.avgHoldSec||0
    };
  })();
  const perPair = pairs.map((p:any)=>({
    pair: p.pair,
    PnL: p.stats.realizedPnl || 0,
    winRate: p.stats.winRate || 0,
    buy: (diffPairs[p.pair]?.buyEntries)||0,
    sell: (diffPairs[p.pair]?.sellEntries)||0,
    rsi: (diffPairs[p.pair]?.rsiExits)||0,
    trail: (diffPairs[p.pair]?.trailExitTotal)||0,
    maxDrawdown: p.stats.maxDrawdown || 0,
    maxConsecLosses: p.stats.maxConsecLosses || 0,
    avgHoldSec: p.stats.avgHoldSec || 0
  }));
  const body = { source, totals, perPair };
  const out = `report-summary-${source}.json`;
  fs.writeFileSync(out, JSON.stringify(body, null, 2));
  console.log(JSON.stringify({ out, pairs: perPair.length }));
})();
