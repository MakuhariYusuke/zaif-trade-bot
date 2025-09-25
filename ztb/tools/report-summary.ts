import dotenv from 'dotenv';
dotenv.config();
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { readLines } from '../utils/toolkit';

function getArg(name: string, def?: string){
  const args = process.argv.slice(2);
  const i = args.indexOf(`--${name}`);
  return i>=0 ? args[i+1] : def;
}

(()=>{
  const source = getArg('source', 'live')!;
  const cacheOut = `report-summary-${source}.json`;
  const sidecar = `report-summary-${source}.cache.json`;
  try {
    if (process.env.REPORT_SUMMARY_USE_CACHE !== '0' && fs.existsSync(cacheOut)){
      const today = new Date().toISOString().slice(0,10);
      const cached = JSON.parse(fs.readFileSync(cacheOut,'utf8'));
      // optional stronger cache key: stats/diff size signature matches
      let sizeOk = true;
  try {
        const meta = JSON.parse(fs.readFileSync(sidecar,'utf8')) as { statsSize?: number; diffSize?: number; signature?: string };
        const sStat = (()=>{ const fname = `stats-${source}.json`; return fs.existsSync(fname) ? fs.statSync(fname) : (fs.existsSync('stats.json') ? fs.statSync('stats.json') : null); })();
        const sDiff = (()=>{ const fname = `stats-diff-${source}.json`; return fs.existsSync(fname) ? fs.statSync(fname) : (fs.existsSync('stats-diff.json') ? fs.statSync('stats-diff.json') : null); })();
        const ss = sStat ? `${sStat.mtimeMs}:${sStat.size}` : '0:0';
        const sd = sDiff ? `${sDiff.mtimeMs}:${sDiff.size}` : '0:0';
        const curSig = crypto.createHash('sha1').update(`${ss}|${sd}`).digest('hex');
  if (meta && (typeof meta.statsSize==='number' || typeof meta.diffSize==='number')){
          if ((typeof meta.statsSize==='number' && meta.statsSize !== (sStat?.size||0)) || (typeof meta.diffSize==='number' && meta.diffSize !== (sDiff?.size||0))) sizeOk = false;
          if (meta.signature && meta.signature !== curSig) sizeOk = false;
        }
      } catch {}
      if (sizeOk && cached && cached.summaryText && (cached.date === undefined || String(cached.date||'').startsWith(today))){
        const sigShort = (()=>{ try{ const sStat = (()=>{ const fname = `stats-${source}.json`; return fs.existsSync(fname) ? fs.statSync(fname) : (fs.existsSync('stats.json') ? fs.statSync('stats.json') : null); })(); const sDiff = (()=>{ const fname = `stats-diff-${source}.json`; return fs.existsSync(fname) ? fs.statSync(fname) : (fs.existsSync('stats-diff.json') ? fs.statSync('stats-diff.json') : null); })(); const ss = sStat ? `${sStat.mtimeMs}:${sStat.size}` : '0:0'; const sd = sDiff ? `${sDiff.mtimeMs}:${sDiff.size}` : '0:0'; return crypto.createHash('sha1').update(`${ss}|${sd}`).digest('hex').slice(0,8);}catch{return '';} })();
        console.log(`[CACHE] report-summary using existing output sig=${sigShort}`);
        console.log(JSON.stringify({ out: cacheOut, pairs: (cached.perPair||[]).length }));
        return;
      }
    }
  } catch {}
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
      const pJsonl = p.replace(/\.json$/, '.jsonl');
      if (fs.existsSync(pJsonl)){
        try {
          // [STREAM] read last non-empty line from JSONL
          // eslint-disable-next-line no-console
          console.log(`[STREAM] reading JSONL ${pJsonl}`);
          const stat = fs.statSync(pJsonl);
          if (stat.size <= 5_000_000){
            const data = fs.readFileSync(pJsonl, 'utf8');
            const lines = data.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
            const last = lines[lines.length-1] || '';
            if (last) return JSON.parse(last);
          } else {
            // fallback to JSON when too large for sync JSONL path
            // eslint-disable-next-line no-console
            console.warn(`[STREAM] fallback JSON (file too large): ${p}`);
          }
        } catch { /* ignore and fallback */ }
      }
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
  // formatted helpers
  function fmtPct(x:number){ return `${(x*100).toFixed(1)}%`; }
  function fmtJPY(x:number){ const s = (x>=0?'+':'') + x.toFixed(0); return `${s} JPY`; }
  const sorted = [...perPair].sort((a,b)=> (b.PnL - a.PnL) || (b.winRate - a.winRate));
  const top3 = sorted.slice(0,3);
  const top3Table = [
    'Pair | PnL | WinRate',
    '---- | ---:| ------:',
    ...top3.map((p:any)=> `${p.pair} | ${p.PnL.toFixed(0)} | ${fmtPct(p.winRate)}`)
  ].join('\n');
  const summaryText = [
    `Source: ${source}`,
    `Total PnL: ${fmtJPY(totals.PnL)} / WinRate: ${fmtPct(totals.winRate)} / 7d: ${fmtPct(totals.trend7dWinRate)}`,
    `Top pairs:`,
    top3Table
  ].join('\n');
  const bodyOut = { date: (statsJson.date || new Date().toISOString().slice(0,10)), ...body, summaryText, top3Table };
  const out = `report-summary-${source}.json`;
  try { const dir = path.dirname(out); if (dir && dir !== '.' && !fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true }); } catch {}
  fs.writeFileSync(out, JSON.stringify(bodyOut, null, 2));
  // sidecar cache meta for stronger cache validation
  try {
    const sStat = (()=>{ const fname = `stats-${source}.json`; return fs.existsSync(fname) ? fs.statSync(fname) : (fs.existsSync('stats.json') ? fs.statSync('stats.json') : null); })();
    const sDiff = (()=>{ const fname = `stats-diff-${source}.json`; return fs.existsSync(fname) ? fs.statSync(fname) : (fs.existsSync('stats-diff.json') ? fs.statSync('stats-diff.json') : null); })();
    const statsSize = sStat?.size || 0; const diffSize = sDiff?.size || 0;
    const ss = sStat ? `${sStat.mtimeMs}:${sStat.size}` : '0:0';
    const sd = sDiff ? `${sDiff.mtimeMs}:${sDiff.size}` : '0:0';
    const signature = crypto.createHash('sha1').update(`${ss}|${sd}`).digest('hex');
    const fileListSig = crypto.createHash('sha1').update([fs.existsSync(`stats-${source}.json`) ? `stats-${source}.json` : (fs.existsSync('stats.json') ? 'stats.json' : ''), fs.existsSync(`stats-diff-${source}.json`) ? `stats-diff-${source}.json` : (fs.existsSync('stats-diff.json') ? 'stats-diff.json' : '')].filter(Boolean).join('|')).digest('hex');
    fs.writeFileSync(sidecar, JSON.stringify({ statsSize, diffSize, signature, fileListSig }, null, 2));
  } catch {}
  console.log(JSON.stringify({ out, pairs: perPair.length }));
})();
