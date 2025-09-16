import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
// lazy requires to avoid worker context issues
const _fs = require('fs') as typeof import('fs');
const _path = require('path') as typeof import('path');
import { spawnSync } from 'child_process';
import { runMlSimulate } from '../../utils/toolkit';

function* range(start: number, end: number, step: number){ for (let v=start; v<=end; v+=step) yield v; }

interface Trial { 
    SELL_RSI_OVERBOUGHT: number;
    BUY_RSI_OVERSOLD: number;
    SMA_SHORT: number;
    SMA_LONG: number;
}

function runSim(params: Trial){
  return runMlSimulate(params as any, process.env.PAIR || 'btc_jpy');
}

function runGridForPair(pair: string){
  const rsiOver = Array.from(range(55, 75, 5));
  const rsiUnder = Array.from(range(20, 35, 5));
  const sShort = Array.from(range(7, 15, 2));
  const sLong = Array.from(range(21, 35, 2));
  const results: any[] = [];
  for (const ro of rsiOver){
    for (const ru of rsiUnder){
      for (const ss of sShort){
        for (const sl of sLong){
          if (ss >= sl) continue;
          const p = { 
            SELL_RSI_OVERBOUGHT: ro, 
            BUY_RSI_OVERSOLD: ru, 
            SMA_SHORT: ss, 
            SMA_LONG: sl 
          };
          const res = runSim(p);
          if (res) results.push({ pair, ...p, ...res });
        }
      }
    }
  }
  return results;
}

function randInt(min: number, max: number){ return Math.floor(min + Math.random() * (max - min + 1)); }
function runRandomForPair(pair: string, steps: number){
  const results: any[] = [];
  for (let i=0;i<steps;i++){
    const ss = randInt(5, 20);
    const sl = randInt(15, 40);
    if (ss >= sl) { i--; continue; }
    const p = {
      SELL_RSI_OVERBOUGHT: randInt(55, 80),
      BUY_RSI_OVERSOLD: randInt(15, 40),
      SMA_SHORT: ss,
      SMA_LONG: sl
    };
    const res = runSim(p);
    if (res) results.push({ pair, ...p, ...res });
  }
  return results;
}

function runEarlyStopForPair(pair: string, patience: number, maxSteps: number){
  let bestScore = -Infinity; let noImprove = 0; const results: any[] = [];
  for (let i=0;i<maxSteps;i++){
    const stepRes = runRandomForPair(pair, 1);
    if (stepRes.length){
      const r = stepRes[0];
      const score = (r.winRate || 0) * 100 + (r.pnl || 0);
      results.push(r);
      if (score > bestScore + 1e-9){ bestScore = score; noImprove = 0; } else { noImprove++; }
      if (noImprove >= patience) break;
    }
  }
  return results;
}

const isTs = __filename.endsWith('.ts');
// Only treat as our own worker when workerData.pair is provided (avoid interfering with Vitest/Tinypool workers)
const isOurWorker = !isMainThread && !!(workerData && (workerData as any).pair);

if (isOurWorker){
  const pair: string = (workerData as any).pair;
  const mode = (process.env.ML_SEARCH_MODE || 'grid').toLowerCase();
  let results: any[] = [];
  if (mode === 'random') results = runRandomForPair(pair, Number(process.env.ML_RANDOM_STEPS || 100));
  else if (mode === 'earlystop') results = runEarlyStopForPair(pair, Number(process.env.ML_EARLY_PATIENCE || 10), Number(process.env.ML_EARLY_MAX_STEPS || 200));
  else results = runGridForPair(pair);
  if (parentPort) parentPort.postMessage(results);
} else {
  // --- dataset caching control ---
  const args = process.argv.slice(2);
  const hasFlag = (f: string) => args.includes(f);
  const useCache = hasFlag('--use-cache') && !hasFlag('--no-cache');
  const datasetPath = _path.resolve(process.cwd(), 'ml-dataset.jsonl');
  const outDir = _path.dirname(datasetPath);
  if (useCache && _fs.existsSync(datasetPath)) {
    console.log('[CACHE] using existing dataset');
  } else if (!useCache) {
    // force regenerate
    const mlExp = _path.resolve(process.cwd(), 'src', 'tools', 'ml', 'ml-export.ts');
    const r = spawnSync('node', ['-e', `require('ts-node').register(); require('${mlExp.replace(/\\/g,'/')}');`], { encoding: 'utf8', env: { ...process.env, QUIET: process.env.QUIET ?? '1' } });
    if (r.status !== 0) process.stderr.write(r.stderr || '');
  } else {
    // use-cache requested but file missing -> generate once
    const mlExp = _path.resolve(process.cwd(), 'src', 'tools', 'ml', 'ml-export.ts');
    const r = spawnSync('node', ['-e', `require('ts-node').register(); require('${mlExp.replace(/\\/g,'/')}');`], { encoding: 'utf8', env: { ...process.env, QUIET: process.env.QUIET ?? '1' } });
    if (r.status !== 0) process.stderr.write(r.stderr || '');
  }
  const pairs = (process.env.PAIRS || process.env.PAIR || 'btc_jpy').split(',').map(s=>s.trim()).filter(Boolean);
  const results: any[] = [];
  const useWorkers = !isTs && !process.env.ML_NO_WORKERS;
  if (useWorkers){
    (async ()=>{
      // Auto-tune workers: CI -> 1, else min(CPU, 4)
      let maxWorkers = Number(process.env.ML_MAX_WORKERS || 0);
      try {
        if (!maxWorkers) {
          if (process.env.CI === 'true' || process.env.FAST_CI === '1') {
            maxWorkers = 1;
          } else {
            const os = require('os') as typeof import('os');
            const cpu = Math.max(1, Math.min(4, (os.cpus?.().length || 1)));
            maxWorkers = cpu;
          }
        }
      } catch { maxWorkers = 1; }
      const queue = [...pairs];
      const workers: Worker[] = [];
      async function startNext(){
        if (!queue.length) return;
        const pair = queue.shift()!;
        const w = new Worker(__filename, { workerData: { pair } });
        workers.push(w);
        w.on('message', (res: any[]) => { results.push(...res); });
        w.on('error', ()=>{});
        w.on('exit', async ()=>{ await startNext(); });
      }
      for (let i=0;i<Math.min(maxWorkers, queue.length);i++) await startNext();
      await Promise.allSettled(workers.map(w => new Promise<void>(res => w.once('exit', ()=>res()))));
      // After async worker completion, write outputs
      const outCsvPath = _path.resolve(outDir, 'ml-search-results.csv');
      try { const d = _path.dirname(outCsvPath); if (!_fs.existsSync(d)) _fs.mkdirSync(d, { recursive: true }); } catch {}
      const csv = [
        'pair,SELL_RSI_OVERBOUGHT,BUY_RSI_OVERSOLD,SMA_SHORT,SMA_LONG,winRate,pnl,trades,avgHoldSec,sharpe,sortino,calmar',
        ...results.map(r=> [
          r.pair,
          r.SELL_RSI_OVERBOUGHT,
          r.BUY_RSI_OVERSOLD,
          r.SMA_SHORT,
          r.SMA_LONG,
          r.winRate,
          r.pnl,
          r.trades,
          r.avgHoldSec,
          r.sharpe ?? '',
          r.sortino ?? '',
          r.calmar ?? ''
        ].join(','))
      ].join('\n');
  _fs.writeFileSync(outCsvPath, csv);
  try { console.log(`[ML-SEARCH] wrote results csv rows=${results.length}`); } catch {}
      const top = results.sort((a,b)=> (b.winRate - a.winRate) || (b.pnl - a.pnl) || ((b.calmar||0) - (a.calmar||0)) || ((b.sharpe||0) - (a.sharpe||0))).slice(0,5);
  const topJsonPath = _path.resolve(outDir, 'ml-search-top.json');
  try { const d = _path.dirname(topJsonPath); if (!_fs.existsSync(d)) _fs.mkdirSync(d, { recursive: true }); } catch {}
      _fs.writeFileSync(topJsonPath, JSON.stringify({ top }, null, 2));
  try { const best = top[0]; if (best) console.log(`[ML-TOP] pair=${best.pair} win=${Math.round((best.winRate||0)*100)} pnl=${best.pnl||0} S=${best.SMA_SHORT},L=${best.SMA_LONG},RSI=${best.SELL_RSI_OVERBOUGHT},${best.BUY_RSI_OVERSOLD}`); } catch {}
      const mode = (process.env.ML_SEARCH_MODE || 'grid').toLowerCase();
      const report = { mode, top };
      _fs.writeFileSync(_path.resolve(outDir, `report-ml-${mode}.json`), JSON.stringify(report, null, 2));
      _fs.writeFileSync(_path.resolve(outDir, `report-ml-${mode}.csv`), [
        'pair,SELL_RSI_OVERBOUGHT,BUY_RSI_OVERSOLD,SMA_SHORT,SMA_LONG,winRate,pnl,trades,avgHoldSec,sharpe,sortino,calmar',
        ...top.map(r=> [r.pair,r.SELL_RSI_OVERBOUGHT,r.BUY_RSI_OVERSOLD,r.SMA_SHORT,r.SMA_LONG,r.winRate,r.pnl,r.trades,r.avgHoldSec,r.sharpe??'',r.sortino??'',r.calmar??''].join(','))
      ].join('\n'));
      console.log(JSON.stringify(report));
    })();
  } else {
    const mode = (process.env.ML_SEARCH_MODE || 'grid').toLowerCase();
    for (const pair of pairs){
      if (mode === 'random') results.push(...runRandomForPair(pair, Number(process.env.ML_RANDOM_STEPS || 100)));
      else if (mode === 'earlystop') results.push(...runEarlyStopForPair(pair, Number(process.env.ML_EARLY_PATIENCE || 10), Number(process.env.ML_EARLY_MAX_STEPS || 200)));
      else results.push(...runGridForPair(pair));
    }
  const outCsvPath = _path.resolve(outDir, 'ml-search-results.csv');
  try { const d = _path.dirname(outCsvPath); if (!_fs.existsSync(d)) _fs.mkdirSync(d, { recursive: true }); } catch {}
    const csv = [
      'pair,SELL_RSI_OVERBOUGHT,BUY_RSI_OVERSOLD,SMA_SHORT,SMA_LONG,winRate,pnl,trades,avgHoldSec,sharpe,sortino,calmar',
      ...results.map(r=> [
        r.pair,
        r.SELL_RSI_OVERBOUGHT,
        r.BUY_RSI_OVERSOLD,
        r.SMA_SHORT,
        r.SMA_LONG,
        r.winRate,
        r.pnl,
        r.trades,
        r.avgHoldSec,
        r.sharpe ?? '',
        r.sortino ?? '',
        r.calmar ?? ''
      ].join(','))
    ].join('\n');
    _fs.writeFileSync(outCsvPath, csv);
  try { console.log(`[ML-SEARCH] wrote results csv rows=${results.length}`); } catch {}
    if (process.env.VITEST_WORKER_ID) {
      console.log(JSON.stringify({ out: outCsvPath }));
    }
    const top = results.sort((a,b)=> (b.winRate - a.winRate) || (b.pnl - a.pnl) || ((b.calmar||0) - (a.calmar||0)) || ((b.sharpe||0) - (a.sharpe||0))).slice(0,5);
  const topJsonPath = _path.resolve(outDir, 'ml-search-top.json');
  try { const d = _path.dirname(topJsonPath); if (!_fs.existsSync(d)) _fs.mkdirSync(d, { recursive: true }); } catch {}
  _fs.writeFileSync(topJsonPath, JSON.stringify({ top }, null, 2));
  try { const best = top[0]; if (best) console.log(`[ML-TOP] pair=${best.pair} win=${Math.round((best.winRate||0)*100)} pnl=${best.pnl||0} S=${best.SMA_SHORT},L=${best.SMA_LONG},RSI=${best.SELL_RSI_OVERBOUGHT},${best.BUY_RSI_OVERSOLD}`); } catch {}
    if (process.env.VITEST_WORKER_ID) {
      console.log(JSON.stringify({ topPath: topJsonPath }));
    }
    // additional report artifact per mode
    const report = { mode, top };
    _fs.writeFileSync(_path.resolve(outDir, `report-ml-${mode}.json`), JSON.stringify(report, null, 2));
    _fs.writeFileSync(_path.resolve(outDir, `report-ml-${mode}.csv`), [
      'pair,SELL_RSI_OVERBOUGHT,BUY_RSI_OVERSOLD,SMA_SHORT,SMA_LONG,winRate,pnl,trades,avgHoldSec,sharpe,sortino,calmar',
      ...top.map(r=> [r.pair,r.SELL_RSI_OVERBOUGHT,r.BUY_RSI_OVERSOLD,r.SMA_SHORT,r.SMA_LONG,r.winRate,r.pnl,r.trades,r.avgHoldSec,r.sharpe??'',r.sortino??'',r.calmar??''].join(','))
    ].join('\n'));
    console.log(JSON.stringify(report));
  }
}
