import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
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

if (!isMainThread){
  const pair: string = workerData.pair;
  const mode = (process.env.ML_SEARCH_MODE || 'grid').toLowerCase();
  let results: any[] = [];
  if (mode === 'random') results = runRandomForPair(pair, Number(process.env.ML_RANDOM_STEPS || 100));
  else if (mode === 'earlystop') results = runEarlyStopForPair(pair, Number(process.env.ML_EARLY_PATIENCE || 10), Number(process.env.ML_EARLY_MAX_STEPS || 200));
  else results = runGridForPair(pair);
  parentPort!.postMessage(results);
} else {
  (async ()=>{
    const pairs = (process.env.PAIRS || process.env.PAIR || 'btc_jpy').split(',').map(s=>s.trim()).filter(Boolean);
    const results: any[] = [];
    const useWorkers = !isTs && !process.env.ML_NO_WORKERS;
  if (useWorkers){
      const maxWorkers = Number(process.env.ML_MAX_WORKERS || 2);
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
      await new Promise<void>(resolve => {
        const check = setInterval(()=>{
          if (workers.every(w=> w.threadId === -1)) { clearInterval(check); resolve(); }
        }, 200);
      });
    } else {
      const mode = (process.env.ML_SEARCH_MODE || 'grid').toLowerCase();
      for (const pair of pairs){
        if (mode === 'random') results.push(...runRandomForPair(pair, Number(process.env.ML_RANDOM_STEPS || 100)));
        else if (mode === 'earlystop') results.push(...runEarlyStopForPair(pair, Number(process.env.ML_EARLY_PATIENCE || 10), Number(process.env.ML_EARLY_MAX_STEPS || 200)));
        else results.push(...runGridForPair(pair));
      }
    }
    const fs = await import('fs');
    const csv = [
      'pair,SELL_RSI_OVERBOUGHT,BUY_RSI_OVERSOLD,SMA_SHORT,SMA_LONG,winRate,pnl,trades,avgHoldSec',
      ...results.map(r=> [
        r.pair,
        r.SELL_RSI_OVERBOUGHT,
        r.BUY_RSI_OVERSOLD,
        r.SMA_SHORT,
        r.SMA_LONG,
        r.winRate,
        r.pnl,
        r.trades,
        r.avgHoldSec
      ].join(','))
    ].join('\n');
    fs.writeFileSync('ml-search-results.csv', csv);
    const top = results.sort((a,b)=> b.winRate - a.winRate || b.pnl - a.pnl).slice(0,5);
    fs.writeFileSync('ml-search-top.json', JSON.stringify({ top }, null, 2));
    // additional report artifact per mode
    const mode = (process.env.ML_SEARCH_MODE || 'grid').toLowerCase();
    const report = { mode, top };
    fs.writeFileSync(`report-ml-${mode}.json`, JSON.stringify(report, null, 2));
    fs.writeFileSync(`report-ml-${mode}.csv`, [
      'pair,SELL_RSI_OVERBOUGHT,BUY_RSI_OVERSOLD,SMA_SHORT,SMA_LONG,winRate,pnl',
      ...top.map(r=> [r.pair,r.SELL_RSI_OVERBOUGHT,r.BUY_RSI_OVERSOLD,r.SMA_SHORT,r.SMA_LONG,r.winRate,r.pnl].join(','))
    ].join('\n'));
    console.log(JSON.stringify(report));
  })();
}
