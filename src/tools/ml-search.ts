import { spawnSync } from 'child_process';
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';

function* range(start: number, end: number, step: number){ for (let v=start; v<=end; v+=step) yield v; }

interface Trial { SELL_RSI_OVERBOUGHT: number; BUY_RSI_OVERSOLD: number; SMA_SHORT: number; SMA_LONG: number; }

function runSim(params: Trial){
  const path = require('path');
  const mlPath = path.resolve(process.cwd(), 'src', 'tools', 'ml-simulate.ts');
  const args = ['-e', `require('ts-node').register(); require('${mlPath.replace(/\\/g,'/')}');`, '--', '--pair', process.env.PAIR || 'btc_jpy', '--params', JSON.stringify(params)];
  const r = spawnSync('node', args, { encoding: 'utf8' });
  if (r.status !== 0) return null;
  try { return JSON.parse(r.stdout.trim()); } catch { return null; }
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
          const p = { SELL_RSI_OVERBOUGHT: ro, BUY_RSI_OVERSOLD: ru, SMA_SHORT: ss, SMA_LONG: sl };
          const res = runSim(p);
          if (res) results.push({ pair, ...p, ...res });
        }
      }
    }
  }
  return results;
}

const isTs = __filename.endsWith('.ts');

if (!isMainThread){
  const pair: string = workerData.pair;
  const results = runGridForPair(pair);
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
      // serial fallback (ts-node or ML_NO_WORKERS=1)
      for (const pair of pairs){
        results.push(...runGridForPair(pair));
      }
    }
    const fs = await import('fs');
    const csv = [
      'pair,SELL_RSI_OVERBOUGHT,BUY_RSI_OVERSOLD,SMA_SHORT,SMA_LONG,winRate,pnl,trades,avgHoldSec',
      ...results.map(r=> [r.pair,r.SELL_RSI_OVERBOUGHT,r.BUY_RSI_OVERSOLD,r.SMA_SHORT,r.SMA_LONG,r.winRate,r.pnl,r.trades,r.avgHoldSec].join(','))
    ].join('\n');
    fs.writeFileSync('ml-search-results.csv', csv);
    const top = results.sort((a,b)=> b.winRate - a.winRate || b.pnl - a.pnl).slice(0,5);
    fs.writeFileSync('ml-search-top.json', JSON.stringify({ top }, null, 2));
    console.log(JSON.stringify({ top }));
  })();
}
