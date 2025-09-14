import fs from 'fs';
import path from 'path';
import { enumerateFeatureFiles, getFeaturesRoot, readLines } from '../../utils/toolkit';

function parseArgs(){
  const args = process.argv.slice(2);
  const get = (k: string, def?: string) => { const i = args.indexOf(`--${k}`); return i>=0 ? args[i+1] : def; };
  const date = get('date'); // YYYY-MM-DD で features-YYYY-MM-DD.* を対象
  return { date };
}

function readAllFeatureFiles(root: string, date?: string){
  const all = enumerateFeatureFiles(root).map(({ file, source, format }) => ({ file, format, source: (source === 'root' ? 'paper' : source) as 'paper'|'live' }));
  if (!date) return all;
  return all.filter(f => f.file.includes(date));
}

async function mergeCsv(files: Array<{file:string; format: 'csv'|'jsonl'; source:'paper'|'live'}>, outPath: string){
  const header = 'ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win,source,tradeFlow,durationSec';
  const tradeFlow = process.env.TRADE_FLOW || '';
  const chunks: string[] = [header + '\n'];
  const tasks = files.map(({file:fp, format, source})=> (async()=>{
    let lastTs: number | null = null;
    if (format === 'jsonl'){
      for await (const line of readLines(fp)){
        const t = String(line).trim(); if (!t) continue;
        try {
          const o = JSON.parse(t);
          const ts = Number(o.ts);
          let durationSec: string|number = '';
          const hasExit = (o.pnl != null && String(o.pnl) !== '' && !Number.isNaN(Number(o.pnl))) || (o.win === 0 || o.win === 1);
          if (hasExit){
            if (lastTs && Number.isFinite(lastTs) && Number.isFinite(ts)) durationSec = Math.max(0, Math.round((ts - lastTs)/1000));
            lastTs = null;
          } else {
            lastTs = ts;
          }
          const csv = [o.ts,o.pair,o.side,o.rsi??'',o.sma_short??'',o.sma_long??'',o.price,o.qty,o.pnl??'',(typeof o.win==='boolean'?(o.win?1:0):o.win??'')].join(',');
          chunks.push(csv + ',"' + source + '","' + tradeFlow + '",' + String(durationSec) + '\n');
        } catch {}
      }
    } else {
  let hdr: string[] | null = null;
  let ci: any = null;
  for await (const line of readLines(fp)){
        const t = String(line).trim(); if (!t) continue;
        if (!hdr){
          hdr = t.split(',');
          const idx = (name: string) => hdr!.indexOf(name);
          ci = { ts: idx('ts'), pair: idx('pair'), side: idx('side'), rsi: idx('rsi'), sma_short: idx('sma_short'), sma_long: idx('sma_long'), price: idx('price'), qty: idx('qty'), pnl: idx('pnl'), win: idx('win') };
          continue;
        }
        const cols = t.split(',');
        const tsStr = cols[ci.ts];
        const pnlStr = ci.pnl>=0 ? cols[ci.pnl] : '';
        const winStr = ci.win>=0 ? cols[ci.win] : '';
        const ts = Number(tsStr);
        let durationSec: string|number = '';
        const hasExit = (pnlStr != null && pnlStr !== '' && !Number.isNaN(Number(pnlStr))) || (winStr === '0' || winStr === '1');
        if (hasExit) {
          if (lastTs && Number.isFinite(lastTs) && Number.isFinite(ts)) {
            durationSec = Math.max(0, Math.round((ts - lastTs) / 1000));
          } else {
            durationSec = '';
          }
          lastTs = null;
        } else {
          lastTs = ts;
        }
        chunks.push(t + ',"' + source + '","' + tradeFlow + '",' + String(durationSec) + '\n');
      }
    }
  })());
  await Promise.allSettled(tasks);
  fs.writeFileSync(outPath, chunks.join(''));
}

(async ()=>{
  const { date } = parseArgs();
  const root = getFeaturesRoot();
  const files = readAllFeatureFiles(root, date);
  const out = path.resolve(process.cwd(), 'ml-dataset.csv');
  try {
    if (process.env.ML_EXPORT_USE_CACHE !== '0' && fs.existsSync(out)){
      // if output is newer than all inputs, reuse
      const outMtime = fs.statSync(out).mtimeMs;
      let newestInput = 0;
      for (const f of files){
        try { const st = fs.statSync(f.file); if (st.mtimeMs > newestInput) newestInput = st.mtimeMs; } catch {}
      }
      if (newestInput && outMtime >= newestInput){
        try { console.log('[CACHE] ml-export using existing dataset'); } catch {}
        console.log(JSON.stringify({ dataset: out, count: files.length, date: date || null }));
        return;
      }
    }
  } catch {}
  await mergeCsv(files, out);
  console.log(JSON.stringify({ dataset: out, count: files.length, date: date || null }));
})();
