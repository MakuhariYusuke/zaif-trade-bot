import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
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

async function mergeJsonl(files: Array<{file:string; format: 'csv'|'jsonl'; source:'paper'|'live'}>, outPath: string){
  const tradeFlow = process.env.TRADE_FLOW || '';
  const out = fs.createWriteStream(outPath, { encoding: 'utf8' });
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
          const rec = { ...o, source, tradeFlow, durationSec };
          out.write(JSON.stringify(rec) + '\n');
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
        const rec: any = { ts:Number(cols[ci.ts]), pair:cols[ci.pair], side:cols[ci.side], rsi:cols[ci.rsi]||undefined, sma_short:cols[ci.sma_short]||undefined, sma_long:cols[ci.sma_long]||undefined, price:Number(cols[ci.price]), qty:Number(cols[ci.qty]) };
        if (pnlStr !== '') rec.pnl = Number(pnlStr);
        if (winStr !== '') rec.win = (winStr==='1') ? 1 : 0;
        out.write(JSON.stringify({ ...rec, source, tradeFlow, durationSec }) + '\n');
      }
    }
  })());
  await Promise.allSettled(tasks);
  out.end();
}

(async ()=>{
  const { date } = parseArgs();
  const root = getFeaturesRoot();
  const files = readAllFeatureFiles(root, date);
  const out = path.resolve(process.cwd(), 'ml-dataset.jsonl');
  const cacheMetaPath = path.resolve(process.cwd(), 'ml-dataset.cache.json');
  try {
  if (process.env.ML_EXPORT_USE_CACHE !== '0' && fs.existsSync(out)){
      // if output is newer than all inputs, reuse
      const outSt = fs.statSync(out);
      const outMtime = outSt.mtimeMs;
      let newestInput = 0; let totalInputSize = 0;
      const sigParts: string[] = [];
      for (const f of files){
        try { const st = fs.statSync(f.file); if (st.mtimeMs > newestInput) newestInput = st.mtimeMs; totalInputSize += st.size||0; sigParts.push(`${f.file}:${st.mtimeMs}:${st.size||0}`); } catch {}
      }
      const signature = crypto.createHash('sha1').update(sigParts.join('|')).digest('hex');
      const fileListSig = crypto.createHash('sha1').update(files.map(f=>f.file).sort().join('|')).digest('hex');
      // optional size/hash signature check via sidecar cache
      let cacheOk = false;
      try {
        const meta = JSON.parse(fs.readFileSync(cacheMetaPath,'utf8')) as { totalInputSize?: number; signature?: string; fileListSig?: string };
        if (meta && typeof meta.totalInputSize === 'number' && meta.totalInputSize === totalInputSize && meta.signature === signature && meta.fileListSig === fileListSig) cacheOk = true;
      } catch {}
      if (newestInput && outMtime >= newestInput && cacheOk){
        try { console.log(`[CACHE] ml-export using existing dataset sig=${signature.slice(0,8)}`); } catch {}
        console.log(JSON.stringify({ dataset: out, count: files.length, date: date || null }));
        return;
      }
    }
  } catch {}
  await mergeJsonl(files, out);
  // write/update sidecar cache meta
  try {
  let totalInputSize = 0; const sigParts: string[] = []; let newestInput=0;
  for (const f of files){ try { const st = fs.statSync(f.file); totalInputSize += st.size||0; if (st.mtimeMs>newestInput) newestInput=st.mtimeMs; sigParts.push(`${f.file}:${st.mtimeMs}:${st.size||0}`); } catch {} }
  const signature = crypto.createHash('sha1').update(sigParts.join('|')).digest('hex');
  const fileListSig = crypto.createHash('sha1').update(files.map(f=>f.file).sort().join('|')).digest('hex');
  fs.writeFileSync(cacheMetaPath, JSON.stringify({ totalInputSize, newestInput, signature, fileListSig }, null, 2));
  } catch {}
  console.log(JSON.stringify({ dataset: out, count: files.length, date: date || null }));
})();
