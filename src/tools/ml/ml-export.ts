import fs from 'fs';
import path from 'path';
import { enumerateFeatureCsvFiles, getFeaturesRoot } from '../../utils/toolkit';

function readAllFeatureCsv(root: string){
  return enumerateFeatureCsvFiles(root).map(({ file, source }) => ({ file, source: (source === 'root' ? 'paper' : source) as 'paper'|'live' }));
}

function mergeCsv(files: Array<{file:string; source:'paper'|'live'}>, outPath: string){
  const header = 'ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win,source,tradeFlow,durationSec';
  const ws = fs.createWriteStream(outPath);
  ws.write(header + '\n');
  const tradeFlow = process.env.TRADE_FLOW || '';
  // maintain simple per-file last ts to approximate hold duration at exit rows
  for (const {file:fp, source} of files){
    const txt = fs.readFileSync(fp, 'utf8');
    const lines = txt.trim().split(/\r?\n/);
    if (lines.length <= 1) continue; // header only
    const hdr = lines[0].split(',');
    const idx = (name: string) => hdr.indexOf(name);
    const ci = {
      ts: idx('ts'), pair: idx('pair'), side: idx('side'), rsi: idx('rsi'), sma_short: idx('sma_short'), sma_long: idx('sma_long'), price: idx('price'), qty: idx('qty'), pnl: idx('pnl'), win: idx('win')
    };
    let lastTs: number | null = null;
    for (let i=1;i<lines.length;i++){
      const line = lines[i];
      if (!line.trim()) continue;
      const cols = line.split(',');
      const tsStr = cols[ci.ts];
      const pnlStr = ci.pnl>=0 ? cols[ci.pnl] : '';
      const winStr = ci.win>=0 ? cols[ci.win] : '';
      const ts = Number(tsStr);
      let durationSec = '' as string|number;
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
      ws.write(line + ',"' + source + '","' + tradeFlow + '",' + String(durationSec) + '\n');
    }
  }
  ws.end();
}

(async ()=>{
  const root = getFeaturesRoot();
  const files = readAllFeatureCsv(root);
  const out = path.resolve(process.cwd(), 'ml-dataset.csv');
  mergeCsv(files, out);
  console.log(JSON.stringify({ dataset: out, count: files.length }));
})();
