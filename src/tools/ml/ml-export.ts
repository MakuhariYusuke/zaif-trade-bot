import fs from 'fs';
import path from 'path';
import { enumerateFeatureCsvFiles, getFeaturesRoot } from '../../utils/toolkit';

function readAllFeatureCsv(root: string){
  return enumerateFeatureCsvFiles(root).map(({ file, source }) => ({ file, source: (source === 'root' ? 'paper' : source) as 'paper'|'live' }));
}

function mergeCsv(files: Array<{file:string; source:'paper'|'live'}>, outPath: string){
  const header = 'ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win,source';
  let wroteHeader = false;
  const ws = fs.createWriteStream(outPath);
  ws.write(header + '\n'); wroteHeader = true;
  for (const {file:fp, source} of files){
    const txt = fs.readFileSync(fp, 'utf8');
    const [hdr, ...lines] = txt.trim().split(/\r?\n/);
    for (const line of lines){ if (line.trim()) ws.write(line + ',"' + source + '"\n'); }
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
