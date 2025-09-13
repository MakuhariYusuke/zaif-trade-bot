import fs from 'fs';
import path from 'path';

function readAllFeatureCsv(root: string){
  const out: Array<{file:string; source:'paper'|'live'}> = [];
  if (!fs.existsSync(root)) return out;
  const candidates = ['paper','live'];
  for (const src of candidates){
    const srcDir = path.join(root, src);
    if (!fs.existsSync(srcDir) || !fs.statSync(srcDir).isDirectory()) continue;
    const pairs = fs.readdirSync(srcDir).filter(n => fs.statSync(path.join(srcDir,n)).isDirectory());
    for (const pair of pairs){
      const dir = path.join(srcDir, pair);
      const files = fs.readdirSync(dir).filter(f => f.startsWith('features-') && f.endsWith('.csv'));
      for (const f of files){ out.push({ file: path.join(dir, f), source: (src as 'paper'|'live') }); }
    }
  }
  // Fallback: legacy root-level pair dirs
  const rootPairs = fs.readdirSync(root).filter(n => {
    const p = path.join(root,n); return fs.statSync(p).isDirectory() && !candidates.includes(n);
  });
  for (const pair of rootPairs){
    const dir = path.join(root, pair);
    const files = fs.readdirSync(dir).filter(f => f.startsWith('features-') && f.endsWith('.csv'));
    for (const f of files){ out.push({ file: path.join(dir, f), source: 'paper' }); }
  }
  return out;
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
  const root = path.resolve(process.cwd(), 'logs', 'features');
  const files = readAllFeatureCsv(root);
  const out = path.resolve(process.cwd(), 'ml-dataset.csv');
  mergeCsv(files, out);
  console.log(JSON.stringify({ dataset: out, count: files.length }));
})();
