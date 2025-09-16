import fs from 'fs';
import path from 'path';

function parseArgs(){
  const args = process.argv.slice(2);
  const get = (k: string, def?: string) => { const i = args.indexOf(`--${k}`); return i>=0 ? args[i+1] : def; };
  const dataset = get('dataset', path.resolve(process.cwd(), 'ml-dataset.jsonl'))!;
  const topN = Number(get('top', process.env.FI_TOP || '20'));
  const outJson = get('out-json', path.resolve(process.cwd(), 'report-ml-feature-importance.json'))!;
  const outCsv = get('out-csv', path.resolve(process.cwd(), 'importance.csv'))!;
  const labelPref = (get('label', process.env.FEATURE_IMPORTANCE_LABEL || 'win') || 'win').toLowerCase();
  return { dataset, topN, outJson, outCsv, labelPref };
}

type Row = Record<string, any>;

function isNum(v: any): v is number { return typeof v === 'number' && Number.isFinite(v); }

function pearson(x: number[], y: number[]): number | null {
  const n = Math.min(x.length, y.length);
  if (n === 0) return null;
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, c = 0;
  for (let i=0;i<n;i++){
    const xi = x[i]; const yi = y[i];
    if (!isNum(xi) || !isNum(yi)) continue;
    c++;
    sx += xi; sy += yi; sxx += xi*xi; syy += yi*yi; sxy += xi*yi;
  }
  if (c < 3) return null;
  const cov = sxy/c - (sx/c)*(sy/c);
  const vx = sxx/c - Math.pow(sx/c,2);
  const vy = syy/c - Math.pow(sy/c,2);
  if (vx <= 0 || vy <= 0) return 0;
  return cov / Math.sqrt(vx*vy);
}

async function main(){
  const { dataset, topN, outJson, outCsv, labelPref } = parseArgs();
  if (!fs.existsSync(dataset)) { console.warn(`[FI] dataset missing: ${dataset}`); return; }
  const rs = fs.createReadStream(dataset, { encoding: 'utf8' });
  const byFeat: Record<string, number[]> = {};
  const labels: number[] = [];
  const featsSeen = new Set<string>();
  let total = 0; let used = 0;
  let buf = '';
  await new Promise<void>((resolve)=>{
    rs.on('data', (chunk: string | Buffer)=>{
      buf += chunk.toString('utf8');
      let idx: number;
      while ((idx = buf.indexOf('\n')) >= 0){
        const line = buf.slice(0, idx); buf = buf.slice(idx+1);
        const t = line.trim(); if (!t) continue; total++;
        try {
          const o: Row = JSON.parse(t);
          let y: number | null = null;
          if (labelPref === 'win') {
            if (o.win === 1 || o.win === true) y = 1;
            else if (o.win === 0 || o.win === false) y = 0;
            else if (isNum(o.pnl)) y = o.pnl > 0 ? 1 : 0;
          } else if (labelPref === 'pnl') {
            if (isNum(o.pnl)) y = o.pnl;
            else if (o.win === 1 || o.win === true) y = 1; else if (o.win === 0 || o.win === false) y = 0; else y = null;
          }
          if (y == null) return;
          // build feature vector from numeric keys (exclude obvious non-features)
          const exclude = new Set<string>(['ts','qty','pnl','win','durationSec','bestBid','bestAsk']);
          const skipPrefix = ['balance','position'];
          const x: Record<string, number> = {};
          for (const [k,v] of Object.entries(o)){
            if (exclude.has(k)) continue;
            if (typeof v === 'number' && Number.isFinite(v)) {
              x[k] = v; featsSeen.add(k);
            }
          }
          const keys = Array.from(featsSeen);
          // align arrays
          for (const k of keys){ const arr = (byFeat[k] = byFeat[k] || []); arr.push(Object.prototype.hasOwnProperty.call(x, k) ? x[k] : NaN); }
          labels.push(y);
          used++;
        } catch {}
      }
    });
    rs.on('end', ()=> resolve());
    rs.on('error', ()=> resolve());
  });
  // compute importance per feature as |corr(x, y)| with NaN filtering
  type Score = { name: string; score: number; count: number };
  const scores: Score[] = [];
  for (const [name, arr] of Object.entries(byFeat)){
    // filter pairs where both x and y are numbers
    const xs: number[] = []; const ys: number[] = [];
    for (let i=0;i<arr.length;i++){
      const xi = arr[i]; const yi = labels[i];
      if (isNum(xi) && isNum(yi)) { xs.push(xi); ys.push(yi); }
    }
    const r = pearson(xs, ys);
    if (r == null) continue;
    scores.push({ name, score: Math.abs(r), count: xs.length });
  }
  scores.sort((a,b)=> (b.score - a.score) || (b.count - a.count));
  const top = scores.slice(0, topN);
  const date = new Date().toISOString().slice(0,10);
  const json = { date, samples: used, total, top_features: top.map(t=> ({ name: t.name, gain: Number(t.score.toFixed(6)) })) };
  try { fs.writeFileSync(outJson, JSON.stringify(json, null, 2), 'utf8'); } catch {}
  try { fs.writeFileSync(outCsv, ['name,score,count', ...scores.map(s=> `${s.name},${s.score},${s.count}`)].join('\n'), 'utf8'); } catch {}
  try { console.log(`[FI] ${used}/${total} samples; top: ${top.slice(0,5).map(t=>t.name).join(', ')}`); } catch {}
  if (process.env.VITEST_WORKER_ID){
    console.log(JSON.stringify({ outJson, outCsv }));
  }
}

main().catch(()=>{});
