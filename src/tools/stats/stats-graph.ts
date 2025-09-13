import dotenv from 'dotenv';
dotenv.config();
import { loadDaily } from '../../utils/daily-stats';
import { loadPairs } from '../../utils/config';

function today(){ return new Date().toISOString().slice(0,10); }

const args = process.argv.slice(2);
function getArg(name: string, def?: string){ const i = args.indexOf(`--${name}`); return i>=0 ? args[i+1] : def; }
const outFile = getArg('out', 'stats.json')!;
const outSvg = getArg('svg', 'stats.svg')!;

(async ()=>{
  const d = today();
  const pairs = loadPairs();
  const data = pairs.length ? pairs.map(p=> ({ pair:p, stats: loadDaily(d,p) })) : [{ pair:'all', stats: loadDaily(d) }];
  const fs = await import('fs');
  fs.writeFileSync(outFile, JSON.stringify({ date: d, data }, null, 2));
  // simple placeholder svg
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="600" height="200"><text x="10" y="20">${d} stats</text></svg>`;
  fs.writeFileSync(outSvg, svg);
  console.log(JSON.stringify({ out: outFile, svg: outSvg, pairs: data.length }));
})();
