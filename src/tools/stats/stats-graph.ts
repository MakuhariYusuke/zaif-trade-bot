import dotenv from 'dotenv';
dotenv.config();
import { loadDaily } from '../../utils/daily-stats';
import { loadPairs } from '../../utils/config';
import { todayStr } from '../../utils/toolkit';

const args = process.argv.slice(2);
function getArg(name: string, def?: string){ const i = args.indexOf(`--${name}`); return i>=0 ? args[i+1] : def; }
const outFile = getArg('out', 'stats.json')!;
const outSvg = getArg('svg', 'stats.svg')!;

(async ()=>{
  const d = todayStr();
  const pairs = loadPairs();
  const data = pairs.length ? pairs.map(p=> ({ pair:p, stats: loadDaily(d,p) })) : [{ pair:'all', stats: loadDaily(d) }];
  // load paper/live if available for overlay textual summary
  const paper = loadDaily(d);
  const live = loadDaily(d, 'btc_jpy');
  const fs = await import('fs');
  fs.writeFileSync(outFile, JSON.stringify({ date: d, data }, null, 2));
  // simple svg with legend and maxDrawdown line (placeholder visualization)
  const width=800, height=240;
  const legend = `<g font-size="12">
    <rect x="10" y="10" width="120" height="50" fill="white" stroke="#ccc"/>
    <text x="20" y="25">PnL線</text>
    <text x="20" y="40">勝率線</text>
    <text x="20" y="55">最大DD線</text>
  </g>`;
  const ddText = data.map((r,i)=>`${r.pair}: DD=${r.stats.maxDrawdown||0}`).join(' | ');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
    ${legend}
    <text x="10" y="90">${d} stats</text>
    <text x="10" y="110">${ddText}</text>
    <text x="10" y="130">paper: PnL=${paper.realizedPnl||0} Win%=${Math.round((paper.wins/(paper.trades||1))*100)||0}</text>
    <text x="10" y="150">live(btc_jpy): PnL=${live.realizedPnl||0} Win%=${Math.round((live.wins/(live.trades||1))*100)||0}</text>
  </svg>`;
  fs.writeFileSync(outSvg, svg);
  console.log(JSON.stringify({ out: outFile, svg: outSvg, pairs: data.length }));
})();
