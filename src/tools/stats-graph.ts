import fs from 'fs';
import path from 'path';
import { loadDaily } from '../utils/daily-stats';
import { loadPairs } from '../utils/config';

function todayStr(){ return new Date().toISOString().slice(0,10); }
const KEY_LIST = ['buyEntries','sellEntries','rsiExits','trailExitTotal','trailStops','winTrades','lossTrades'] as const;

(async ()=>{
  const args = process.argv.slice(2);
  const outJson = args.includes('--out') ? args[args.indexOf('--out')+1] : 'stats.json';
  const outSvg = args.includes('--svg') ? args[args.indexOf('--svg')+1] : 'stats.svg';
  const outTimeline = args.includes('--timeline') ? args[args.indexOf('--timeline')+1] : 'stats-timeline.svg';
  const date = todayStr();
  const pairs = loadPairs();
  const perPair = pairs.map(pair => {
    const s: any = loadDaily(date, pair);
    const vals: Record<string, number> = {} as any;
    for (const k of KEY_LIST){ (vals as any)[k] = Number(s[k] || 0); }
    const wins = Number(s.wins || 0); const trades = Number(s.trades || 0);
    const winRate = trades > 0 ? wins / trades : 0;
    return { pair, values: vals, realizedPnl: Number(s.realizedPnl||0), winRate, streakWin: Number(s.streakWin||0), streakLoss: Number(s.streakLoss||0) };
  });
  const aggregate = perPair.reduce((acc, p)=>{
    for (const k of KEY_LIST){ acc.values[k] = (acc.values[k]||0) + (p.values[k]||0); }
    acc.realizedPnl += p.realizedPnl; return acc;
  }, { values: {} as Record<string, number>, realizedPnl: 0 });
  const data = { date, pairs: perPair, aggregate };
  fs.writeFileSync(outJson, JSON.stringify(data, null, 2));

  // simple SVG bar chart
  const max = Math.max(1, ...perPair.flatMap(pp=> KEY_LIST.map(k=> pp.values[k])));
  const width = 640, height = 260, pad = 24; const barW = Math.floor((width - pad*2) / KEY_LIST.length) - 6; const scale = (height - pad*2 - 40) / max;
  let rects = '';
  KEY_LIST.forEach((k, i) => {
    const v = aggregate.values[k] || 0;
    const h = Math.max(0, v * scale);
    const x = pad + i * (barW + 6);
    const y = height - pad - h;
    rects += `<rect x="${x}" y="${y}" width="${barW}" height="${h}" fill="#4e79a7" />\n`;
    rects += `<text x="${x + barW/2}" y="${height-6}" font-size="10" text-anchor="middle">${String(k)}</text>\n`;
    rects += `<text x="${x + barW/2}" y="${y - 4}" font-size="10" text-anchor="middle">${v}</text>\n`;
  });
  // Draw PnL line
  const pnl = aggregate.realizedPnl;
  const pnlScale = (height - pad*2 - 40) / Math.max(1, Math.abs(pnl));
  const midY = Math.floor(height*0.65);
  let pnlElems = '';
  pnlElems += `<line x1="${pad}" y1="${midY}" x2="${width-pad}" y2="${midY}" stroke="#ddd" />`;
  const pnlY = midY - (pnl * pnlScale);
  pnlElems += `<circle cx="${width - pad - 10}" cy="${pnlY}" r="4" fill="#e15759" />`;
  pnlElems += `<text x="${width - pad - 10}" y="${pnlY - 6}" font-size="10" text-anchor="end">Total PnL ${pnl.toFixed(2)}</text>`;

  // Per-pair legend with PnL and winRate; emphasize ETH/JPY and XRP/JPY if present
  let legendY = pad;
  const highlight = new Set(['eth_jpy','xrp_jpy']);
  perPair.forEach(pp => {
    const line = `${pp.pair}: PnL=${pp.realizedPnl.toFixed(2)} Win=${(pp.winRate*100).toFixed(0)}%`;
    const weight = highlight.has(pp.pair) ? 'bold' : 'normal';
    pnlElems += `<text x="${width - pad}" y="${legendY}" font-size="11" font-weight="${weight}" text-anchor="end">${line}</text>`;
    legendY += 12;
  // Add streak information here
  // Example: pnlElems += `<text ... >Streak: ...</text>`;
  // This is where the streak logic will be implemented.
  });
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">\n<rect width="100%" height="100%" fill="#fff"/>\n${rects}\n${pnlElems}\n</svg>`;
  fs.writeFileSync(outSvg, svg);
  console.log(JSON.stringify({ outJson: path.resolve(outJson), outSvg: path.resolve(outSvg) }));

  // Timeline over last N days (default 7)
  const N = Number(process.env.STATS_TIMELINE_DAYS || 7);
  const days: string[] = [];
  for (let i=0;i<N;i++){
    const d = new Date(); d.setDate(d.getDate()-i);
    days.push(d.toISOString().slice(0,10));
  }
  const perPairSeries = pairs.map(pair=>{
    let cum = 0; const pnlSeries: number[] = []; const winSeries: number[] = [];
    days.slice().reverse().forEach(d=>{
      const s: any = loadDaily(d, pair);
      cum += Number(s.realizedPnl||0);
      const wr = (Number(s.wins||0) && Number(s.trades||0)) ? (Number(s.wins)/Number(s.trades)) : 0;
      pnlSeries.push(cum);
      winSeries.push(wr);
    });
    return { pair, pnlSeries, winSeries };
  });
  // Optionally, overlay live series (aggregate) if live summaries exist
  let liveSeries: number[] | null = null;
  try {
    const liveDir = path.resolve(process.cwd(), 'logs', 'live');
    const files = fs.existsSync(liveDir) ? fs.readdirSync(liveDir).filter(f=> f.startsWith('summary-') && f.endsWith('.json')).sort() : [];
    let cum = 0; const arr: number[] = [];
    for (const f of files){
      try { const j = JSON.parse(fs.readFileSync(path.join(liveDir,f),'utf8')); cum += Number(j?.stats?.incPnl||0); arr.push(cum); } catch {}
    }
    if (arr.length) liveSeries = arr;
  } catch {}
  // draw simple polyline timeline with paper/live legend
  const W = 720, H = 300, P = 30;
  const maxPnl = Math.max(1, ...perPairSeries.flatMap(s=> s.pnlSeries));
  const pathLines: string[] = [];
  perPairSeries.forEach((s, idx)=>{
    const color = ['#4e79a7','#e15759','#f28e2b','#76b7b2','#59a14f'][idx%5];
    const pts = s.pnlSeries.map((v,i)=>{
      const x = P + i*( (W-2*P)/(s.pnlSeries.length-1||1) );
      const y = H-P - (v/maxPnl)*(H-2*P);
      return `${x},${y}`;
    }).join(' ');
    pathLines.push(`<polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}" />`);
    // winRate overlay (scale 0..1 on same canvas, docked at top area)
    const ptsW = s.winSeries.map((v,i)=>{
      const x = P + i*( (W-2*P)/(s.winSeries.length-1||1) );
      const y = P + (1 - v) * (H-2*P) * 0.5; // use top half for winRate
      return `${x},${y}`;
    }).join(' ');
    pathLines.push(`<polyline fill="none" stroke="${color}" stroke-dasharray="4 3" stroke-width="1.5" points="${ptsW}" />`);
  });
  if (liveSeries && liveSeries.length>1){
    const ptsLive = liveSeries.map((v,i)=>{
      const x = P + i*( (W-2*P)/(liveSeries!.length-1||1) );
      const y = H-P - (v/Math.max(maxPnl,1))*(H-2*P);
      return `${x},${y}`;
    }).join(' ');
    pathLines.push(`<polyline fill="none" stroke="#9c755f" stroke-width="2" stroke-dasharray="2 2" points="${ptsLive}" />`);
  }
  // legend
  const legend = `<g font-size="11"><text x="${W-10}" y="${P}" text-anchor="end">累積PnL線（paper）</text><text x="${W-10}" y="${P+14}" text-anchor="end">勝率線（paper、破線）</text><text x="${W-10}" y="${P+28}" text-anchor="end">累積PnL線（live、点線）</text></g>`;
  const timelineSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}"><rect width="100%" height="100%" fill="#fff"/>${legend}${pathLines.join('')}</svg>`;
  fs.writeFileSync(outTimeline, timelineSvg);
})();
