import fs from 'fs';
import path from 'path';
import { loadDaily } from '../utils/daily-stats';

function isoWeek(d: Date): { year: number; week: number } {
  const date = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
  const dayNum = date.getUTCDay() || 7; // Mon=1..Sun=7
  date.setUTCDate(date.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(date.getUTCFullYear(), 0, 1));
  const weekNo = Math.ceil((((date.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
  return { year: date.getUTCFullYear(), week: weekNo };
}

function dateStr(d: Date): string { return d.toISOString().slice(0,10); }

function summarize(days: Array<{date: string; agg: ReturnType<typeof loadDaily>}>){
  const trades = days.reduce((a,d)=>a+(d.agg.trades||0),0);
  const wins = days.reduce((a,d)=>a+(d.agg.wins||0),0);
  const pnl = days.reduce((a,d)=>a+(d.agg.realizedPnl||0),0);
  const winRate = trades ? wins / trades : 0;
  const maxDD = Math.max(...days.map(d=>d.agg.maxDrawdown||0), 0);
  return { trades, wins, winRate, PnL: pnl, maxDrawdown: maxDD };
}

(async ()=>{
  const today = new Date();
  const dates: string[] = [];
  for (let i = 0; i < 7; i++){
    const d = new Date(today.getTime() - i*86400000);
    dates.push(dateStr(d));
  }
  const perDay = dates.map(date => ({ date, agg: loadDaily(date) }));
  const sum7 = summarize(perDay);

  const nowStr = dateStr(today);
  const outDayDir = path.resolve(process.cwd(), `reports/day-${nowStr}`);
  const outLatestDir = path.resolve(process.cwd(), 'reports/latest');
  const { year, week } = isoWeek(today);
  const outWeekDir = path.resolve(process.cwd(), `reports/week-${year}-${String(week).padStart(2,'0')}`);
  const trend = { generatedAt: new Date().toISOString(), range: { from: dates[dates.length-1], to: dates[0] }, totals: sum7, days: perDay };
  try { fs.mkdirSync(outDayDir, { recursive: true }); } catch {}
  try { fs.mkdirSync(outWeekDir, { recursive: true }); } catch {}
  try { fs.mkdirSync(outLatestDir, { recursive: true }); } catch {}
  fs.writeFileSync(path.join(outDayDir, 'trend-7d.json'), JSON.stringify(trend, null, 2));
  fs.writeFileSync(path.join(outWeekDir, 'weekly-summary.json'), JSON.stringify(trend, null, 2));
  fs.writeFileSync(path.join(outLatestDir, 'trend-weekly.json'), JSON.stringify(trend, null, 2));
  // eslint-disable-next-line no-console
  console.log(`[TREND] 7d totals: PnL=${sum7.PnL} Win%=${Math.round(sum7.winRate*100)} trades=${sum7.trades}`);
})();
