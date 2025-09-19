import * as fs from 'fs';
import * as path from 'path';

function findLatestNightlyDir(root = path.resolve(process.cwd(), 'reports', 'nightly')): string | null {
  try {
    const latest = path.join(root, 'latest');
    if (fs.existsSync(latest)) return latest;
    const dirs = fs.readdirSync(root).filter(d => /(\d{8}-\d{4})/.test(d)).sort();
    if (!dirs.length) return null;
    return path.join(root, dirs[dirs.length - 1]);
  } catch { return null; }
}

export function buildSlackSummary(dir?: string) {
  const target = dir || findLatestNightlyDir();
  if (!target) return 'nightly: no reports';
  const planPath = path.join(target, 'trade-plan.json');
  const metricsPath = path.join(target, 'metrics.json');
  let phase: any = '?'; let planned: any = '?'; let executed: any = '?'; let fail: any = '?'; let pnl: any = '?'; let guards: any = 0;
  try {
    if (fs.existsSync(planPath)) {
      const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
      phase = plan.phase; planned = plan.plannedOrders;
    }
  } catch {}
  try {
    if (fs.existsSync(metricsPath)) {
      const m = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
      executed = m.executedCount ?? executed;
      fail = m.failCount ?? fail;
      pnl = m.pnlTotal ?? pnl;
      guards = m.guardTrips ?? guards;
      if (m.tradePhase?.phase && phase === '?') phase = m.tradePhase.phase;
      if (m.plannedOrders && planned === '?') planned = m.plannedOrders;
    }
  } catch {}
  const warn = (guards && guards > 0) ? ' ⚠️' : '';
  return `Phase=${phase} Planned=${planned} Exec=${executed} Fail=${fail} PnL=${pnl}${warn}`;
}

if (require.main === module) {
  console.log(buildSlackSummary());
}
