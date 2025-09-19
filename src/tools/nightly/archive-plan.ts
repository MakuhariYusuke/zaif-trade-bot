import * as fs from 'fs';
import * as path from 'path';
import { runTradeLive } from '../trade-live';

function timestamp() {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const yyyy = d.getFullYear();
  const MM = pad(d.getMonth() + 1);
  const dd = pad(d.getDate());
  const hh = pad(d.getHours());
  const mm = pad(d.getMinutes());
  return `${yyyy}${MM}${dd}-${hh}${mm}`;
}

export async function archivePlan() {
  const stamp = timestamp();
  const baseDir = path.resolve(process.cwd(), 'reports', 'nightly', stamp);
  fs.mkdirSync(baseDir, { recursive: true });
  const plan = await runTradeLive({ dryRun: true });
  const planPath = path.join(baseDir, 'trade-plan.json');
  fs.writeFileSync(planPath, JSON.stringify(plan, null, 2));
  // copy current trade-state.json (if exists)
  const stateFile = process.env.TRADE_STATE_FILE || 'trade-state.json';
  if (fs.existsSync(stateFile)) {
    try {
      fs.copyFileSync(stateFile, path.join(baseDir, 'trade-state.json'));
    } catch {}
  }
  // emit path summary for workflow usage
  try { console.log(JSON.stringify({ archived: true, dir: baseDir, planPath, plan })); } catch {}
  return { dir: baseDir, planPath, plan };
}

if (require.main === module) {
  archivePlan().catch(e => { console.error(e?.message || String(e)); process.exit(1); });
}
