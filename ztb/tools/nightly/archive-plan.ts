import * as fs from 'fs';
import * as path from 'path';
import { runTradeLive } from '../trade-live';
import * as child_process from 'child_process';

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
  // ml-summary (best-effort) from search results
  try {
    const mlTop = path.resolve(process.cwd(), 'ml-search-top.json');
    if (fs.existsSync(mlTop)) {
      const topRaw = fs.readFileSync(mlTop, 'utf8');
      // minimal summary extraction placeholder; keep original JSON as is
      fs.writeFileSync(path.join(baseDir, 'ml-summary.json'), topRaw);
    }
  } catch {}
  // update 'latest' symlink/copy directory
  try {
    const latestDir = path.resolve(process.cwd(), 'reports', 'nightly', 'latest');
    // remove existing latest
    if (fs.existsSync(latestDir)) {
      try {
        const st = fs.lstatSync(latestDir);
        if (st.isSymbolicLink()) fs.unlinkSync(latestDir);
        else {
          // remove directory recursively
          fs.rmSync(latestDir, { recursive: true, force: true });
        }
      } catch {}
    }
    try {
      fs.symlinkSync(baseDir, latestDir, 'dir');
    } catch {
      // fallback: copy contents
      fs.mkdirSync(latestDir, { recursive: true });
      for (const f of fs.readdirSync(baseDir)) {
        fs.copyFileSync(path.join(baseDir, f), path.join(latestDir, f));
      }
    }
  } catch {}
  // emit path summary for workflow usage
  try { console.log(JSON.stringify({ archived: true, dir: baseDir, planPath, plan })); } catch {}
  return { dir: baseDir, planPath, plan };
}

if (require.main === module) {
  archivePlan().catch(e => { console.error(e?.message || String(e)); process.exit(1); });
}
