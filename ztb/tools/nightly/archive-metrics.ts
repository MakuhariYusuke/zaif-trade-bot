import * as fs from 'fs';
import * as path from 'path';
import { runOnceCollect } from '../metrics-dash';

function timestampDirRoot(): string | null {
  const root = path.resolve(process.cwd(), 'reports', 'nightly');
  if (!fs.existsSync(root)) return null;
  const dirs = fs.readdirSync(root).filter(d => /^(\d{8}-\d{4})$/.test(d)).sort();
  if (!dirs.length) return null;
  return path.join(root, dirs[dirs.length - 1]);
}

export function archiveMetrics() {
  // collect latest metrics from logs (default behavior of metrics-dash runOnceCollect)
  const res = runOnceCollect();
  const targetDir = timestampDirRoot();
  if (!targetDir) {
    throw new Error('No nightly plan directory found. Run archive-plan first.');
  }
  const outPath = path.join(targetDir, 'metrics.json');
  fs.writeFileSync(outPath, JSON.stringify(res, null, 2));
  try { console.log(JSON.stringify({ archived: true, dir: targetDir, metricsPath: outPath })); } catch {}
  return { dir: targetDir, metricsPath: outPath };
}

if (require.main === module) {
  try { archiveMetrics(); } catch (e: any) { console.error(e?.message || String(e)); process.exit(1); }
}
