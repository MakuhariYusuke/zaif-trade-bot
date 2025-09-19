import { runOnceCollect } from './metrics-dash';
import { loadTradeConfig } from '../config/trade-config';
import { loadTradeState } from '../config/trade-state';
import * as fs from 'fs';
import * as path from 'path';

function parseArgs(argv: string[]) {
  return { json: argv.includes('--json') };
}

function readCoverageSummary(): { statements?: string; lines?: string } {
  try {
    if (fs.existsSync('coverage-merged/summary.json')) {
      return JSON.parse(fs.readFileSync('coverage-merged/summary.json','utf8'));
    }
    // fallback: pick any vitest-report json to derive simple stmt pct if present (not implemented)
  } catch {}
  return {};
}

function readLastSysMetrics(): any | undefined {
  // Search logs directory for latest file containing SYS metrics JSON lines
  try {
    const root = process.cwd();
    const logsDir = path.join(root, 'logs');
    if (!fs.existsSync(logsDir)) return undefined;
    const files = fs.readdirSync(logsDir).filter(f=>f.endsWith('.log')).map(f=>path.join(logsDir,f));
    let latest: string|undefined; let mt=0;
    for (const f of files){ const st=fs.statSync(f); if (st.mtimeMs>mt){ mt=st.mtimeMs; latest=f; } }
    if (!latest) return undefined;
    const content = fs.readFileSync(latest,'utf8').trim().split(/\r?\n/).slice(-500).reverse();
    for (const line of content){
      if (line.includes('"category":"SYS"') && line.includes('"metrics"')) {
        try { const obj = JSON.parse(line); if (obj?.data && obj.data[0]) return obj.data[0]; } catch {}
      }
    }
  } catch {}
  return undefined;
}

export function buildSlackSummary(){
  const cfg = loadTradeConfig();
  const st = loadTradeState();
  const collect = runOnceCollect();
  const cov = readCoverageSummary();
  const sha = process.env.GITHUB_SHA?.slice(0,7) || process.env.COMMIT_SHA?.slice(0,7) || '';
  const sys = readLastSysMetrics();
  const lines: string[] = [];
  lines.push(`pair: ${cfg.pair}`);
  if (st.phase) lines.push(`phase: ${st.phase}`);
  if (collect?.plannedOrders != null) lines.push(`planned: ${collect.plannedOrders}`);
  lines.push(`executed: ${collect?.executedCount ?? 0}`);
  lines.push(`fails: ${collect?.failCount ?? 0}`);
  if (typeof collect?.pnlTotal === 'number') lines.push(`pnl: ${collect.pnlTotal.toFixed(4)}`);
  if (collect?.guardTrips) lines.push(`guards: ${collect.guardTrips}`);
  if (collect?.phaseEscalations) lines.push(`esc: ${collect.phaseEscalations}`);
  if (collect?.phaseDowngrades) lines.push(`down: ${collect.phaseDowngrades}`);
  if (cov.statements) lines.push(`cov: ${cov.statements}%`);
  if (sha) lines.push(`sha:${sha}`);
  if (sys) {
    if (typeof sys.rssMb === 'number') lines.push(`rss:${sys.rssMb}MB`);
    if (typeof sys.loopP95 === 'number') lines.push(`loopP95:${sys.loopP95}ms`);
  }
  const text = lines.join(' | ');
  const payload = { text: `Trade Summary: ${text}`, meta: { coverage: cov, commit: sha, metrics: collect, sys } };
  return { payload, text };
}

async function main(){
  const { payload, text } = buildSlackSummary();
  const args = parseArgs(process.argv);
  if (args.json) {
    console.log(JSON.stringify(payload));
    return;
  }
  if (process.env.SLACK_WEBHOOK_URL) {
    try {
      const res = await fetch(process.env.SLACK_WEBHOOK_URL, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      if (!res.ok) console.error('Slack POST failed', res.status);
    } catch (e:any){ console.error('Slack error', e?.message || String(e)); }
  } else {
    console.log(text);
  }
}

if(require.main===module){
  main();
}
