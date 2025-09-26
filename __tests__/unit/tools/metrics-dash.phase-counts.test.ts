import { describe, it, expect } from 'vitest';
import { runOnceCollect } from '../../../ztb/tools/metrics-dash';
import * as fs from 'fs';
import * as path from 'path';

describe('metrics-dash phase escalation/downgrade counts', () => {
  it('counts phaseEscalations and phaseDowngrades from EVENT/TRADE_PHASE logs', () => {
    const tmp = path.join(process.cwd(), 'tmp-test-dash-phase');
    fs.mkdirSync(tmp, { recursive: true });
    const logPath = path.join(tmp, 'test.log');
    // Craft JSONL lines with EVENT metrics flush (minimal) and published events
    const lines = [
      JSON.stringify({ ts: new Date().toISOString(), level:'INFO', category:'EVENT', message:'published', data:[{ type:'EVENT/TRADE_PHASE', fromPhase:1, toPhase:2, reason:'promotion' }] }),
      JSON.stringify({ ts: new Date().toISOString(), level:'INFO', category:'EVENT', message:'published', data:[{ type:'EVENT/TRADE_PHASE', fromPhase:2, toPhase:1, reason:'downgrade' }] }),
      JSON.stringify({ ts: new Date().toISOString(), level:'INFO', category:'EVENT', message:'metrics', data:[{ windowMs:1000, types:{} }] }),
    ];
    fs.writeFileSync(logPath, lines.join('\n')+'\n','utf8');
    process.env.METRICS_LOG = logPath;
    const res = runOnceCollect();
    expect(res.phaseEscalations).toBe(1);
    expect(res.phaseDowngrades).toBe(1);
  });
});
