import { describe, it, expect, beforeEach } from 'vitest';
import * as fs from 'fs';
import { runTradeLive } from '../../../ztb/tools/trade-live';
import { loadTradeConfig } from '../../../ztb/config/trade-config';

function writeTradeConfig(extra: any){
  const base = loadTradeConfig();
  const merged = { ...base, ...extra };
  fs.writeFileSync('trade-config.json', JSON.stringify(merged, null, 2));
}

describe('trade-live guards', () => {
  beforeEach(() => {
    // reset state
    fs.writeFileSync('trade-state.json', JSON.stringify({ phase: 1, consecutiveDays: 0, totalSuccess: 0, lastDate: null }));
  });

  it('respects maxOrdersPerDay guard', async () => {
    writeTradeConfig({ phase:1, phaseSteps:[{phase:1, ordersPerDay:5}], maxOrdersPerDay:2 });
    const res = await runTradeLive({ daySuccess:0 });
    expect(res.executed).toBeLessThanOrEqual(2);
  });

  it('respects maxLossPerDay guard (simulated negative pnl forces stop)', async () => {
    // We approximate by setting very low maxLossPerDay so first negative pnl stops.
    writeTradeConfig({ phase:1, phaseSteps:[{phase:1, ordersPerDay:5}], maxLossPerDay:1 });
    const res = await runTradeLive({ daySuccess:0 });
    // executed may be 0 or 1 depending on random pnl; but never full 5
    expect(res.executed).toBeLessThan(5);
  });

  it('slippageGuardPct rejects large deviation', async () => {
    writeTradeConfig({ phase:1, phaseSteps:[{phase:1, ordersPerDay:5}], slippageGuardPct:0.0000001 }); // tiny threshold to force rejection
    const res = await runTradeLive({ daySuccess:0 });
    // with extremely small threshold practically all simulated orders rejected
    expect(res.executed).toBe(0);
    expect(res.attempts).toBeGreaterThan(0);
  });
});
