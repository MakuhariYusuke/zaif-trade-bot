import { describe, it, expect } from 'vitest';
import { evaluateExitConditions, Position } from '../../../src/core/risk';

describe('max hold guard (core)', () => {
  it('emits TIME_LIMIT when hold exceeds MAX_HOLD_SEC', () => {
    const now = Date.now();
    const pos: Position = { id: 'p1', pair: 'btc_jpy', side: 'long', entryPrice: 100, amount: 0.01, timestamp: now - 61_000 } as any;
    const old = process.env.MAX_HOLD_SEC;
    process.env.MAX_HOLD_SEC = '60';
    const sigs = evaluateExitConditions([pos], 110, null as any, {
      stopLossPct: 0.02, takeProfitPct: 0.05, positionPct: 0.05, smaPeriod: 20,
      positionsFile: '', trailTriggerPct: 0.05, trailStopPct: 0.03, dcaStepPct: 0.01,
      maxPositions: 5, maxDcaPerPair: 3, minTradeSize: 0.0001, maxSlippagePct: 0.005, indicatorIntervalSec: 60
    }, 50);
    process.env.MAX_HOLD_SEC = old;
    expect(sigs.some(s=> s.reason==='TIME_LIMIT')).toBe(true);
  });
});
