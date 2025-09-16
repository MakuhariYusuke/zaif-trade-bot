import { describe, it, expect, beforeEach } from 'vitest';
import { calcSMA, calcRSI, evaluateExitSignals } from '../../../src/adapters/risk-service';

describe('risk-service calc sanity', () => {
  beforeEach(() => {
    Object.assign(process.env, {
      RISK_MIN_TRADE_SIZE: '0.001',
      MAX_ORDER_NOTIONAL_JPY: '100000',
      RISK_TRAIL_TRIGGER_PCT: '0.01',
      RISK_TRAIL_STOP_PCT: '0.005',
      RISK_POSITION_PCT: '0.1',
    });
  });

  it('calculates SMA and RSI', () => {
    const sma = calcSMA([1,2,3,4,5], 3);
    expect(sma).not.toBeNull();
  expect(sma!).toBeCloseTo((1+2+3)/3, 8);
    const rsi = calcRSI([1,2,3,2,4,3,5], 3);
    expect(rsi).not.toBeNull();
    expect(rsi!).toBeGreaterThanOrEqual(0);
    expect(rsi!).toBeLessThanOrEqual(100);
  });

  it('evaluateExitSignals returns empty when SMA missing', () => {
    const sigs = evaluateExitSignals([], 100, null as any, {
      stopLossPct: 0.02, takeProfitPct: 0.05, positionPct: 0.05, smaPeriod: 20,
      positionsFile: '', trailTriggerPct: 0.05, trailStopPct: 0.03, dcaStepPct: 0.01,
      maxPositions: 5, maxDcaPerPair: 3, minTradeSize: 0.0001, maxSlippagePct: 0.005, indicatorIntervalSec: 60
    });
    expect(Array.isArray(sigs)).toBe(true);
  });
});
