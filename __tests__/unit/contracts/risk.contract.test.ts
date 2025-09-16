import { describe, test, expect, beforeEach } from 'vitest';
import { CoreRiskManager } from '../../../src/core/risk';
import { createServiceRiskManager } from '../../../src/adapters/risk-service';
import type { RiskManager } from '@contracts';

function cases(): [string, () => RiskManager][] {
  return [
    ['core', () => new CoreRiskManager()],
    ['services-adapter', () => createServiceRiskManager()],
  ];
}

beforeEach(() => {
  // reset env boundaries to stable defaults
  for (const [k, v] of Object.entries({
    RISK_MIN_TRADE_SIZE: '0.001',
    MAX_ORDER_NOTIONAL_JPY: '100000',
    RISK_TRAIL_TRIGGER_PCT: '0.01',
    RISK_TRAIL_STOP_PCT: '0.005',
    RISK_POSITION_PCT: '0.1',
  })) { (process.env as any)[k] = v; }
});

describe('RiskManager contract', () => {
  test.each(cases())('%s: validateOrder returns error for invalid intent', (_n, make) => {
    const rm = make();
    const r1 = rm.validateOrder({ qty: 0, price: 100 });
    expect(r1.ok).toBe(false);
    const r2 = rm.validateOrder({ qty: 0.0005, price: 100 });
    expect(r2.ok).toBe(false);
    const r3 = rm.validateOrder({ qty: 2000, price: 100 }); // notional too high
    expect(r3.ok).toBe(false);
    const r4 = rm.validateOrder({ qty: 1, price: 100 });
    expect(r4.ok).toBe(true);
  });

  test.each(cases())('%s: clampExposure clamps qty by positionPct', (_n, make) => {
    const rm = make();
    const out = rm.clampExposure({ jpy: 100000 }, { side: 'buy', price: 1000000, qty: 1 });
    // positionPct=0.1 -> alloc=10000 -> qty=0.01
    expect(out.qty).toBeCloseTo(0.01, 8);
  });

  test.each(cases())('%s: manageTrailingStop produces trigger', (_n, make) => {
    const rm = make();
    const state: any = { side: 'long', entryPrice: 100, trailArmed: true, highestPrice: 110 };
    const act = rm.manageTrailingStop(state, 112);
    expect(act).not.toBeNull();
    expect(act!.trigger).toBeGreaterThan(0);
  });
});
