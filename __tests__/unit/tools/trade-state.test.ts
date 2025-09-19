import { describe, it, expect } from 'vitest';
import { DEFAULT_STATE, applyPhaseProgress, DEFAULT_RULES } from '../../../src/config/trade-state';

describe('trade-state promotion', () => {
  it('promotes 1->2 after consecutive days', () => {
    let s = { ...DEFAULT_STATE };
    const rules = { ...DEFAULT_RULES, to2_consecutiveDays: 2 };
    ({ state: s } = applyPhaseProgress(s, { date: '2025-01-01', daySuccess: 1 }, rules));
    const res = applyPhaseProgress(s, { date: '2025-01-02', daySuccess: 1 }, rules);
    expect(res.state.phase).toBe(2);
    expect(res.promotion?.fromPhase).toBe(1);
    expect(res.promotion?.toPhase).toBe(2);
  });
  it('promotes 2->3 and 3->4 by total success', () => {
    let s = { ...DEFAULT_STATE, phase: 2 };
    const rules = { ...DEFAULT_RULES, to3_totalSuccess: 3, to4_totalSuccess: 5 };
    ({ state: s } = applyPhaseProgress(s, { date: '2025-01-01', daySuccess: 2 }, rules));
    let r = applyPhaseProgress(s, { date: '2025-01-02', daySuccess: 1 }, rules);
    expect(r.state.phase).toBe(3);
    ({ state: s } = r);
    r = applyPhaseProgress(s, { date: '2025-01-03', daySuccess: 2 }, rules);
    expect(r.state.phase).toBe(4);
  });
});
