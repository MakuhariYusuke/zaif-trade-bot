import { describe, it, expect, beforeEach } from 'vitest';
import { calculateSma, calculateRsi, positionSizeFromBalance, evaluateExitConditions, getRiskConfig, Position } from '../../../src/core/risk';

describe('core/risk', ()=>{
  beforeEach(()=>{ process.env = { ...process.env, RISK_POSITION_PCT: '0.1' }; });

  it('calculateSma returns null when not enough data', ()=>{
    expect(calculateSma([1,2,3], 5)).toBeNull();
  });
  it('calculateRsi returns 50 when gains=losses=0', ()=>{
    // flat prices
    const prices = Array.from({length: 15}, (_,i)=>100);
    expect(calculateRsi(prices, 14)).toBe(50);
  });
  it('positionSizeFromBalance applies pct and rounds', ()=>{
    const cfg = getRiskConfig();
    const size = positionSizeFromBalance(100000, 100, cfg); // 10% of 100k = 10k / 100 = 100
    expect(size).toBeCloseTo(100, 8);
  });
  it('evaluateExitConditions yields RSI_EXIT when over threshold', ()=>{
    const pos: Position = { id:'1', pair:'btc_jpy', side:'long', entryPrice:100, amount:1, timestamp:Date.now() };
    process.env.SELL_RSI_OVERBOUGHT = '60';
    // For long, RSI_EXIT triggers regardless of SMA/TP/SL when rsi >= th
    const sigs = evaluateExitConditions([pos], 100, null, getRiskConfig(), 70);
    expect(sigs.some(s=>s.reason==='RSI_EXIT')).toBe(true);
  });
  it('evaluateExitConditions yields empty when no condition', ()=>{
    const pos: Position = { id:'1', pair:'btc_jpy', side:'long', entryPrice:100, amount:1, timestamp:Date.now() };
    const sigs = evaluateExitConditions([pos], 101, null, getRiskConfig(), 50);
    expect(sigs.length).toBe(0);
  });
});
