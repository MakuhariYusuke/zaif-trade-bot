import { describe, it, expect } from 'vitest';
import { calcRealizedPnL } from '../../../ztb/core/market';

describe('core/market', () => {
  it('calcRealizedPnL matches FIFO and sums correctly', ()=>{
    const hist = [
      { trade_type:'bid', price:100, amount:2 },
      { trade_type:'bid', price:120, amount:1 },
      { trade_type:'ask', price:130, amount:2 }, // consumes 2 @ 100 -> 2*30
      { trade_type:'ask', price:110, amount:1 }, // consumes 1 @ 120 -> 1*(-10)
    ];
    const r = calcRealizedPnL(hist);
    expect(r.realized).toBeCloseTo(50, 10);
    expect(r.trades).toBe(4);
  });
});
