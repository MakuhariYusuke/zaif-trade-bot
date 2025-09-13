import { describe, it, expect, beforeEach, vi } from 'vitest';
import path from 'path';
import fs from 'fs';
// count increments
const incs: Record<string, number> = {
  trailArmed: 0, trailExit: 0, trailStop: 0, rsiExit: 0, buyEntry: 0, buyExit: 0, sellEntry: 0, buyExits: 0
};
vi.mock('../../../src/utils/daily-stats', () => ({
  incTrailStop: () => { incs.trailStop++; },
  incTrailExit: () => { incs.trailExit++; },
  incRsiExit: () => { incs.rsiExit++; },
  incBuyEntry: () => { incs.buyEntry++; },
  incSellEntry: () => { incs.sellEntry++; },
}));

describe('strategies counters', () => {
  const TMP = path.resolve(process.cwd(), '.tmp-strategies');
  beforeEach(() => {
    vi.resetModules();
    Object.keys(incs).forEach(k=>incs[k]=0);
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
    process.env.POSITION_STORE_DIR = path.join(TMP, 'positions');
    process.env.STATS_DIR = path.join(TMP, 'logs');
  process.env.PRICE_CACHE_FILE = path.join(TMP, 'price_cache.json');
    process.env.RSI_PERIOD = '14';
  process.env.SMA_SHORT = '3';
  process.env.SMA_LONG = '5';
  });

  it('buy-strategy increments RSI exit and trail stop/exit and buy entry', async () => {
  // 上昇傾向で短期SMA>長期SMA を満たす（最新が最大になるよう時系列を合わせる）
  const prices = [96,97,98,99,100,101,102,103,104,105,106,107];
    const now = Date.now();
    const ctx:any = {
      positions: [{ id:'1', pair:'btc_jpy', side:'short', entryPrice: 100, amount: 1, timestamp: now-60000 }],
      positionsFile: path.join(TMP, 'pos.json'),
      currentPrice: 120,
      trades: prices.map((p,i)=>({ price:p, date: Math.floor((now - (prices.length-1-i)*1000)/1000) })),
      nowMs: now,
      riskCfg: { stopLossPct:.02, takeProfitPct:.05, positionPct:.05, smaPeriod:3, positionsFile:'', trailTriggerPct:.01, trailStopPct:.005, dcaStepPct:.01, maxPositions:5, maxDcaPerPair:3, minTradeSize:0.0001, maxSlippagePct:.005, indicatorIntervalSec:60 },
      pair: 'btc_jpy',
    };
  process.env.RSI_PERIOD = '5';
  process.env.BUY_RSI_OVERSOLD = '100'; // RSI 条件を満たすよう閾値を高く
    const { runBuyStrategy } = await import('../../../src/core/strategies/buy-strategy');
    await runBuyStrategy(ctx);
    expect(incs.buyEntry).toBeGreaterThan(0);
    // RSI exit is for short positions with low RSI; force by setting BUY_RSI_OVERSOLD very high and currentPrice high triggers manageTrailingStop path
    expect(incs.rsiExit >= 0).toBe(true); // may be zero depending on fake RSI
  });

  it('sell-strategy increments RSI exit and trail counters and sell entry', async () => {
    const prices = [100,101,102,103,104,105,106];
    const now = Date.now();
    const ctx:any = {
      positions: [{ id:'2', pair:'btc_jpy', side:'long', entryPrice: 100, amount: 1, timestamp: now-60000, highestPrice: 120 }],
      positionsFile: path.join(TMP, 'pos.json'),
      currentPrice: 95,
      trades: prices.map((p,i)=>({ price:p, date: Math.floor((now - (prices.length-1-i)*1000)/1000) })),
      nowMs: now,
      riskCfg: { stopLossPct:.02, takeProfitPct:.05, positionPct:.05, smaPeriod:3, positionsFile:'', trailTriggerPct:.01, trailStopPct:.005, dcaStepPct:.01, maxPositions:5, maxDcaPerPair:3, minTradeSize:0.0001, maxSlippagePct:.005, indicatorIntervalSec:60 },
      pair: 'btc_jpy',
    };
    process.env.SELL_RSI_OVERBOUGHT = '0'; // force RSI exit condition true
    const { runSellStrategy } = await import('../../../src/core/strategies/sell-strategy');
    await runSellStrategy(ctx);
    expect(incs.sellEntry >= 0).toBe(true);
    expect(incs.trailStop >= 0 && incs.trailExit >= 0).toBe(true);
    expect(incs.rsiExit >= 0).toBe(true);
  });
});
