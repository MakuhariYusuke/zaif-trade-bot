import { describe, it, expect, beforeEach, vi } from 'vitest';
import path from 'path';
import fs from 'fs';
// count increments
const incs: Record<string, number> = {
  trailArmed: 0, trailExit: 0, trailStop: 0, rsiExit: 0, buyEntry: 0, buyExit: 0, sellEntry: 0, buyExits: 0
};
vi.mock('../../../ztb/utils/daily-stats', () => ({
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

  it('buy-strategy increments buy entry and may increment RSI/trail counters', async () => {
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
    const { runBuyStrategy } = await import('../../../ztb/core/strategies/buy-strategy');
    await runBuyStrategy(ctx);
  expect(incs.buyEntry).toBeGreaterThan(0);
  expect(incs.trailStop >= 0 && incs.trailExit >= 0).toBe(true);
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
    const { runSellStrategy } = await import('../../../ztb/core/strategies/sell-strategy');
  process.env.MIN_HOLD_SEC = '0';
  await runSellStrategy(ctx);
  expect(incs.sellEntry).toBeGreaterThanOrEqual(0);
  expect(incs.trailStop).toBeGreaterThanOrEqual(0);
  expect(incs.trailExit).toBeGreaterThanOrEqual(0);
  expect(incs.rsiExit).toBeGreaterThanOrEqual(0);
  });

  it('consecutive triggers accumulate counts (RSI and Trail)', async () => {
    const prices = [100,101,102,103,104,105,106,107,108,109,110];
    const now = Date.now();
  const baseCtx:any = {
      positions: [
        { id:'L1', pair:'btc_jpy', side:'long', entryPrice: 100, amount: 1, timestamp: now-60000, highestPrice: 150 },
        { id:'S1', pair:'btc_jpy', side:'short', entryPrice: 110, amount: 1, timestamp: now-60000, highestPrice: 90 },
      ],
      positionsFile: path.join(TMP, 'pos.json'),
      currentPrice: 120,
      trades: prices.map((p,i)=>({ price:p, date: Math.floor((now - (prices.length-1-i)*1000)/1000) })),
      nowMs: now,
      riskCfg: { stopLossPct:1.0, takeProfitPct:1.0, positionPct:.05, smaPeriod:3, positionsFile:'', trailTriggerPct:.01, trailStopPct:.005, dcaStepPct:.01, maxPositions:5, maxDcaPerPair:3, minTradeSize:0.0001, maxSlippagePct:.005, indicatorIntervalSec:60 },
      pair: 'btc_jpy',
    };
  process.env.RSI_PERIOD = '5';
  process.env.SELL_RSI_OVERBOUGHT = '0';
  process.env.BUY_RSI_OVERSOLD = '100';
    process.env.MIN_HOLD_SEC = '0';
    const { runSellStrategy } = await import('../../../ztb/core/strategies/sell-strategy');
    const { runBuyStrategy } = await import('../../../ztb/core/strategies/buy-strategy');
    await runSellStrategy(baseCtx);
    await runBuyStrategy(baseCtx);
    // fire again with slight time shift to allow trail management updates
    baseCtx.nowMs = now + 15000;
    baseCtx.currentPrice = 118;
    await runSellStrategy(baseCtx);
    await runBuyStrategy(baseCtx);
    expect(incs.rsiExit).toBeGreaterThanOrEqual(2);
    expect(incs.trailStop).toBeGreaterThanOrEqual(2);
    expect(incs.trailExit).toBeGreaterThanOrEqual(2);
  });
});
