import { Position, RiskConfig, evaluateExitConditions, manageTrailingStop, describeExit, savePositionsToFile, calculateSma, calculateRsi } from "../risk";
import { logInfo } from "../../utils/logger";
import { logSignal } from "../../utils/trade-logger";
import { appendPriceSamples, getPriceSeries } from "../../utils/price-cache";
import { incSellEntry, incTrailStop, incRsiExit, incTrailExit } from "../../utils/daily-stats";
import { logFeatureSample } from "../../utils/features-logger";

export interface StrategyContext {
  positions: Position[];
  positionsFile: string;
  currentPrice: number;
  trades: any[];
  nowMs: number;
  riskCfg: RiskConfig;
  pair?: string;
}

export function runSellStrategy(ctx: StrategyContext) {
  // price series and indicators
  appendPriceSamples(ctx.trades.map((t:any)=> ({ ts:t.date? t.date*1000: ctx.nowMs, price:t.price })));
  const priceSeries=getPriceSeries(Math.max(ctx.riskCfg.smaPeriod,200));
  const base = (ctx.pair || '').split('_')[0]?.toUpperCase();
  const pickNum = (k: string, def: number) => { const v = process.env[k]; return v != null ? Number(v) : def; };
  const resolve = (baseKey: string, genericKey: string, fallbackKey: string, def: number) => {
    if (base) {
      const v = process.env[`${base}_${baseKey}`]; if (v != null) return Number(v);
    }
    if (process.env[genericKey] != null) return Number(process.env[genericKey]);
    if (process.env[fallbackKey] != null) return Number(process.env[fallbackKey]);
    return def;
  };
  const shortPeriod=resolve('SELL_SMA_SHORT','SELL_SMA_SHORT','SMA_SHORT',9);
  const longPeriod=resolve('SELL_SMA_LONG','SELL_SMA_LONG','SMA_LONG',26);
  const rsiPeriod=Number(process.env.RSI_PERIOD||14);
  const smaShort=calculateSma(priceSeries, shortPeriod);
  const smaLong=calculateSma(priceSeries, longPeriod);
  const rsi=calculateRsi(priceSeries, rsiPeriod);
  // Log a snapshot feature at evaluation tick (no PnL here)
  try { logFeatureSample({ ts: ctx.nowMs, pair: ctx.pair || 'unknown', side: 'ask', rsi: rsi ?? undefined, sma_short: smaShort ?? undefined, sma_long: smaLong ?? undefined, price: ctx.currentPrice, qty: 0 }); } catch {}

  // exits for long positions
  const longPositions=ctx.positions.filter((p:Position)=> p.side !== 'short');
  const exits=evaluateExitConditions(longPositions, ctx.currentPrice, calculateSma(priceSeries, ctx.riskCfg.smaPeriod), ctx.riskCfg, rsi);

  // trailing stop management
  const exitTrail: string[] = [];
  for (const pos of ctx.positions){
    const trail = manageTrailingStop(ctx.currentPrice, pos as any, ctx.nowMs);
    if (trail?.signal==='EXIT_TRAIL'){ exitTrail.push(pos.id); }
  }
  const allExits=[...exits];
  for (const x of exits){ if (x.reason==='RSI_EXIT') incRsiExit(new Date().toISOString().slice(0,10), ctx.pair); }
  if (exitTrail.length){
    const d=new Date().toISOString().slice(0,10);
    for(const id of exitTrail){
      const pos=ctx.positions.find(p=>p.id===id);
      if(pos) allExits.push({ position:pos, reason:'TRAIL_STOP', targetPrice:ctx.currentPrice });
  // Count both the stop event and the realized trail exit
  incTrailStop(d, ctx.pair);
  incTrailExit(d, ctx.pair);
    }
  }
  // Exit logs are handled in app/index.ts to avoid duplication

  // SELL-first entry heuristic
  const haveShort=ctx.positions.some((p:Position)=> p.side==='short');
  const coreSell=!!(smaShort && smaLong && smaShort < smaLong);
  const rsiTh = resolve('SELL_RSI_OVERBOUGHT','SELL_RSI_OVERBOUGHT','SELL_RSI_OVERBOUGHT',70);
  const rsiSellBoost=!!(rsi && rsi >= rsiTh);
  const finalSellSignal=!haveShort && coreSell && rsiSellBoost;
  if (finalSellSignal){
    const msg = `[SIGNAL][mode=SELL] SELL entry condition met`;
    logInfo(msg);
    logSignal(msg);
  incSellEntry(new Date().toISOString().slice(0,10), ctx.pair);
    try { logFeatureSample({ ts: Date.now(), pair: ctx.pair || 'unknown', side: 'ask', rsi: rsi ?? undefined, sma_short: smaShort ?? undefined, sma_long: smaLong ?? undefined, price: ctx.currentPrice, qty: 0 }); } catch {}
  }

  return { allExits };
}
