import { Position, RiskConfig, evaluateExitConditions, manageTrailingStop, describeExit, calculateSma, calculateRsi, CoreRiskManager } from "../risk";
import { logInfo, logger } from "../../utils/logger";
import { logSignal } from "../../utils/trade-logger";
import { appendPriceSamples, getPriceSeries } from "../../utils/price-cache";
import { incTrailStop, incBuyEntry, incRsiExit, incTrailExit } from "../../utils/daily-stats";
import { logFeatureSample } from "../../utils/features-logger";
import { BaseStrategy } from "./base-strategy";

export interface StrategyContext {
  positions: Position[];
  positionsFile: string;
  currentPrice: number;
  trades: any[];
  nowMs: number;
  riskCfg: RiskConfig;
  pair?: string;
}

export class BuyStrategy extends BaseStrategy {
  protected applyRisk(decision: any, ctx: any){
    try { const r = this.risk.validateOrder(decision); if ((r as any)?.ok === false) this.logger?.warn?.('risk rejected', (r as any).error); } catch {}
    return decision;
  }
  protected async decide(ctx: StrategyContext): Promise<any> {
    // price series and indicators
    appendPriceSamples(ctx.trades.map((t:any)=> ({ ts:t.date? t.date*1000: ctx.nowMs, price:t.price })));
    const priceSeries=getPriceSeries(Math.max(ctx.riskCfg.smaPeriod,200));
    const base = (ctx.pair || '').split('_')[0]?.toUpperCase();
    const resolve = (baseKey: string, genericKey: string, fallbackKey: string, def: number) => {
      if (base) { const v = process.env[`${base}_${baseKey}`]; if (v != null) return Number(v); }
      if (process.env[genericKey] != null) return Number(process.env[genericKey]);
      if (process.env[fallbackKey] != null) return Number(process.env[fallbackKey]);
      return def;
    };
    const shortPeriod=resolve('BUY_SMA_SHORT','BUY_SMA_SHORT','SMA_SHORT',9);
    const longPeriod=resolve('BUY_SMA_LONG','BUY_SMA_LONG','SMA_LONG',26);
    const rsiPeriod=Number(process.env.RSI_PERIOD||14);
    const smaShort=calculateSma(priceSeries, shortPeriod);
    const smaLong=calculateSma(priceSeries, longPeriod);
    const rsi=calculateRsi(priceSeries, rsiPeriod);
    try { logFeatureSample({ ts: ctx.nowMs, pair: ctx.pair || 'unknown', side: 'bid', rsi: rsi ?? undefined, sma_short: smaShort ?? undefined, sma_long: smaLong ?? undefined, price: ctx.currentPrice, qty: 0 }); } catch {}

    // exits for short positions (BUY-back)
    const shortPositions=ctx.positions.filter((p:Position)=> p.side === 'short');
    const exits=evaluateExitConditions(shortPositions, ctx.currentPrice, calculateSma(priceSeries, ctx.riskCfg.smaPeriod), ctx.riskCfg, rsi);

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
        incTrailStop(d, ctx.pair);
        incTrailExit(d, ctx.pair);
      }
    }
    // Exit logs are handled in app/index.ts to avoid duplication

    // BUY モードのシグナル（例: デッドクロスからの反発条件）
    const haveLong=ctx.positions.some((p:Position)=> p.side==='long');
    const coreBuy=!!(smaShort && smaLong && smaShort > smaLong);
    const rsiTh = resolve('BUY_RSI_OVERSOLD','BUY_RSI_OVERSOLD','BUY_RSI_OVERSOLD',30);
    const rsiBuyBoost=!!(rsi && rsi <= rsiTh);
    const finalBuySignal=!haveLong && coreBuy && rsiBuyBoost;
    if (finalBuySignal){
      const msg = `[SIGNAL][mode=BUY] BUY entry condition met`;
      logInfo(msg);
      logSignal(msg);
      incBuyEntry(new Date().toISOString().slice(0,10), ctx.pair);
      try { logFeatureSample({ ts: Date.now(), pair: ctx.pair || 'unknown', side: 'bid', rsi: rsi ?? undefined, sma_short: smaShort ?? undefined, sma_long: smaLong ?? undefined, price: ctx.currentPrice, qty: 0 }); } catch {}
    }

    return { allExits };
  }
}

export function runBuyStrategy(ctx: StrategyContext) {
  const s = new BuyStrategy(logger, new CoreRiskManager());
  return s.runCycle(ctx);
}
