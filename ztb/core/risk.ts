import { getMaxHoldSec } from "../utils/toolkit";
import { updatePositionFields } from "./position-store";
import type { RiskManager, TrailingAction, ClampedIntent } from "@contracts";
import { ok, err } from "../utils/result";

export interface RiskConfig { stopLossPct: number; takeProfitPct: number; positionPct: number; smaPeriod: number; positionsFile: string; trailTriggerPct: number; trailStopPct: number; dcaStepPct: number; maxPositions: number; maxDcaPerPair: number; minTradeSize: number; maxSlippagePct: number; indicatorIntervalSec: number; }
export type PositionSide = "long" | "short";
export interface Position { id: string; pair: string; side: PositionSide; entryPrice: number; amount: number; timestamp: number; highestPrice?: number; dcaCount?: number; openOrderIds?: number[]; }
export interface ExitSignal { position: Position; reason: "STOP_LOSS" | "TAKE_PROFIT" | "MA_BREAK" | "TRAIL_STOP" | "RSI_EXIT" | "TIME_LIMIT"; targetPrice: number; }
export interface TrailResult { signal?: "EXIT_TRAIL"; reason?: any }

/**
 * Manage trailing stop for a position.
 * @param {number} lastPrice The last traded price.
 * @param {Position} position The position to manage.
 * @param {number} now The current time.
 * @returns {TrailResult|undefined} A trailing stop result or undefined.
 */
export function manageTrailingStop(lastPrice: number, position: Position, now: number): TrailResult | undefined {
    const TRIGGER_PCT = Number(process.env.RISK_TRAIL_TRIGGER_PCT || 0.01);
    const TRAIL_PCT = Number(process.env.RISK_TRAIL_STOP_PCT || 0.005);
    const MIN_HOLD_SEC = Number(process.env.MIN_HOLD_SEC || 30);
    const COOLDOWN_SEC = Number(process.env.COOLDOWN_SEC || 10);
    const holdMs = now - position.timestamp; let hasChanged = false;
    if (!(position as any).trailArmed) {
        if (position.entryPrice > 0 && (lastPrice / position.entryPrice - 1) >= TRIGGER_PCT) {
            (position as any).trailArmed = true;
            (position as any).highestPrice = lastPrice; hasChanged = true;
        }
    } else {
        const hp = (position as any).highestPrice || lastPrice;
        if (lastPrice > hp) {
            (position as any).highestPrice = lastPrice;
            hasChanged = true;
        }
        const trailStop = (position as any).highestPrice * (1 - TRAIL_PCT);
        (position as any).trailStop = trailStop;
        if (!(position as any).lastTrailAt || now - (position as any).lastTrailAt > COOLDOWN_SEC * 1000) {
            (position as any).lastTrailAt = now;
            hasChanged = true;
        }
        if (holdMs >= MIN_HOLD_SEC * 1000 && lastPrice <= trailStop) { return { signal: "EXIT_TRAIL", reason: { highestPrice: (position as any).highestPrice, trailStop } }; }
    }
    if (hasChanged) { updatePositionFields(position.pair, { trailArmed: (position as any).trailArmed, trailStop: (position as any).trailStop, highestPrice: (position as any).highestPrice, lastTrailAt: now }); }
    return undefined;
}

/**
 * Get the risk management configuration.
 * @returns {RiskConfig} The risk management configuration.
 */
export function getRiskConfig(): RiskConfig {
    return {
        stopLossPct: Number(process.env.RISK_STOP_LOSS_PCT || 0.02),
        takeProfitPct: Number(process.env.RISK_TAKE_PROFIT_PCT || 0.05),
        positionPct: Number(process.env.RISK_POSITION_PCT || 0.05),
        smaPeriod: Number(process.env.RISK_SMA_PERIOD || 20),
        positionsFile: String(process.env.POSITIONS_FILE || ".positions.json"),
        trailTriggerPct: Number(process.env.RISK_TRAIL_TRIGGER_PCT || 0.05),
        trailStopPct: Number(process.env.RISK_TRAIL_STOP_PCT || 0.03),
        dcaStepPct: Number(process.env.RISK_DCA_STEP_PCT || 0.01),
        maxPositions: Number(process.env.RISK_MAX_POSITIONS || 5),
        maxDcaPerPair: Number(process.env.RISK_MAX_DCA_PER_PAIR || 3),
        minTradeSize: Number(process.env.RISK_MIN_TRADE_SIZE || 0.0001),
        maxSlippagePct: Number(process.env.RISK_MAX_SLIPPAGE_PCT || 0.005),
        indicatorIntervalSec: Number(process.env.RISK_INDICATOR_INTERVAL_SEC || 60)
    };
}
// IO helpers moved to adapters/risk-config
export function calculateSma(prices: number[], period: number): number | null {
    if (prices.length < period) return null;
    const slice = prices.slice(0, period);
    const sum = slice.reduce((a, b) => a + b, 0);
    return sum / slice.length;
}
export function calculateRsi(prices: number[], period = 14): number | null {
    if (prices.length < period + 1) return null;
    let gains = 0, losses = 0;
    for (let i = 1; i <= period; i++) {
        const diff = prices[i - 1] - prices[i];
        if (diff > 0) gains += diff;
        else losses -= diff;
    }
    if (gains === 0 && losses === 0) return 50;
    const rs = losses === 0 ? 100 : gains / (losses || 1e-9);
    return 100 - (100 / (1 + rs));
}
export function positionSizeFromBalance(jpyBalance: number, price: number, cfg: RiskConfig): number {
    if (price <= 0) return 0;
    const alloc = jpyBalance * cfg.positionPct;
    const amount = alloc / price;
    return Math.floor(amount * 1e8) / 1e8;
}
export function evaluateExitConditions(positions: Position[], currentPrice: number, sma: number | null, cfg: RiskConfig, rsi?: number | null): ExitSignal[] {
    const signals: ExitSignal[] = [];
    const maxHold = getMaxHoldSec();
    function resolveRsiOver(pair: string): number {
        const base = pair.split('_')[0]?.toUpperCase();
        const key = `${base}_SELL_RSI_OVERBOUGHT`;
        if (base && process.env[key] != null) return Number(process.env[key]);
        if (process.env.SELL_RSI_OVERBOUGHT != null) return Number(process.env.SELL_RSI_OVERBOUGHT);
        return 70;
    }
    function resolveRsiUnder(pair: string): number {
        const base = pair.split('_')[0]?.toUpperCase();
        const key = `${base}_BUY_RSI_OVERSOLD`;
        if (base && process.env[key] != null) return Number(process.env[key]);
        if (process.env.BUY_RSI_OVERSOLD != null) return Number(process.env.BUY_RSI_OVERSOLD);
        return 30;
    }
    for (const pos of positions) {
        if (maxHold != null && (Date.now() - pos.timestamp) >= maxHold * 1000) {
            signals.push({ position: pos, reason: 'TIME_LIMIT', targetPrice: currentPrice });
            continue;
        }
        if (pos.side === 'long') {
            if (pos.highestPrice == null || currentPrice > pos.highestPrice) pos.highestPrice = currentPrice;
            if (typeof pos.highestPrice === 'number' && pos.highestPrice >= pos.entryPrice * (1 + cfg.trailTriggerPct)) {
                const trailStopLine = pos.highestPrice * (1 - cfg.trailStopPct);
                if (currentPrice <= trailStopLine && currentPrice > pos.entryPrice) {
                    signals.push({ position: pos, reason: 'TRAIL_STOP', targetPrice: currentPrice });
                    continue;
                }
            }
            const stopLossPrice = pos.entryPrice * (1 - cfg.stopLossPct);
            if (currentPrice <= stopLossPrice) {
                signals.push({ position: pos, reason: 'STOP_LOSS', targetPrice: currentPrice });
                continue;
            }
            const takeProfitPrice = pos.entryPrice * (1 + cfg.takeProfitPct);
            if (currentPrice >= takeProfitPrice) {
                signals.push({ position: pos, reason: 'TAKE_PROFIT', targetPrice: currentPrice });
                continue;
            }
            if (sma !== null && currentPrice < sma && currentPrice > pos.entryPrice) {
                signals.push({ position: pos, reason: 'MA_BREAK', targetPrice: currentPrice });
                continue;
            }
            if (rsi != null && rsi >= resolveRsiOver(pos.pair)) {
                signals.push({ position: pos, reason: 'RSI_EXIT', targetPrice: currentPrice });
                continue;
            }
        } else if (pos.side === 'short') {
            if (pos.highestPrice == null || currentPrice < pos.highestPrice) pos.highestPrice = currentPrice;
            if (typeof pos.highestPrice === 'number' && pos.highestPrice <= pos.entryPrice * (1 - cfg.trailTriggerPct)) {
                const trailStopLine = pos.highestPrice * (1 + cfg.trailStopPct);
                if (currentPrice >= trailStopLine && currentPrice < pos.entryPrice) {
                    signals.push({ position: pos, reason: 'TRAIL_STOP', targetPrice: currentPrice });
                    continue;
                }
            }
            const stopLossPrice = pos.entryPrice * (1 + cfg.stopLossPct);
            if (currentPrice >= stopLossPrice) {
                signals.push({ position: pos, reason: 'STOP_LOSS', targetPrice: currentPrice });
                continue;
            }
            const takeProfitPrice = pos.entryPrice * (1 - cfg.takeProfitPct);
            if (currentPrice <= takeProfitPrice) {
                signals.push({ position: pos, reason: 'TAKE_PROFIT', targetPrice: currentPrice });
                continue;
            }
            if (sma !== null && currentPrice > sma && currentPrice < pos.entryPrice) {
                signals.push({ position: pos, reason: 'MA_BREAK', targetPrice: currentPrice });
                continue;
            }
            if (rsi != null && rsi <= resolveRsiUnder(pos.pair)) {
                signals.push({ position: pos, reason: 'RSI_EXIT', targetPrice: currentPrice });
                continue;
            }
        }
    }
    return signals;
}
export function describeExit(signal: ExitSignal): string {
    return `exit(${signal.reason}) id=${signal.position.id} entry=${signal.position.entryPrice} -> price=${signal.targetPrice}`;
}

// --- Contract implementation (non-breaking) ---
export class CoreRiskManager implements RiskManager {
    validateOrder(intent: any) {
        const minQty = Number(process.env.RISK_MIN_TRADE_SIZE || "0.0001");
        const maxNotional = Number(process.env.MAX_ORDER_NOTIONAL_JPY || "100000");
        const qty = Number(intent?.qty || 0);
        const price = Number(intent?.price || 0);
        if (qty <= 0) return err('RISK_QTY', `qty<=0`);
        if (qty < minQty) return err('RISK_MIN_SIZE', `min trade size ${minQty}`);
        if (price > 0 && price * qty > maxNotional) return err('RISK_NOTIONAL', `max notional ${maxNotional}`);
        return ok(void 0);
    }
    manageTrailingStop(state: any, price: number): TrailingAction | null {
        if (!state) return null;
        const TRIGGER_PCT = Number(process.env.RISK_TRAIL_TRIGGER_PCT || 0.01);
        const TRAIL_PCT = Number(process.env.RISK_TRAIL_STOP_PCT || 0.005);
        const entry = Number(state.entryPrice || 0);
        const side = (state.side === 'short') ? 'sell' : 'buy';
        if (!state.trailArmed) {
            if (entry > 0 && (Math.abs(price / entry - 1) >= TRIGGER_PCT)) {
                state.trailArmed = true;
                state.highestPrice = price;
            }
            return null;
        }
        const hp = state.highestPrice || price;
        if ((side === 'buy' && price > hp) || (side === 'sell' && price < hp)) state.highestPrice = price;
        const trigger = side === 'buy' ? state.highestPrice * (1 - TRAIL_PCT) : state.highestPrice * (1 + TRAIL_PCT);
        state.trailStop = trigger;
        return { side, trigger };
    }
    clampExposure(balance: any, intent: any): ClampedIntent {
        const positionPct = Number(process.env.RISK_POSITION_PCT || "0.05");
        const price = Number(intent?.price || 0);
        const side = String(intent?.side || 'buy') as 'buy'|'sell';
        if (!price) return { side, qty: 0, price };
        const alloc = Number(balance?.jpy || balance?.JPY || 0) * positionPct;
        const maxQty = Math.floor((alloc / price) * 1e8) / 1e8;
        const qty = Math.min(Number(intent?.qty || 0), maxQty);
        return { side, qty, price };
    }
}
