import fs from "fs";
import path from "path";
import { logInfo, logError, logWarn } from "../utils/logger";
import { getMaxHoldSec } from "../utils/toolkit";
import { updatePositionFields } from "./position-store";

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
    const positionsFile = process.env.POSITIONS_FILE || path.resolve(process.cwd(), ".positions.json");
    return { stopLossPct: Number(process.env.RISK_STOP_LOSS_PCT || 0.02), takeProfitPct: Number(process.env.RISK_TAKE_PROFIT_PCT || 0.05), positionPct: Number(process.env.RISK_POSITION_PCT || 0.05), smaPeriod: Number(process.env.RISK_SMA_PERIOD || 20), positionsFile, trailTriggerPct: Number(process.env.RISK_TRAIL_TRIGGER_PCT || 0.05), trailStopPct: Number(process.env.RISK_TRAIL_STOP_PCT || 0.03), dcaStepPct: Number(process.env.RISK_DCA_STEP_PCT || 0.01), maxPositions: Number(process.env.RISK_MAX_POSITIONS || 5), maxDcaPerPair: Number(process.env.RISK_MAX_DCA_PER_PAIR || 3), minTradeSize: Number(process.env.RISK_MIN_TRADE_SIZE || 0.0001), maxSlippagePct: Number(process.env.RISK_MAX_SLIPPAGE_PCT || 0.005), indicatorIntervalSec: Number(process.env.RISK_INDICATOR_INTERVAL_SEC || 60) };
}
function readFileSafe(file: string): string | null {
    try {
        return fs.readFileSync(file, "utf8");
    } catch { return null; }
}
export function getPositions(file: string): Position[] {
    const txt = readFileSafe(file);
    if (!txt) return [];
    try {
        const data = JSON.parse(txt);
        if (Array.isArray(data)) return data as Position[];
    } catch (err) { logError("JSON parse fail", err); }
    return [];
}
export function savePositionsToFile(file: string, positions: Position[]) {
    try { fs.writeFileSync(file, JSON.stringify(positions, null, 2)); } catch (err) { logError("save positions fail", err); }
}
export function openPosition(file: string, p: Omit<Position, "id" | "timestamp"> & { entryPrice: number; amount: number; }): Position {
    const positions = getPositions(file);
    const pos: Position = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        timestamp: Date.now(),
        ...p,
        highestPrice: p.entryPrice,
        dcaCount: 0,
        openOrderIds: []
    };
    positions.push(pos);
    savePositionsToFile(file, positions);
    logInfo(`ポジション追加 id=${pos.id} pair=${pos.pair} entry=${pos.entryPrice}`);
    return pos;
}
export function incrementDca(file: string, posId: string) {
    const positions = getPositions(file);
    const target = positions.find(p => p.id === posId);
    if (target) {
        target.dcaCount = (target.dcaCount || 0) + 1;
        savePositionsToFile(file, positions);
    }
}
export function removePosition(file: string, id: string) {
    const positions = getPositions(file).filter(p => p.id !== id);
    savePositionsToFile(file, positions);
}
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
            logWarn(`[TIME_LIMIT] force exit id=${pos.id} pair=${pos.pair} holdSec=${Math.floor((Date.now()-pos.timestamp)/1000)}>=${maxHold}`);
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
