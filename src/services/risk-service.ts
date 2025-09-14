import fs from "fs";
import path from "path";
import { logInfo, logError, logWarn } from "../utils/logger";
import { getMaxHoldSec } from "../utils/toolkit";
import { updateFields } from "./position-store";

export interface RiskConfig {
    stopLossPct: number;        // 0.02 => 2%
    takeProfitPct: number;      // 0.05 => 5%
    positionPct: number;        // 0.05 => 5% of capital
    smaPeriod: number;          // e.g. 20
    positionsFile: string;      // JSON file path
    trailTriggerPct: number;    // 利益がこの%到達でトレーリング開始
    trailStopPct: number;       // 最高値からのドローダウン%で利確
    dcaStepPct: number;         // 前回エントリー価格からの下落率閾値で平均化追加
    maxPositions: number;       // 最大ポジション数
    maxDcaPerPair: number;      // DCA の最大回数(ポジション数)
    minTradeSize: number;       // 取引最小サイズ
    maxSlippagePct: number;     // 許容スリッページ
    indicatorIntervalSec: number; // インジケータ更新間隔秒
}

export type PositionSide = "long" | "short";

export interface Position {
    id: string;                  // ユニークID (timestamp-random)
    pair: string;                // e.g. "btc_jpy"
    side: PositionSide;          // ポジション方向
    entryPrice: number;          // entry price
    amount: number;              // base amount
    timestamp: number;           // ms
    highestPrice?: number;       // トレーリング用
    dcaCount?: number;           // 追加回数
    openOrderIds?: number[];     // 関連する未約定注文ID
}

export interface ExitSignal { position: Position; reason: "STOP_LOSS" | "TAKE_PROFIT" | "MA_BREAK" | "TRAIL_STOP" | "TIME_LIMIT"; targetPrice: number; }

export interface TrailResult { signal?: "EXIT_TRAIL"; reason?: any }

/**
 * Manage trailing stop for a position.
 * @param {number} lastPrice The last traded price.
 * @param {Position} position The position to manage.
 * @param {number} now The current time.
 * @returns {TrailResult|undefined} A trailing stop result or undefined.
 */
export function trailManager(lastPrice: number, position: Position, now: number): TrailResult | undefined {
    const TRIGGER_PCT = Number(process.env.RISK_TRAIL_TRIGGER_PCT || 0.01);
    const TRAIL_PCT = Number(process.env.RISK_TRAIL_STOP_PCT || 0.005);
    const MIN_HOLD_SEC = Number(process.env.MIN_HOLD_SEC || 30);
    const COOLDOWN_SEC = Number(process.env.COOLDOWN_SEC || 10);
    const holdMs = now - position.timestamp;
    let changed = false;
    if (!(position as any).trailArmed) {
        if (position.entryPrice > 0 && (lastPrice / position.entryPrice - 1) >= TRIGGER_PCT) {
            (position as any).trailArmed = true;
            (position as any).highestPrice = lastPrice;
            changed = true;
        }
    } else {
        const hp = (position as any).highestPrice || lastPrice;
        if (lastPrice > hp) { (position as any).highestPrice = lastPrice; changed = true; }
        const trailStop = (position as any).highestPrice * (1 - TRAIL_PCT);
        (position as any).trailStop = trailStop;
        if (!(position as any).lastTrailAt || now - (position as any).lastTrailAt > COOLDOWN_SEC * 1000) {
            (position as any).lastTrailAt = now; changed = true;
        }
        if (holdMs >= MIN_HOLD_SEC * 1000 && lastPrice <= trailStop) {
            return { signal: "EXIT_TRAIL", reason: { highestPrice: (position as any).highestPrice, trailStop } };
        }
    }
    if (changed) {
        updateFields(position.pair, {
            trailArmed: (position as any).trailArmed,
            trailStop: (position as any).trailStop,
            highestPrice: (position as any).highestPrice,
            lastTrailAt: now
        });
    }
    return undefined;
}

/**
 * Get the risk management configuration.
 * @returns {RiskConfig} The risk management configuration.
 */
export function loadRiskConfig(): RiskConfig {
    const positionsFile = process.env.POSITIONS_FILE || path.resolve(process.cwd(), ".positions.json");
    return {
        stopLossPct: Number(process.env.RISK_STOP_LOSS_PCT || 0.02),
        takeProfitPct: Number(process.env.RISK_TAKE_PROFIT_PCT || 0.05),
        positionPct: Number(process.env.RISK_POSITION_PCT || 0.05),
        smaPeriod: Number(process.env.RISK_SMA_PERIOD || 20),
        positionsFile,
        trailTriggerPct: Number(process.env.RISK_TRAIL_TRIGGER_PCT || 0.05),
        trailStopPct: Number(process.env.RISK_TRAIL_STOP_PCT || 0.03),
        dcaStepPct: Number(process.env.RISK_DCA_STEP_PCT || 0.01),
        maxPositions: Number(process.env.RISK_MAX_POSITIONS || 5),
        maxDcaPerPair: Number(process.env.RISK_MAX_DCA_PER_PAIR || 3),
        minTradeSize: Number(process.env.RISK_MIN_TRADE_SIZE || 0.0001),
        maxSlippagePct: Number(process.env.RISK_MAX_SLIPPAGE_PCT || 0.005),
        indicatorIntervalSec: Number(process.env.RISK_INDICATOR_INTERVAL_SEC || 60),
    };
}

/**
 * Read a file safely.
 * @param {string} file The file path.
 * @returns {string|null} The file contents or null if an error occurred.
 */
function readFileSafe(file: string): string | null {
    try { return fs.readFileSync(file, "utf8"); } catch { return null; }
}

/** Load positions from a file.
 * @param {string} file The file path.
 * @returns {Position[]} The list of positions.
 */
export function loadPositions(file: string): Position[] {
    const txt = readFileSafe(file);
    if (!txt) return [];
    try {
        const data = JSON.parse(txt);
        if (Array.isArray(data)) return data as Position[];
    } catch (err) { logError("JSONパース失敗", err); }
    return [];
}

/** Save positions to a file.
 * @param {string} file The file path.
 * @param {Position[]} positions The list of positions to save.
 */
export function savePositions(file: string, positions: Position[]) {
    try { fs.writeFileSync(file, JSON.stringify(positions, null, 2)); } catch (err) { logError("ポジション保存失敗", err); }
}

/** Open a new position and save it to file.
 * @param {string} file The file path.
 * @param {Omit<Position,"id"|"timestamp"> & { entryPrice:number; amount:number; }} p The position data excluding id and timestamp.
 * @returns {Position} The newly created position.
 */
export function openPosition(file: string, p: Omit<Position, "id" | "timestamp"> & { entryPrice: number; amount: number; }): Position {
    const positions = loadPositions(file);
    const pos: Position = { 
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, 
        timestamp: Date.now(), 
        ...p, 
        highestPrice: p.entryPrice, 
        dcaCount: 0, 
        openOrderIds: [] 
    };
    positions.push(pos);
    savePositions(file, positions);
    logInfo(`ポジション追加 id=${pos.id} pair=${pos.pair} entry=${pos.entryPrice}`);
    return pos;
}

/** Increment the DCA count for a position.
 * @param {string} file The file path.
 * @param {string} posId The position ID.
 */
export function incrementDca(file: string, posId: string) {
    const positions = loadPositions(file);
    const target = positions.find(p => p.id === posId);
    if (target) {
        target.dcaCount = (target.dcaCount || 0) + 1;
        savePositions(file, positions);
    }
}

/** Remove a position by ID.
 * @param {string} file The file path.
 * @param {string} id The position ID to remove.
 */
export function removePosition(file: string, id: string) {
    const positions = loadPositions(file).filter(p => p.id !== id);
    savePositions(file, positions);
}

/** Calculate Simple Moving Average (SMA).
 * @param {number[]} prices The list of prices (latest first).
 * @param {number} period The period for SMA calculation.
 * @returns {number|null} The calculated SMA or null if not enough data.
 */
export function calcSMA(prices: number[], period: number): number | null {
    if (prices.length < period) return null;
    const slice = prices.slice(0, period); // prices配列は先頭が最新価格である前提（例: tradesの配列）。この前提が変わる場合は要調整。
    const sum = slice.reduce((a, b) => a + b, 0);
    return sum / slice.length;
}

/** Calculate Relative Strength Index (RSI). 
 * @param {number[]} prices The list of prices (latest first).
 * @param {number} period The period for RSI calculation (default 14).
 * @returns {number|null} The calculated RSI or null if not enough data.
 */
export function calcRSI(prices: number[], period: number = 14): number | null {
    if (prices.length < period + 1) return null;
    let gains = 0, losses = 0;
    for (let i = 1; i <= period; i++) {
        const diff = prices[i - 1] - prices[i];
        if (diff > 0) gains += diff; else losses -= diff;
    }
    if (gains === 0 && losses === 0) return 50;
    const rs = losses === 0 ? 100 : gains / (losses || 1e-9);
    const rsi = 100 - (100 / (1 + rs));
    return rsi;
}

/** Calculate position size based on account balance and risk config.
 * @param {number} jpyBalance The account balance in JPY.
 * @param {number} price The current price of the asset.
 * @param {RiskConfig} cfg The risk configuration.
 * @returns {number} The calculated position size (amount of base currency).
 */
export function positionSizeFromBalance(jpyBalance: number, price: number, cfg: RiskConfig): number {
    if (price <= 0) return 0;
    const alloc = jpyBalance * cfg.positionPct;
    const amount = alloc / price;
    return Math.floor(amount * 1e8) / 1e8; // 8桁に丸め
}

/** Evaluate exit conditions for positions.
 * @param {Position[]} positions The list of current positions.
 * @param {number} currentPrice The current market price.
 * @param {number|null} sma The current SMA value or null if not available.
 * @param {RiskConfig} cfg The risk configuration.
 * @returns {ExitSignal[]} The list of exit signals for positions that meet exit conditions.
 */
export function evaluateExitSignals(positions: Position[], currentPrice: number, sma: number | null, cfg: RiskConfig): ExitSignal[] {
    const signals: ExitSignal[] = [];
    const maxHold = getMaxHoldSec();
    for (const pos of positions) {
        if (maxHold != null && (Date.now() - pos.timestamp) >= maxHold * 1000) {
            logWarn(`[TIME_LIMIT] force exit id=${pos.id} pair=${pos.pair} holdSec=${Math.floor((Date.now()-pos.timestamp)/1000)}>=${maxHold}`);
            signals.push({ position: pos, reason: "TIME_LIMIT", targetPrice: currentPrice });
            continue;
        }
        if (pos.side === "long") {
            // Trailing highest update (long)
            if (pos.highestPrice == null || currentPrice > pos.highestPrice) {
                pos.highestPrice = currentPrice;
            }
            // Trailing stop condition (long)
            if (typeof pos.highestPrice === "number" && pos.highestPrice >= pos.entryPrice * (1 + cfg.trailTriggerPct)) {
                const trailStopLine = pos.highestPrice * (1 - cfg.trailStopPct);
                if (currentPrice <= trailStopLine && currentPrice > pos.entryPrice) {
                    signals.push({ position: pos, reason: "TRAIL_STOP", targetPrice: currentPrice });
                    continue;
                }
            }
            // Stop loss (long)
            const stopLossPrice = pos.entryPrice * (1 - cfg.stopLossPct);
            if (currentPrice <= stopLossPrice) {
                signals.push({ position: pos, reason: "STOP_LOSS", targetPrice: currentPrice });
                continue;
            }
            // Take profit (long)
            const takeProfitPrice = pos.entryPrice * (1 + cfg.takeProfitPct);
            if (currentPrice >= takeProfitPrice) {
                signals.push({ position: pos, reason: "TAKE_PROFIT", targetPrice: currentPrice });
                continue;
            }
            // MA break (long)
            if (sma !== null && currentPrice < sma && currentPrice > pos.entryPrice) {
                signals.push({ position: pos, reason: "MA_BREAK", targetPrice: currentPrice });
                continue;
            }
        } else if (pos.side === "short") {
            // Trailing lowest update (short)
            if (pos.highestPrice == null || currentPrice < pos.highestPrice) {
                pos.highestPrice = currentPrice;
            }
            // Trailing stop condition (short)
            if (typeof pos.highestPrice === "number" && pos.highestPrice <= pos.entryPrice * (1 - cfg.trailTriggerPct)) {
                const trailStopLine = pos.highestPrice * (1 + cfg.trailStopPct);
                if (currentPrice >= trailStopLine && currentPrice < pos.entryPrice) {
                    signals.push({ position: pos, reason: "TRAIL_STOP", targetPrice: currentPrice });
                    continue;
                }
            }
            // Stop loss (short)
            const stopLossPrice = pos.entryPrice * (1 + cfg.stopLossPct);
            if (currentPrice >= stopLossPrice) {
                signals.push({ position: pos, reason: "STOP_LOSS", targetPrice: currentPrice });
                continue;
            }
            // Take profit (short)
            const takeProfitPrice = pos.entryPrice * (1 - cfg.takeProfitPct);
            if (currentPrice <= takeProfitPrice) {
                signals.push({ position: pos, reason: "TAKE_PROFIT", targetPrice: currentPrice });
                continue;
            }
            // MA break (short)
            if (sma !== null && currentPrice > sma && currentPrice < pos.entryPrice) {
                signals.push({ position: pos, reason: "MA_BREAK", targetPrice: currentPrice });
                continue;
            }
        }
    }
    return signals;
}

/**
 * Returns a string describing the exit signal.
 * @example
 * // Output: exit(STOP_LOSS) id=1681234567890-abc123 entry=5000000 -> price=4900000
 * describeExit(signal);
 * @param {ExitSignal} signal The exit signal to describe.
 * @returns {string} A string describing the exit signal.
 */
export function describeExit(signal: ExitSignal): string {
    return `exit(${signal.reason}) id=${signal.position.id} entry=${signal.position.entryPrice} -> price=${signal.targetPrice}`;
}
