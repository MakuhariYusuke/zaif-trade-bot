import fs from "fs";
import path from "path";
import { OrderLifecycleSummary } from "../types/private";

export interface DailyAggregate {
    date: string;
    trades: number;
    wins: number;
    winTrades?: number;
    lossTrades?: number;
    streakWin?: number;
    streakLoss?: number;
    realizedPnl: number;
    sumSlippage: number;
    maxSlippage: number;
    sumRetries: number;
    maxRetries: number;  // maximum number of retries for a trade
    filledCount?: number;   // number of partial exit fills counted separately
    trailArmedTotal?: number;
    trailExitTotal?: number;
    sellEntries?: number; // SELL モードのエントリーカウント
    buyEntries?: number;  // BUY モードのエントリーカウント
    buyExits?: number;    // SELL モードの買い戻し等、必要なら継続
    rsiExits?: number;
    trailStops?: number;
    sumSubmitRetryCount?: number;
    sumPollRetryCount?: number;
    sumCancelRetryCount?: number;
    sumNonceRetryCount?: number;
    maxSubmitRetryCount?: number;
    maxPollRetryCount?: number;
    maxCancelRetryCount?: number;
    maxNonceRetryCount?: number;
}

const DIR = process.env.STATS_DIR || path.resolve(process.cwd(), "logs");

function fileFor(date: string, pair?: string) {
    if (pair) return path.join(DIR, 'pairs', pair, `stats-${date}.json`);
    const envPair = process.env.STATS_PAIR;
    if (envPair) return path.join(DIR, 'pairs', envPair, `stats-${date}.json`);
    return path.join(DIR, `stats-${date}.json`);
}

/**
 * Load the daily aggregate statistics for the given date.
 * If the file does not exist or cannot be read, returns a default aggregate with zeroed fields.
 * @param {string} date - The date string in 'YYYY-MM-DD' format.
 * @return {DailyAggregate} The daily aggregate statistics.
 */
export function loadDaily(date: string, pair?: string): DailyAggregate {
    try {
    const f = fileFor(date, pair);
        if (!fs.existsSync(f)) return { date, trades: 0, wins: 0, realizedPnl: 0, sumSlippage: 0, maxSlippage: 0, sumRetries: 0, maxRetries: 0 };
        return JSON.parse(fs.readFileSync(f, "utf8"));
    } catch { return { date, trades: 0, wins: 0, realizedPnl: 0, sumSlippage: 0, maxSlippage: 0, sumRetries: 0, maxRetries: 0 }; }
}

/**
 * Appends a single order lifecycle summary to the daily aggregate file for the given date.
 *
 * This function:
 * - Loads the existing aggregate for `date` via `loadDaily(date)`.
 * - Updates aggregate counters and statistics (trade count, wins, realized PnL, filled count).
 * - Accumulates and tracks slippage statistics (sum and max absolute slippage).
 * - Accumulates retry statistics: total sum of retries and per-category sums
 *   and per-category maxima (submit, poll, cancel, nonce).
 * - Ensures the target directory exists (best-effort; errors during directory creation are swallowed).
 * - Persists the updated aggregate to disk by writing a pretty-printed JSON file returned by `fileFor(date)`.
 *
 * Notes:
 * - The function has side effects: it reads and mutates on-disk state and depends on external functions/values
 *   (`loadDaily`, `fileFor`, `DIR`, and `fs`).
 * - It does not return a value.
 * - The directory creation is attempted inside a try/catch; failures to create the directory are ignored.
 * - `fs.writeFileSync` is used to persist the file and may throw on I/O or permission errors.
 * - Concurrent invocations for the same date are not synchronized by this function and can lead to lost updates
 *   (race conditions). If concurrent access is possible, callers should serialize updates or use a safer
 *   atomic/locking strategy.
 *
 * Parameters on `s` that are read (all optional unless noted):
 * - `pnl?: number` — realized profit and loss to add to `realizedPnl`.
 * - `win?: boolean` — when truthy increments the `wins` counter.
 * - `filledCount?: number` — increments `filledCount`.
 * - `slippagePct?: number` — contributes to `sumSlippage` and updates `maxSlippage` if its absolute value is larger.
 * - `totalRetryCount?: number` — added to `sumRetries` and may update `maxRetries`.
 * - `submitRetryCount?: number`, `pollRetryCount?: number`, `cancelRetryCount?: number`, `nonceRetryCount?: number`
 *   — per-category retry sums and maxima are updated accordingly.
 *
 * @param {string} date - Identifier for the day to which the summary should be appended (string, typically a date key).
 * @param {OrderLifecycleSummary & { pnl?: number; win?: boolean }} s - An order lifecycle summary augmented with optional `pnl` and `win` fields.
 *
 * @throws {Error} If writing the aggregate JSON to disk fails (e.g. I/O error, permission denied).
 *
 * @returns {void}
 */
export function appendSummary(date: string, s: OrderLifecycleSummary & { pnl?: number; win?: boolean; pair?: string }) {
    const agg = loadDaily(date, s.pair);
    agg.trades += 1;
    agg.wins += s.win ? 1 : 0;
    if (typeof s.pnl === 'number') agg.realizedPnl += s.pnl;
    agg.filledCount = (agg.filledCount || 0) + (s.filledCount || 0);
    if (typeof s.slippagePct === 'number') {
        agg.sumSlippage += s.slippagePct;
        if (Math.abs(s.slippagePct) > Math.abs(agg.maxSlippage)) agg.maxSlippage = s.slippagePct;
    }
    const totalRetries = (s.totalRetryCount || 0);
    agg.sumRetries += totalRetries;
    if (totalRetries > (agg.maxRetries || 0)) agg.maxRetries = totalRetries;
    // per category
    agg.sumSubmitRetryCount = (agg.sumSubmitRetryCount || 0) + (s.submitRetryCount || 0);
    agg.sumPollRetryCount = (agg.sumPollRetryCount || 0) + (s.pollRetryCount || 0);
    agg.sumCancelRetryCount = (agg.sumCancelRetryCount || 0) + (s.cancelRetryCount || 0);
    agg.sumNonceRetryCount = (agg.sumNonceRetryCount || 0) + (s.nonceRetryCount || 0);
    if ((s.submitRetryCount || 0) > (agg.maxSubmitRetryCount || 0)) agg.maxSubmitRetryCount = s.submitRetryCount || 0;
    if ((s.pollRetryCount || 0) > (agg.maxPollRetryCount || 0)) agg.maxPollRetryCount = s.pollRetryCount || 0;
    if ((s.cancelRetryCount || 0) > (agg.maxCancelRetryCount || 0)) agg.maxCancelRetryCount = s.cancelRetryCount || 0;
    if ((s.nonceRetryCount || 0) > (agg.maxNonceRetryCount || 0)) agg.maxNonceRetryCount = s.nonceRetryCount || 0;
    try {
        const f = fileFor(date, s.pair);
        const d = path.dirname(f);
        if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
        fs.writeFileSync(f, JSON.stringify(agg, null, 2));
    } catch { }
}

// Append only realized PnL for partial EXIT fills without counting as a trade lifecycle
export function appendFillPnl(date: string, pnl: number, pair?: string) {
    const agg = loadDaily(date, pair);
    agg.realizedPnl += pnl;
    agg.filledCount = (agg.filledCount || 0) + 1;
    if (pnl > 0) {
        agg.winTrades = (agg.winTrades || 0) + 1;
        agg.streakWin = (agg.streakWin || 0) + 1;
        agg.streakLoss = 0;
    } else if (pnl < 0) {
        agg.lossTrades = (agg.lossTrades || 0) + 1;
        agg.streakLoss = (agg.streakLoss || 0) + 1;
        agg.streakWin = 0;
    }
    try {
        const f = fileFor(date, pair);
        const d = path.dirname(f);
        if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
        fs.writeFileSync(f, JSON.stringify(agg, null, 2));
    } catch { }
}

export function incTrailArmed(date: string, pair?: string) {
    const agg = loadDaily(date, pair);
    agg.trailArmedTotal = (agg.trailArmedTotal || 0) + 1;
    try {
        const f = fileFor(date, pair);
        const d = path.dirname(f);
        if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
        fs.writeFileSync(f, JSON.stringify(agg, null, 2));
    } catch {}
}

export function incTrailExit(date: string, pair?: string) {
    const agg = loadDaily(date, pair);
    agg.trailExitTotal = (agg.trailExitTotal || 0) + 1;
    try {
        const f = fileFor(date, pair);
        const d = path.dirname(f);
        if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
        fs.writeFileSync(f, JSON.stringify(agg, null, 2));
    } catch {}
}

export function incSellEntry(date: string, pair?: string) {
    const agg = loadDaily(date, pair);
    agg.sellEntries = (agg.sellEntries||0)+1;
    try { 
        const f = fileFor(date, pair);
        const d = path.dirname(f);
        if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
        fs.writeFileSync(f, JSON.stringify(agg, null, 2));
    } catch {}
}
export function incBuyEntry(date: string, pair?: string) {
    const agg = loadDaily(date, pair);
    agg.buyEntries = (agg.buyEntries||0)+1;
    try {
        const f = fileFor(date, pair);
        const d = path.dirname(f);
        if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
        fs.writeFileSync(f, JSON.stringify(agg, null, 2));
    } catch {}
}
export function incBuyExit(date: string, pair?: string) {
    const agg = loadDaily(date, pair);
    agg.buyExits = (agg.buyExits||0)+1;
    try {
        const f = fileFor(date, pair);
        const d = path.dirname(f);
        if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
        fs.writeFileSync(f, JSON.stringify(agg, null, 2));
    } catch {}
}
export function incRsiExit(date: string, pair?: string) {
    const agg = loadDaily(date, pair);
    agg.rsiExits = (agg.rsiExits||0)+1;
    try {
        const f = fileFor(date, pair);
        const d = path.dirname(f);
        if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
        fs.writeFileSync(f, JSON.stringify(agg, null, 2));
    } catch {}
}
export function incTrailStop(date: string, pair?: string) {
    const agg = loadDaily(date, pair);
    agg.trailStops = (agg.trailStops||0)+1;
    try {
        const f = fileFor(date, pair);
        const d = path.dirname(f);
        if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
        fs.writeFileSync(f, JSON.stringify(agg, null, 2));
    } catch {}
}

export function summarizeDaily(date: string, pair?: string) {
    const agg = loadDaily(date, pair);
    const avgSlip = agg.trades ? agg.sumSlippage / agg.trades : 0;
    const avgRetries = agg.trades ? agg.sumRetries / agg.trades : 0;
    const avgTotalRetries = avgRetries;
    const maxTotalRetries = agg.maxRetries || 0;
    console.log(`[DAILY] pair=${pair||process.env.STATS_PAIR||'-'} trades=${agg.trades} fills=${agg.filledCount||0} pnl=${agg.realizedPnl} winStreak=${agg.streakWin||0} lossStreak=${agg.streakLoss||0} avgRetries=${avgTotalRetries.toFixed(2)} maxRetries=${maxTotalRetries} trailArmed=${agg.trailArmedTotal||0} trailExit=${agg.trailExitTotal||0} sellEntries=${agg.sellEntries||0} buyExits=${agg.buyExits||0} rsiExits=${agg.rsiExits||0} trailStops=${agg.trailStops||0}`);
    return { ...agg, avgSlippage: avgSlip, avgRetries: avgTotalRetries, winRate: agg.trades ? agg.wins / agg.trades : 0 };
}