import fs from "fs";
import path from "path";

const LOG_DIR = process.env.LOG_DIR || path.resolve(process.cwd(), "logs");

function dailyLogPath(dateStr?: string) {
    const d = dateStr || new Date().toISOString().slice(0, 10);
    return path.join(LOG_DIR, `trades-${d}.log`);
}

let currentDate = new Date().toISOString().slice(0, 10);
let currentLogPath = dailyLogPath(currentDate);

function rotateIfNeeded() {
    const today = new Date().toISOString().slice(0, 10);
    if (today !== currentDate) {
        currentDate = today;
        currentLogPath = dailyLogPath(today);
    }
}

function ensureDir() {
    fs.mkdirSync(LOG_DIR, { recursive: true });
}

export type TradeLogType = "SIGNAL" | "ORDER" | "EXECUTION" | "ERROR" | "INFO";

export interface TradeLogEntry {
    ts: string;           // ISO timestamp
    type: TradeLogType;
    message: string;
    data?: any;
}

export function logTrade(entry: TradeLogEntry) {
    try {
        rotateIfNeeded();
        ensureDir();
        fs.appendFileSync(currentLogPath, JSON.stringify(entry) + "\n");
    } catch (err) {
        console.error("Failed to write trade log:", err);
    }
}

export function logSignal(message: string, data?: any) {
    logTrade({ ts: new Date().toISOString(), type: "SIGNAL", message, data });
}
export function logOrder(message: string, data?: any) {
    logTrade({ ts: new Date().toISOString(), type: "ORDER", message, data });
}
export function logExecution(message: string, data?: any) {
    logTrade({ ts: new Date().toISOString(), type: "EXECUTION", message, data });
}
export function logTradeError(message: string, data?: any) {
    logTrade({ ts: new Date().toISOString(), type: "ERROR", message, data });
}
export function logTradeInfo(message: string, data?: any) {
    logTrade({ ts: new Date().toISOString(), type: "INFO", message, data });
}

export interface DailyReport {
    date: string;
    trades: number;
    signals: number;
    pnlEstimate: number; // 簡易 (約定ログが無いため推定用)
}

/**
 * Generate a daily report for the given date.
 * @param {string} date - The date string in 'YYYY-MM-DD' format.
 * @returns {DailyReport} The daily report for the specified date.
 */
export function generateDailyReport(date: string): DailyReport {
    ensureDir();
    const file = dailyLogPath(date);
    if (!fs.existsSync(file)) return { date, trades: 0, signals: 0, pnlEstimate: 0 };
    const content = fs.readFileSync(file, "utf8").trim();
    if (!content || content === "") return { date, trades: 0, signals: 0, pnlEstimate: 0 };
    const lines = content.split(/\n+/).filter(line => line.trim() !== "");
    let trades = 0, signals = 0, pnl = 0;
    for (const line of lines) {
        try {
            const e = JSON.parse(line) as TradeLogEntry;
            if (e.type === "SIGNAL") signals++;
            if (e.type === "EXECUTION") trades++;
            if (e.type === "EXECUTION" && e.data && typeof e.data.pnl === "number") pnl += e.data.pnl;
        } catch { /* ignore */ }
    }
    return { date, trades, signals, pnlEstimate: pnl };
}