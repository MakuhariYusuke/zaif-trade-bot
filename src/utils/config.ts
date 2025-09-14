import fs from "fs";
import path from "path";
import { setNonceBase } from "./signer";

export interface AppConfig {
    dryRun: boolean;
    nonceStorePath: string;
}

const DEFAULT_NONCE_FILE = path.resolve(process.cwd(), ".nonce_store");

function isTruthyEnv(value?: string): boolean {
    if (!value) return false;
    const truthyValues = ["1", "true", "yes", "on"];
    return truthyValues.includes(value.toLowerCase());
}

let __cachedAppConfig: AppConfig | null = null;
export function loadAppConfig(): AppConfig {
    if (__cachedAppConfig) return __cachedAppConfig;
    __cachedAppConfig = {
        dryRun: isTruthyEnv(process.env.DRY_RUN),
        nonceStorePath: process.env.NONCE_FILE || DEFAULT_NONCE_FILE,
    };
    return __cachedAppConfig;
}

/**
 * Test helper: reset cached app config so subsequent calls re-read env.
 */
export function resetConfigCache(){
    __cachedAppConfig = null;
}

export function loadTradeMode(): "SELL" | "BUY" {
    const v = (process.env.TRADE_MODE || 'SELL').toUpperCase();
    return v === 'BUY' ? 'BUY' : 'SELL';
}

export function loadTradeFlow(): "BUY_ONLY" | "SELL_ONLY" | "BUY_SELL" | "SELL_BUY" {
    const v = (process.env.TRADE_FLOW || 'BUY_SELL').toUpperCase();
    if (v === 'BUY_ONLY' || v === 'SELL_ONLY' || v === 'SELL_BUY') return v;
    return 'BUY_SELL';
}

/** Load pairs from env: PAIRS=btc_jpy,eth_jpy; default ['btc_jpy'] */
export function loadPairs(): string[] {
    const v = process.env.PAIRS || process.env.PAIR || 'btc_jpy,eth_jpy,xrp_jpy';
    return v.split(',')
        .map(s => s.trim())
        .filter(Boolean)
        .map(s => s.replace('/', '_').toLowerCase());
}

export function restoreNonce(file: string) {
    try {
        if (fs.existsSync(file)) {
            const txt = fs.readFileSync(file, "utf8").trim();
            const n = Number(txt);
            if (!Number.isNaN(n)) setNonceBase(n);
        }
    } catch { /* ignore */ }
}

export function persistNonce(file: string, nonce: number) {
    try {
        fs.writeFileSync(file, String(nonce));
    } catch { /* ignore */}
}
