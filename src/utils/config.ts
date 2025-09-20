import fs from "fs";
import path from "path";
import { z } from "zod";
import { setNonceBase } from "./signer";

export interface AppConfig {
    dryRun: boolean;
    nonceStorePath: string;
}

const DEFAULT_NONCE_FILE = path.resolve(process.cwd(), ".nonce_store");

// Zod schema for environment validation
const envSchema = z.object({
    DRY_RUN: z.string().optional().transform(val => val ? ["1", "true", "yes", "on"].includes(val.toLowerCase()) : false),
    NONCE_FILE: z.string().optional(),
    TRADE_MODE: z.enum(["SELL", "BUY"]).optional().default("SELL"),
    TRADE_FLOW: z.enum(["BUY_ONLY", "SELL_ONLY", "BUY_SELL", "SELL_BUY"]).optional().default("BUY_SELL"),
    PAIRS: z.string().optional().default("btc_jpy,eth_jpy,xrp_jpy"),
    // Add more env vars as needed
});

type EnvVars = z.infer<typeof envSchema>;

function isTruthyEnv(value?: string): boolean {
    if (!value) return false;
    const truthyValues = ["1", "true", "yes", "on"];
    return truthyValues.includes(value.toLowerCase());
}

let __cachedAppConfig: AppConfig | null = null;
export function loadAppConfig(): AppConfig {
    if (__cachedAppConfig) return __cachedAppConfig;
    const env = envSchema.parse(process.env);
    __cachedAppConfig = {
        dryRun: env.DRY_RUN,
        nonceStorePath: env.NONCE_FILE || DEFAULT_NONCE_FILE,
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
    const env = envSchema.parse(process.env);
    return env.TRADE_MODE;
}

export function loadTradeFlow(): "BUY_ONLY" | "SELL_ONLY" | "BUY_SELL" | "SELL_BUY" {
    const env = envSchema.parse(process.env);
    return env.TRADE_FLOW;
}

/** Load pairs from env: PAIRS=btc_jpy,eth_jpy; default ['btc_jpy'] */
export function loadPairs(): string[] {
    const env = envSchema.parse(process.env);
    return env.PAIRS.split(',')
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
