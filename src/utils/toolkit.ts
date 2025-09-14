import type { PrivateApi } from "../types/private";
import { logWarn } from './logger';
import { spawnSync } from 'child_process';
import fs from 'fs';
import readline from 'readline';

/** Return YYYY-MM-DD for given date (defaults to now). */
export function todayStr(d: Date = new Date()): string { return d.toISOString().slice(0, 10); }

/**
 * Sleep helper for tools.
 * - When FAST_CI=1, shorten waits to speed up CI runs. Upper bound is TEST_SLEEP_MS (default 5ms).
 */
export function sleep(ms: number): Promise<void> {
  let delay = ms;
  try {
    if (process.env.FAST_CI === '1') {
      const cap = Math.max(0, Number(process.env.TEST_SLEEP_MS || '5'));
      delay = Math.min(delay, cap);
    }
  } catch {}
  return new Promise(r => setTimeout(r, delay));
}

/** Extract base asset from pair like "btc_jpy" -> "btc" (always lowercase). */
export function baseFromPair(pair: string): string { return ((pair || 'btc_jpy').split('_')[0]).toLowerCase(); }

/**
 * Read balances via PrivateApi.get_info2 and return funds map.
 * Throws when response is not successful.
 */
export async function fetchBalances(api: PrivateApi): Promise<Record<string, number>> {
  const r = await api.get_info2();
  if (!r.success || !r.return) throw new Error(r.error || 'get_info2 failed');
  return r.return.funds || {};
}

/**
 * Clamp the trade amount for safety if SAFETY_MODE=1.
 * - bid: limit to 10% of JPY notional at given price
 * - ask: limit to 10% of base asset balance
 */
export function clampAmountForSafety(
  side: 'bid' | 'ask', amount: number, price: number, funds: Record<string, number>, pair: string
): number {
  if (process.env.SAFETY_MODE !== '1') return amount;
  const pct = Math.max(0, Math.min(1, Number(process.env.SAFETY_CLAMP_PCT ?? '0.10')));
  const base = baseFromPair(pair);
  if (side === 'bid') {
    const jpy = Number((funds as any).jpy || 0);
    const maxSpend = jpy * pct;
    if (maxSpend <= 0 || price <= 0) return amount;
    const maxQty = maxSpend / price;
    if (amount > maxQty) {
      const clamped = maxQty;
      try { logWarn(`[SAFETY] amount clamped side=bid requested=${amount} clamped=${clamped} pct=${(pct*100).toFixed(1)}%`); } catch {}
      return clamped;
    }
    return amount;
  } else {
    const bal = Number((funds as any)[base] || 0);
    const maxQty = bal * pct;
    if (amount > maxQty) {
      const clamped = maxQty;
      try { logWarn(`[SAFETY] amount clamped side=ask requested=${amount} clamped=${clamped} pct=${(pct*100).toFixed(1)}%`); } catch {}
      return clamped;
    }
    return amount;
  }
}

/** Exposure warn threshold (ratio), default 0.05 (5%). */
export function getExposureWarnPct(): number {
  const raw = Number(process.env.EXPOSURE_WARN_PCT ?? '0.05');
  if (!isFinite(raw) || raw < 0) return 0.05;
  if (raw > 1) return 1;
  return raw;
}

/** Compute exposure ratio: bid -> notional/JPY, ask -> qty/baseBalance. Returns 0 if denominator is not positive. */
export function computeExposureRatio(side: 'bid'|'ask', qty: number, price: number, funds: Record<string, number>, pair: string): number {
  try {
    if (side === 'bid') {
      const jpy = Number((funds as any).jpy || 0);
      if (jpy <= 0 || price <= 0) return 0;
      return (qty * price) / jpy;
    } else {
      const base = baseFromPair(pair);
      const bal = Number((funds as any)[base] || 0);
      if (bal <= 0) return 0;
      return qty / bal;
    }
  } catch { return 0; }
}

/** Return configured max exposure ratio (hard cap), default Infinity (disabled). */
export function getExposureHardCap(): number | null {
  const raw = process.env.EXPOSURE_HARD_CAP;
  if (!raw) return null;
  const v = Number(raw);
  if (!isFinite(v) || v <= 0) return null;
  return Math.min(v, 1);
}

/** Check and clamp qty by exposure hard cap if configured; emits WARN when clamped. */
export function clampByExposureCap(side: 'bid'|'ask', qty: number, price: number, funds: Record<string, number>, pair: string): number {
  const cap = getExposureHardCap();
  if (cap == null) return qty;
  const ratio = computeExposureRatio(side, qty, price, funds, pair);
  if (ratio <= cap) return qty;
  let newQty = qty;
  if (side === 'bid') {
    const jpy = Number((funds as any).jpy || 0);
    newQty = (cap * jpy) / Math.max(1e-9, price);
  } else {
    const base = baseFromPair(pair);
    const bal = Number((funds as any)[base] || 0);
    newQty = cap * bal;
  }
  try { logWarn(`[SAFETY] exposure cap applied side=${side} cap=${(cap*100).toFixed(1)}% qty:${qty}=>${newQty}`); } catch {}
  return newQty;
}

/** Max position hold seconds (hard stop), default disabled. */
export function getMaxHoldSec(): number | null {
  const raw = process.env.MAX_HOLD_SEC;
  if (!raw) return null;
  const v = Number(raw);
  if (!isFinite(v) || v <= 0) return null;
  return Math.floor(v);
}

// --- ML/Features helpers ---
export interface FeatureCsvRow {
  ts: number; pair: string; side: string;
  rsi?: number; sma_short?: number; sma_long?: number;
  price: number; qty: number; pnl?: number; win?: number;
}

/**
 * Read all feature CSV files from a directory into rows.
 * Expects a header line with columns used by features-logger.
 */
export function readFeatureCsvRows(dir: string, date?: string): FeatureCsvRow[] {
  const fs = require('fs') as typeof import('fs');
  const path = require('path') as typeof import('path');
  const rows: FeatureCsvRow[] = [];
  if (!fs.existsSync(dir)) return rows;
  const filter = (f: string) => f.startsWith('features-') && (f.endsWith('.csv') || f.endsWith('.jsonl')) && (!date || f.includes(date));
  const files = fs.readdirSync(dir).filter(filter);
  for (const f of files) {
    const full = path.join(dir, f);
    try {
      if (f.endsWith('.jsonl')){
        const txt = fs.readFileSync(full, 'utf8');
        const lines = txt.split(/\r?\n/);
        for (const line of lines){
          const t = line.trim(); if (!t) continue;
          try {
            const o = JSON.parse(t);
            const toNum = (v: any): number|undefined => { const n = Number(v); return Number.isFinite(n) ? n : undefined; };
            rows.push({
              ts: Number(o.ts), pair: String(o.pair), side: String(o.side),
              rsi: toNum(o.rsi), sma_short: toNum(o.sma_short), sma_long: toNum(o.sma_long),
              price: Number(o.price), qty: Number(o.qty), pnl: toNum(o.pnl), win: toNum(o.win)
            });
          } catch { }
        }
      } else {
        const txt = fs.readFileSync(full, 'utf8');
        const [header, ...lines] = txt.trim().split(/\r?\n/);
        const cols = header.split(',');
        for (const line of lines) {
          if (!line.trim()) continue;
          const parts = line.split(',');
          const rec: any = {};
          cols.forEach((c, i) => rec[c] = parts[i]);
          const toNum = (v: any): number|undefined => {
            if (v == null) return undefined;
            const s = String(v).trim();
            if (s === '') return undefined;
            const n = Number(s);
            return Number.isFinite(n) ? n : undefined;
          };
          rows.push({
            ts: Number(rec.ts), pair: rec.pair, side: rec.side,
            rsi: toNum(rec.rsi),
            sma_short: toNum(rec.sma_short),
            sma_long: toNum(rec.sma_long),
            price: Number(rec.price), qty: Number(rec.qty),
            pnl: toNum(rec.pnl),
            win: toNum(rec.win),
          });
        }
      }
    } catch { /* ignore file errors */ }
  }
  return rows;
}

/** Resolve the root directory for features logs. */
export function getFeaturesRoot(): string {
  const path = require('path') as typeof import('path');
  const base = process.env.FEATURES_LOG_DIR ? path.resolve(process.env.FEATURES_LOG_DIR) : path.resolve(process.cwd(), 'logs');
  return path.join(base, 'features');
}

export interface FeatureCsvFile { file: string; source: 'paper'|'live'|'root'; pair: string }
export interface FeatureFile extends FeatureCsvFile { format: 'csv'|'jsonl' }

/** Enumerate feature CSV files under features root, scanning paper/, live/, and root-level pair dirs. */
export function enumerateFeatureCsvFiles(rootDir?: string): FeatureCsvFile[] {
  const fs = require('fs') as typeof import('fs');
  const path = require('path') as typeof import('path');
  const root = rootDir ? path.resolve(rootDir) : getFeaturesRoot();
  const out: FeatureCsvFile[] = [];
  if (!fs.existsSync(root)) return out;
  const sources: Array<'paper'|'live'> = ['paper','live'];
  for (const src of sources) {
    const srcDir = path.join(root, src);
    if (!fs.existsSync(srcDir) || !fs.statSync(srcDir).isDirectory()) continue;
    const pairs = fs.readdirSync(srcDir).filter((n: string) => fs.statSync(path.join(srcDir, n)).isDirectory());
    for (const pair of pairs) {
      const dir = path.join(srcDir, pair);
  const files = fs.readdirSync(dir).filter((f: string) => f.startsWith('features-') && f.endsWith('.csv'));
      for (const f of files) out.push({ file: path.join(dir, f), source: src, pair });
    }
  }
  // root-level pairs (legacy)
  const rootPairs = fs.readdirSync(root).filter((n: string) => {
    const p = path.join(root, n);
    return fs.statSync(p).isDirectory() && !sources.includes(n as any);
  });
  for (const pair of rootPairs) {
    const dir = path.join(root, pair);
    const files = fs.readdirSync(dir).filter((f: string) => f.startsWith('features-') && f.endsWith('.csv'));
    for (const f of files) out.push({ file: path.join(dir, f), source: 'root', pair });
  }
  return out;
}

/** Enumerate both CSV and JSONL feature files. */
export function enumerateFeatureFiles(rootDir?: string): FeatureFile[] {
  const fs = require('fs') as typeof import('fs');
  const path = require('path') as typeof import('path');
  const root = rootDir ? path.resolve(rootDir) : getFeaturesRoot();
  const out: FeatureFile[] = [];
  if (!fs.existsSync(root)) return out;
  const sources: Array<'paper'|'live'> = ['paper','live'];
  for (const src of sources) {
    const srcDir = path.join(root, src);
    if (!fs.existsSync(srcDir) || !fs.statSync(srcDir).isDirectory()) continue;
    const pairs = fs.readdirSync(srcDir).filter((n: string) => fs.statSync(path.join(srcDir, n)).isDirectory());
    for (const pair of pairs) {
      const dir = path.join(srcDir, pair);
      const files = fs.readdirSync(dir).filter((f: string) => f.startsWith('features-') && (f.endsWith('.csv') || f.endsWith('.jsonl')));
      for (const f of files) out.push({ file: path.join(dir, f), source: src, pair, format: f.endsWith('.jsonl') ? 'jsonl':'csv' });
    }
  }
  // root-level pairs (legacy)
  const rootPairs = fs.readdirSync(root).filter((n: string) => {
    const p = path.join(root, n);
    return fs.statSync(p).isDirectory() && !sources.includes(n as any);
  });
  for (const pair of rootPairs) {
    const dir = path.join(root, pair);
    const files = fs.readdirSync(dir).filter((f: string) => f.startsWith('features-') && (f.endsWith('.csv') || f.endsWith('.jsonl')));
    for (const f of files) out.push({ file: path.join(dir, f), source: 'root', pair, format: f.endsWith('.jsonl') ? 'jsonl':'csv' });
  }
  return out;
}

/** Run ml-simulate (ts) with given params and pair, returning parsed JSON result or null on failure. */
export function runMlSimulate(params: Record<string, any>, pair: string): any | null {
  const path = require('path') as typeof import('path');
  // Use top-level shim for portability
  // Resolve relative to this file to avoid interference from mocked process.cwd()
  const mlShim = path.resolve(__dirname, '..', 'tools', 'ml-simulate.ts');
  const args = ['-e', `require('ts-node').register(); require('${mlShim}');`, '--', '--pair', pair, '--params', JSON.stringify(params)];
  const env = { ...process.env, QUIET: process.env.QUIET ?? '1' };
  const r = spawnSync('node', args, { encoding: 'utf8', env });
  if (r.status !== 0) return null;
  const stdout = String(r.stdout || '').trim();
  if (!stdout) return null;
  const lines = stdout.split(/\r?\n/).filter(Boolean);
  const jsonLine = [...lines].reverse().find(l => {
    const t = l.trim();
    return (t.startsWith('{') && t.endsWith('}')) ||
           (t.startsWith('[') && t.endsWith(']'));
  }) || lines[lines.length-1];
  try { return JSON.parse(jsonLine); } catch { return null; }
}

// --- I/O utilities ---
/** Async generator to read file line-by-line with low memory overhead. */
export async function* readLines(filePath: string): AsyncGenerator<string> {
  try {
    const rl = readline.createInterface({ input: fs.createReadStream(filePath), crlfDelay: Infinity });
    for await (const line of rl) { yield line as string; }
  } catch { /* ignore */ }
}
