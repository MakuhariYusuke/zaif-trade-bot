import type { PrivateApi } from "../types/private";

/** Return YYYY-MM-DD for given date (defaults to now). */
export function todayStr(d: Date = new Date()): string { return d.toISOString().slice(0, 10); }

/** Sleep helper for tools. */
export function sleep(ms: number): Promise<void> { return new Promise(r => setTimeout(r, ms)); }

/** Extract base asset from pair like "btc_jpy" -> "btc". */
export function baseFromPair(pair: string): string { return (pair || 'btc_jpy').split('_')[0]; }

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
    return amount > maxQty ? maxQty : amount;
  } else {
    const bal = Number((funds as any)[base] || 0);
    const maxQty = bal * pct;
    return amount > maxQty ? maxQty : amount;
  }
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
export function readFeatureCsvRows(dir: string): FeatureCsvRow[] {
  const fs = require('fs') as typeof import('fs');
  const path = require('path') as typeof import('path');
  const rows: FeatureCsvRow[] = [];
  const files = fs.existsSync(dir) ? fs.readdirSync(dir).filter((f: string) => f.endsWith('.csv')) : [];
  for (const f of files) {
    const full = path.join(dir, f);
    try {
      const txt = fs.readFileSync(full, 'utf8');
      const [header, ...lines] = txt.trim().split(/\r?\n/);
      const cols = header.split(',');
      for (const line of lines) {
        if (!line.trim()) continue;
        const parts = line.split(',');
        const rec: any = {};
        cols.forEach((c, i) => rec[c] = parts[i]);
        rows.push({
          ts: Number(rec.ts), pair: rec.pair, side: rec.side,
          rsi: rec.rsi ? Number(rec.rsi) : undefined,
          sma_short: rec.sma_short ? Number(rec.sma_short) : undefined,
          sma_long: rec.sma_long ? Number(rec.sma_long) : undefined,
          price: Number(rec.price), qty: Number(rec.qty),
          pnl: rec.pnl ? Number(rec.pnl) : undefined,
          win: rec.win ? Number(rec.win) : undefined,
        });
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

/** Run ml-simulate (ts) with given params and pair, returning parsed JSON result or null on failure. */
export function runMlSimulate(params: Record<string, any>, pair: string): any | null {
  const { spawnSync } = require('child_process') as typeof import('child_process');
  const path = require('path') as typeof import('path');
  // Use top-level shim for portability
  const mlShim = path.resolve(process.cwd(), 'src', 'tools', 'ml-simulate.ts');
  const args = ['-e', `require('ts-node').register(); require('${mlShim.replace(/\\/g,'/')}');`, '--', '--pair', pair, '--params', JSON.stringify(params)];
  const env = { ...process.env, QUIET: process.env.QUIET ?? '1' };
  const r = spawnSync('node', args, { encoding: 'utf8', env });
  if (r.status !== 0) return null;
  const stdout = String(r.stdout || '').trim();
  if (!stdout) return null;
  const lines = stdout.split(/\r?\n/).filter(Boolean);
  const jsonLine = [...lines].reverse().find(l => l.trim().startsWith('{') && l.trim().endsWith('}')) || lines[lines.length-1];
  try { return JSON.parse(jsonLine); } catch { return null; }
}
