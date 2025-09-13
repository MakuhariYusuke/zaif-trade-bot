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
  const base = baseFromPair(pair);
  if (side === 'bid') {
    const jpy = Number((funds as any).jpy || 0);
    const maxSpend = jpy * 0.10;
    if (maxSpend <= 0 || price <= 0) return amount;
    const maxQty = maxSpend / price;
    return amount > maxQty ? maxQty : amount;
  } else {
    const bal = Number((funds as any)[base] || 0);
    const maxQty = bal * 0.10;
    return amount > maxQty ? maxQty : amount;
  }
}
