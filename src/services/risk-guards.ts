export interface GuardContext {
  now: number;
  side: "BUY"|"SELL";
  price: number;
  qty: number;
  baseSymbol: string;
  openOrdersCount: number;
  positionQty: number;
  lastExitAt?: number;
}
export interface GuardResult { ok: boolean; reason?: string; }

/**
 * Check if an order can be placed based on risk parameters.
 * 
 * @param {GuardContext} ctx The guard context.
 * @returns {GuardResult} The result of the guard check.
 */
export function canPlaceOrder(ctx: GuardContext): GuardResult {
  const minQty = Number(process.env.RISK_MIN_TRADE_SIZE || "0.0001");
  const maxPos = Number(process.env.MAX_POSITION_SIZE || "0.02");
  const maxNotional = Number(process.env.MAX_ORDER_NOTIONAL_JPY || "100000");
  const cooldownSec = Number(process.env.COOLDOWN_SEC || "10");
  const maxOpen = Number(process.env.MAX_OPEN_ORDERS || "3");

  if (ctx.qty < minQty) return { ok:false, reason:`min trade size ${minQty}` };
  if (ctx.positionQty + (ctx.side==="BUY"?ctx.qty:0) > maxPos) return { ok:false, reason:`max position ${maxPos}` };
  if (ctx.price * ctx.qty > maxNotional) return { ok:false, reason:`max notional ${maxNotional}` };
  if (ctx.openOrdersCount >= maxOpen) return { ok:false, reason:`max open orders ${maxOpen}` };
  if (ctx.lastExitAt && (ctx.now - ctx.lastExitAt) < cooldownSec*1000) return { ok:false, reason:`cooldown ${cooldownSec}s` };
  return { ok:true };
}
