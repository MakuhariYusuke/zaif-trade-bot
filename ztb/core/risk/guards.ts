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
export interface GuardResult { isAllowed: boolean; reason?: string; }

export function validateOrderPlacement(orderGuard: GuardContext): GuardResult {
  const minQty = Number(process.env.RISK_MIN_TRADE_SIZE || "0.0001");
  const maxPos = Number(process.env.MAX_POSITION_SIZE || "0.02");
  const maxNotional = Number(process.env.MAX_ORDER_NOTIONAL_JPY || "100000");
  const cooldownSec = Number(process.env.COOLDOWN_SEC || "10");
  const maxOpen = Number(process.env.MAX_OPEN_ORDERS || "3");
  if (orderGuard.qty < minQty) return { isAllowed:false, reason:`min trade size ${minQty}` };
  if (orderGuard.positionQty + (orderGuard.side==="BUY"?orderGuard.qty:0) > maxPos) return { isAllowed:false, reason:`max position ${maxPos}` };
  if (orderGuard.price * orderGuard.qty > maxNotional) return { isAllowed:false, reason:`max notional ${maxNotional}` };
  if (orderGuard.openOrdersCount >= maxOpen) return { isAllowed:false, reason:`max open orders ${maxOpen}` };
  if (orderGuard.lastExitAt && (orderGuard.now - orderGuard.lastExitAt) < cooldownSec*1000) return { isAllowed:false, reason:`cooldown ${cooldownSec}s` };
  return { isAllowed:true };
}

export function guardOk(r: GuardResult): boolean { return !!r.isAllowed; }
