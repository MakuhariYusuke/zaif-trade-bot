import { getEventBus } from '../bus';
import { log } from '../../../utils/logger';

export function registerLoggerSubscriber(){
  const bus = getEventBus();
  const logEvent = (ev: any) => {
    try { log('INFO', 'ORDER-EVENT', ev.type, { requestId: ev.requestId, pair: ev.pair, side: ev.side, amount: ev.amount, price: ev.price, orderId: ev.orderId }); } catch {}
  };
  const types = ['ORDER_SUBMITTED','ORDER_FILLED','ORDER_PARTIAL','ORDER_CANCELED','ORDER_EXPIRED','SLIPPAGE_REPRICED'];
  for (const t of types) bus.subscribe(t as any, logEvent as any);
}
