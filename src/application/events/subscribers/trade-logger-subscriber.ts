import { getEventBus } from '../bus';
import { logTrade } from '../../../utils/trade-logger';

export function registerTradeLoggerSubscriber(){
  const bus = getEventBus();
  const types = ['ORDER_SUBMITTED','ORDER_FILLED','ORDER_PARTIAL','ORDER_CANCELED','ORDER_EXPIRED','SLIPPAGE_REPRICED'];
  for (const t of types){
    bus.subscribe(t as any, (ev: any) => {
      try {
        const safe = { requestId: ev.requestId, pair: ev.pair, side: ev.side, amount: ev.amount, price: ev.price, orderId: ev.orderId, retries: ev.retries, cause: ev.cause };
        logTrade({ ts: new Date().toISOString(), type: 'INFO', message: `[EVENT] ${t}`, data: safe });
      } catch {}
    });
  }
}
