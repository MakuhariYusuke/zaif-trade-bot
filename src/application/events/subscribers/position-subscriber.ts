import { getEventBus } from '../bus';
import { updateOnFill, clearOpenOrderId, addOpenOrderId } from '../../../adapters/position-store';
import { log } from '../../../utils/logger';

export function registerPositionSubscriber(){
  const bus = getEventBus();
  bus.subscribe('ORDER_SUBMITTED' as any, (ev: any) => {
    try { if (ev.orderId) addOpenOrderId(ev.pair, Number(ev.orderId)); } catch {}
  });
  bus.subscribe('ORDER_FILLED' as any, (ev: any) => {
    try {
      const side = ev.side === 'buy' ? 'bid' : 'ask';
      updateOnFill({ pair: ev.pair, side, price: ev.avgPrice ?? ev.price, amount: ev.filled ?? ev.amount, ts: Date.now(), matchMethod: 'event' });
      if (ev.orderId) try { clearOpenOrderId(ev.pair, Number(ev.orderId)); } catch {}
    } catch {}
  });
  bus.subscribe('ORDER_CANCELED' as any, (ev: any) => {
    try { if (ev.orderId) clearOpenOrderId(ev.pair, Number(ev.orderId)); } catch {}
  });
  bus.subscribe('ORDER_EXPIRED' as any, (ev: any) => {
    try { if (ev.orderId) clearOpenOrderId(ev.pair, Number(ev.orderId)); } catch {}
  });
  bus.subscribe('SLIPPAGE_REPRICED' as any, (ev: any) => {
    try { log('INFO','ORDER-EVENT','SLIPPAGE_REPRICED',{ requestId: ev.requestId, pair: ev.pair, side: ev.side, amount: ev.amount, price: ev.price, orderId: ev.orderId, attempts: ev.attempts }); } catch {}
  });
}
