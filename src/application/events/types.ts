export type OrderSide = 'buy' | 'sell';

export interface EventBaseMeta {
  requestId: string;
  pair: string;
  side: OrderSide;
  amount: number;
  price: number;
  orderId?: string;
  eventId?: string;
  retries?: number;
  cause?: { code?: any; message?: string; detail?: any };
}

export type OrderEvent =
  | ({ type: 'ORDER_SUBMITTED' } & EventBaseMeta & { orderId: string })
  | ({ type: 'ORDER_FILLED' } & EventBaseMeta & { orderId: string; filled: number; avgPrice: number })
  | ({ type: 'ORDER_PARTIAL' } & EventBaseMeta & { orderId: string; filled: number; remaining: number })
  | ({ type: 'ORDER_CANCELED' } & EventBaseMeta & { orderId: string })
  | ({ type: 'ORDER_EXPIRED' } & EventBaseMeta & { orderId: string })
  | ({ type: 'SLIPPAGE_REPRICED' } & EventBaseMeta & { orderId: string; attempts: number })
  | ({ type: 'PNL_REALIZED' } & EventBaseMeta & { pnl: number });

export type AppEvent = OrderEvent;
