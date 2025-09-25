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

export type ErrorEvent = ({ type: 'EVENT/ERROR' } & EventBaseMeta & { code: string });

export type IndicatorEvent = {
  type: 'EVENT/INDICATOR';
  ts: number;
  pair: string;
  snapshot: any;
};

export type TradePlanEvent = {
  type: 'EVENT/TRADE_PLAN';
  ts: number;
  pair: string;
  phase: number;
  plannedOrders: number;
  dryRun: boolean;
};

export type TradePhaseEvent = {
  type: 'EVENT/TRADE_PHASE';
  ts: number;
  pair: string;
  fromPhase: number;
  toPhase: number;
  reason?: string;
};

export type TradeExecutedEvent = {
  type: 'EVENT/TRADE_EXECUTED';
  ts: number;
  pair: string;
  side: 'BUY' | 'SELL';
  qty: number;
  price: number;
  pnl?: number;
  success: boolean;
  requestId: string;
  reason?: ExecutionFailReason;
};

export type ExecutionFailReason = 'MAX_ORDERS' | 'MAX_LOSS' | 'SLIPPAGE';

export type AppEvent = OrderEvent | ErrorEvent | IndicatorEvent | TradePlanEvent | TradePhaseEvent | TradeExecutedEvent;
