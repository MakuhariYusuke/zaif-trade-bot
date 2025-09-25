// Centralized private API & adapter types

export type Side = "BUY" | "SELL";
export interface OrderLifecycleSummary {
    requestId: string;
    side: Side;
    intendedQty: number;
    filledQty: number;
    avgExpectedPrice: number;
    avgFillPrice: number;
    slippagePct: number;
    durationMs: number;
    submitRetryCount: number;
    pollRetryCount: number;
    cancelRetryCount: number;
    nonceRetryCount: number;
    totalRetryCount: number;
    filledCount: number;
    repriceAttempts?: number;
    orderId?: string;
}

export interface GetInfo2Response { 
    success: 0 | 1;
    return?: { 
        funds: Record<string, number>; 
        rights: { 
            info: boolean; 
            trade: boolean; 
            withdraw?: boolean 
        }; 
        open_orders: number; 
        server_time: number 
    }; 
    error?: string 
}
export interface ActiveOrder { 
    order_id: string; 
    pair: string; 
    side: "bid" | "ask"; 
    price: number; 
    amount: number; 
    timestamp: number 
}
export interface TradeHistoryRecord { 
    tid: number; 
    order_id?: string; 
    side: "bid" | "ask"; 
    price: number; 
    amount: number; 
    timestamp: number 
}
export interface TradeResult { 
    success: 1; 
    return: { order_id: string } 
}
export interface CancelResult { 
    success: 1; 
    return: { order_id: string } 
}

export type TradeMode = "SELL" | "BUY";
export type TradeFlow = "BUY_ONLY" | "SELL_ONLY" | "BUY_SELL" | "SELL_BUY";

export interface PrivateApi {
    get_info2(): Promise<GetInfo2Response>;
    active_orders(p?: any): Promise<ActiveOrder[]>;
    trade_history(p?: any): Promise<TradeHistoryRecord[]>;
    trade(p: any): Promise<TradeResult>;
    cancel_order(p: { order_id: string }): Promise<CancelResult>;
    healthCheck?(): Promise<any>;
    testGetInfo2?(): Promise<any>;
}

export interface ZaifApiConfig {
    key: string;
    secret: string;
    timeoutMs?: number;
    nonceStorePath?: string;
}

// Legacy compatibility types (will be phased out)
export type OrderType = "bid" | "ask";
export interface ZaifBalanceResponse { 
    success: number; 
    return?: { 
        funds: Record<string, number>; 
        rights?: Record<string, number>; 
        open_orders?: number; 
        server_time?: number 
    }; 
    error?: string 
}
export interface ZaifTradeResponse { 
    success: number; 
    return?: { 
        received: number; 
        remains: number; 
        order_id: number; 
        funds: Record<string, number> 
    }; 
    error?: string 
}
export interface ZaifActiveOrdersResponse { 
    success: number; 
    return?: Record<string, { 
        currency_pair: string; 
        action: OrderType; 
        amount: number; 
        price: number; 
        timestamp: number 
    }>; 
    error?: string 
}
export interface ZaifCancelOrderResponse { 
    success: number; 
    return?: { 
        funds?: Record<string, number>; 
        order_id: number 
    }; 
    error?: string 
}
export interface ZaifTradeHistoryResponse { 
    success: number; 
    return?: Array<{ 
        date: number; 
        price: number; 
        amount: number; 
        tid: number; 
        currency_pair: string; 
        trade_type: OrderType 
    }>; 
    error?: string 
}
export interface PlaceOrderParams { 
    currency_pair: string; 
    action: OrderType; 
    price: number; 
    amount: number 
}
export interface CancelOrderParams { 
    order_id: number 
}

