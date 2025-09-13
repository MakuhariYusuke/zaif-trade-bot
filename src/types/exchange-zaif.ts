// Zaif-specific response wire types (legacy/raw)
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
    error?: string }
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
