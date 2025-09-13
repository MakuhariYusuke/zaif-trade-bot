export interface CoincheckBalanceResponse {
  success: boolean;
  jpy: string;
  btc: string;
  // Other currencies and reserved/debt fields can be added as needed
  [key: string]: any;
}

export interface CoincheckOrderRequest {
  pair: string;     // "btc_jpy"
  order_type: "buy" | "sell";
  rate: number;
  amount: number;
}

export interface CoincheckOrderResponse {
  success: boolean;
  id: number;
  rate: string;
  amount: string;
  order_type: string;
  pair: string;
  created_at: string;
}

export interface CoincheckCancelResponse { success: boolean; id: number; }

export interface CoincheckOpenOrdersResponse {
  success: boolean;
  orders: Array<{ id: number; order_type: string; rate: string; pending_amount: string; created_at: string }>;
}

export interface CoincheckTradeHistoryResponse {
  success: boolean;
  trades: Array<{ id: number; order_id: number; rate: string; amount: string; order_type: string; created_at: string }>;
}
