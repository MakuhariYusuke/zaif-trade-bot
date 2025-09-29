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
}