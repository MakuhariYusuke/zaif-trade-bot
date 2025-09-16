// Centralized contracts for PositionStore and RiskManager
// Moved from src/types/contracts.ts

// PositionStore contract interfaces (kept aligned with existing StoredPosition shape)

export interface PositionState {
  pair: string;
  qty: number;
  avgPrice: number;
  dcaCount: number;
  openOrderIds: number[];
  dcaRemainder?: number;
  highestPrice?: number;
  trailArmed?: boolean;
  trailStop?: number;
  lastTrailAt?: number;
  side?: 'long' | 'short';
}

export interface PositionStore {
  load(pair: string): Promise<PositionState>;
  save(pair: string, next: PositionState): Promise<void>;
  update(pair: string, patch: Partial<PositionState>): Promise<PositionState>;
  clear?(pair: string): Promise<void>;
}

// --- Risk contracts ---
export interface RiskError { code: string; message: string }
export interface TrailingAction { side: 'buy'|'sell'; trigger: number }
export interface ClampedIntent { side: 'buy'|'sell'; qty: number; price?: number }

export interface RiskManager {
  // intent shape is exchange-agnostic; implementer interprets it
  validateOrder(intent: any): import('../utils/result').Result<void, RiskError>;
  manageTrailingStop(state: any, price: number): TrailingAction | null;
  clampExposure(balance: any, intent: any): ClampedIntent;
}
