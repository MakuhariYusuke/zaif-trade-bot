/** @deprecated 次メジャーで core に統合予定 */
import {
    loadPosition as fsLoad,
    savePosition as fsSave,
    removePosition as fsRemove,
    updateOnFill as fsUpdateOnFill,
    addOpenOrderId as fsAddOpenOrderId,
    clearOpenOrderId as fsClearOpenOrderId,
    CorePositionStore
} from "./position-store-fs";
import type { PositionStore as IPositionStore } from "@contracts";
export type { PositionStore as PositionStoreContract } from "@contracts";

// Re-export legacy types for compatibility
export interface StoredPosition {
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
export interface FillEvent { pair: string; side: 'bid' | 'ask'; price: number; amount: number; ts: number; matchMethod?: string; }

// Thin adapter delegating to core implementation; public API kept identical
export function loadPosition(pair: string) { return fsLoad(pair); }
export function savePosition(pos: StoredPosition) { return fsSave(pos as any); }
export function removePosition(pair: string) { return fsRemove(pair); }
export function updateFields(pair: string, patch: Partial<StoredPosition>) {
    const cur = fsLoad(pair) || { pair, qty: 0, avgPrice: 0, dcaCount: 0, openOrderIds: [] } as StoredPosition;
    return fsSave({ ...(cur as any), ...patch });
}
export function addOpenOrderId(pair: string, orderId: number) { return fsAddOpenOrderId(pair, orderId); }
export function clearOpenOrderId(pair: string, orderId: number) { return fsClearOpenOrderId(pair, orderId); }
export function updateOnFill(fill: FillEvent) { return fsUpdateOnFill(fill as any); }

// Factory for contract consumers (optional)
export function createServicePositionStore(): IPositionStore { return new CorePositionStore(); }

export interface OrderSummaryStats { totalTrades: number; totalFilledQty: number; avgSlippagePct: number; maxSlippagePct: number; }
