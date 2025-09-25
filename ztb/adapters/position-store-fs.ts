import fs from "fs";
import path from "path";
import { logTradeError, logTradeInfo } from "../utils/trade-logger";
import { writeFileAtomic } from "../utils/fs-atomic";
import type { PositionStore as IPositionStore, PositionState } from "@contracts";

export interface StoredPosition { pair: string; qty: number; avgPrice: number; dcaCount: number; openOrderIds: number[]; dcaRemainder?: number; highestPrice?: number; trailArmed?: boolean; trailStop?: number; lastTrailAt?: number; side?: 'long' | 'short'; }
export interface FillEvent { pair: string; side: 'bid' | 'ask'; price: number; amount: number; ts: number; matchMethod?: string; }

function getStoreFile(){ return path.resolve(process.cwd(), process.env.POSITION_STORE_FILE || ".positions.store.json"); }
function getStoreDir(){ return path.resolve(process.cwd(), process.env.POSITION_STORE_DIR || ".positions"); }
const DCA_MIN_INCREMENT = Number(process.env.DCA_MIN_INCREMENT || 0.00005);
function pairFile(pair: string) { return path.join(getStoreDir(), `${pair}.json`); }
function readLegacyStore(): Record<string, StoredPosition> {
  const f = getStoreFile();
  try { if (!fs.existsSync(f)) return {}; return JSON.parse(fs.readFileSync(f, 'utf8')); } catch { return {}; }
}
function loadFromPerPair(pair: string): StoredPosition | undefined {
  try { const f = pairFile(pair); if (!fs.existsSync(f)) return undefined; return JSON.parse(fs.readFileSync(f,'utf8')); } catch { return undefined; }
}
function savePerPair(pos: StoredPosition) {
  try {
    const file = pairFile(pos.pair);
    writeFileAtomic(file, JSON.stringify(pos, null, 2));
  } catch (err:any) {
    logTradeError('Position store write failed', { error: err?.message||String(err) });
  }
}
export function loadPosition(pair: string) {
  const per = loadFromPerPair(pair); if (per) return per;
  const legacy = readLegacyStore(); return legacy[pair];
}
export function savePosition(pos: StoredPosition) { savePerPair(pos); }
export function removePosition(pair: string) {
  try { const f = pairFile(pair); if (fs.existsSync(f)) fs.unlinkSync(f); } catch (err:any) { logTradeError('Position store remove failed', { error: err?.message||String(err) }); }
}
export function updatePositionFields(pair: string, patch: Partial<StoredPosition>) {
  const cur = loadPosition(pair);
  if (!cur) return;
  const next = { ...cur, ...patch } as StoredPosition; savePerPair(next);
}
export function addOpenOrderId(pair: string, orderId: number) {
  const p = loadPosition(pair) || { pair, qty: 0, avgPrice: 0, dcaCount: 0, openOrderIds: [] } as StoredPosition;
  if (!p.openOrderIds.includes(orderId)) p.openOrderIds.push(orderId);
  savePosition(p);
}
export function clearOpenOrderId(pair: string, orderId: number) {
  const p = loadPosition(pair);
  if (!p) return;
  p.openOrderIds = p.openOrderIds.filter(id => id !== orderId);
  savePosition(p);
}
export function updateOnFill(fill: FillEvent) {
  let pos = loadPosition(fill.pair);
  const isShort = pos?.side === 'short';
  if (isShort) {
    if (fill.side === 'ask') {
      const oldQty = pos!.qty;
      const newQty = oldQty + fill.amount;
      const oldValue = pos!.avgPrice * oldQty;
      const newValue = oldValue + fill.amount * fill.price;
      pos!.qty = newQty;
      pos!.avgPrice = newQty > 0 ? newValue / newQty : 0;
      savePosition(pos!);
    } else if (fill.side === 'bid') {
      if (!pos) return;
      const oldQty = pos.qty;
      const newQty = Math.max(0, oldQty - fill.amount);
      pos.qty = newQty;
      if (newQty === 0) { pos.avgPrice = 0; pos.dcaRemainder = 0; }
      savePosition(pos);
      if (newQty === 0) logTradeInfo('Short position fully closed', { pair: fill.pair });
    }
    return;
  }
  if (fill.side === 'bid') {
    if (!pos) { pos = { pair: fill.pair, qty: 0, avgPrice: 0, dcaCount: 0, openOrderIds: [] }; }
    const oldQty = pos.qty;
    const newQty = oldQty + fill.amount;
    const oldValue = pos.avgPrice * oldQty;
    const newValue = oldValue + fill.amount * fill.price;
    pos.qty = newQty;
    pos.avgPrice = newQty > 0 ? newValue / newQty : 0;
    if (oldQty > 0) {
      const totalForDca = (pos.dcaRemainder || 0) + fill.amount;
      const increments = Math.floor(totalForDca / DCA_MIN_INCREMENT);
      if (increments > 0) {
        const oldCount = pos.dcaCount;
        pos.dcaCount += increments;
        pos.dcaRemainder = totalForDca - increments * DCA_MIN_INCREMENT;
        logTradeInfo('DCA increment', { pair: fill.pair, oldCount, newCount: pos.dcaCount, increments, fillAmount: fill.amount });
      } else {
        pos.dcaRemainder = totalForDca;
      }
    }
    savePosition(pos);
  } else {
    if (!pos) return;
    const oldQty = pos.qty;
    const newQty = Math.max(0, oldQty - fill.amount);
    pos.qty = newQty;
    if (newQty === 0) { pos.avgPrice = 0; pos.dcaRemainder = 0; }
    savePosition(pos);
    if (newQty === 0) logTradeInfo('Position fully closed', { pair: fill.pair });
  }
}

function defaultState(pair: string): StoredPosition {
  return { pair, qty: 0, avgPrice: 0, dcaCount: 0, openOrderIds: [] };
}

export class CorePositionStore implements IPositionStore {
  async load(pair: string): Promise<PositionState> {
    const cur = loadPosition(pair);
    return (cur ?? defaultState(pair)) as PositionState;
  }
  async save(_pair: string, next: PositionState): Promise<void> { savePosition(next as unknown as StoredPosition); }
  async update(pair: string, patch: Partial<PositionState>): Promise<PositionState> {
    const cur = (await this.load(pair)) as StoredPosition;
    const next = { ...cur, ...patch } as StoredPosition;
    savePosition(next);
    return next as unknown as PositionState;
  }
  async clear(pair: string): Promise<void> { removePosition(pair); }
}

export function createServicePositionStore(): IPositionStore { return new CorePositionStore(); }
