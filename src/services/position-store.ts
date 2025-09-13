import fs from "fs";
import path from "path";
import { logTradeError, logTradeInfo } from "../utils/trade-logger";

export interface StoredPosition {
    pair: string;
    qty: number;           // total base asset quantity (long only)
    avgPrice: number;      // weighted average entry price
    dcaCount: number;      // number of DCA (confirmed fills counted)
    openOrderIds: number[];// associated open order IDs
    dcaRemainder?: number; // remainder below threshold
    highestPrice?: number; // for trailing stop continuation
    trailArmed?: boolean;
    trailStop?: number;
    lastTrailAt?: number;
    side?: 'long' | 'short'; // SELL-first mode uses 'short'
}

export interface FillEvent {
    pair: string; // e.g. "btc_jpy"
    side: "bid" | "ask"; // bid=entry, ask=exit
    price: number; // fill price
    amount: number; // base amount filled
    ts: number; // timestamp ms
    matchMethod?: string; // order_id | heuristic
}

const STORE_FILE = path.resolve(process.cwd(), process.env.POSITION_STORE_FILE || ".positions.store.json");
const DCA_MIN_INCREMENT = Number(process.env.DCA_MIN_INCREMENT || 0.00005);

/**
 * Read the position store from file.
 * @returns {Record<string, StoredPosition>} The position store.
 */
function readStore(): Record<string, StoredPosition> {
    try {
        if (!fs.existsSync(STORE_FILE)) return {};
        return JSON.parse(fs.readFileSync(STORE_FILE, "utf8"));
    } catch (err) {
        logTradeError("Failed to read position store", { error: (err as Error).message });
        return {};
    }
}

let pendingWrite: NodeJS.Timeout | null = null;
let lastData: Record<string, StoredPosition> | null = null;

/** Write the position store to file with debounce.
 * @param {Record<string, StoredPosition>} data The position store data to write.
 */
function writeStore(data: Record<string, StoredPosition>) {
    lastData = data;
    if (pendingWrite) clearTimeout(pendingWrite);
    pendingWrite = setTimeout(() => {
        fs.writeFile(STORE_FILE, JSON.stringify(lastData, null, 2), (err) => {
            if (err) logTradeError("Position store write failed", { error: err.message });
        });
        pendingWrite = null;
    }, 200); // 200ms debounce
}

export function loadPosition(pair: string): StoredPosition | undefined {
    const db = readStore();
    return db[pair];
}

export function savePosition(pos: StoredPosition) {
    const db = readStore();
    db[pos.pair] = pos;
    writeStore(db);
}

export function removePosition(pair: string) {
    const db = readStore();
    delete db[pair];
    writeStore(db);
}

export function updateFields(pair: string, patch: Partial<StoredPosition>) {
    const db = readStore();
    const cur = db[pair];
    if (!cur) return;
    db[pair] = { ...cur, ...patch };
    writeStore(db);
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

/** 
 * Update position based on a fill event.
 * Supports both long-mode (bid to open, ask to close) and short-mode (ask to open, bid to close).
 * @param {FillEvent} fill The fill event.
 */
export function updateOnFill(fill: FillEvent) {
    let pos = loadPosition(fill.pair);
    // Determine if this is a long-mode or short-mode position
    const isShort = pos?.side === 'short';
    // SELL-first mode: ask opens/increases, bid closes/decreases
    if (isShort) {
        if (fill.side === 'ask') {
            // opening / adding to short
            const oldQty = pos!.qty;
            const newQty = oldQty + fill.amount;
            const oldValue = pos!.avgPrice * oldQty;
            const newValue = oldValue + fill.amount * fill.price;
            pos!.qty = newQty;
            pos!.avgPrice = newQty > 0 ? newValue / newQty : 0; // weighted avg sell price
            savePosition(pos!);
        } else if (fill.side === 'bid') {
            if (!pos) return;
            const oldQty = pos.qty;
            const newQty = Math.max(0, oldQty - fill.amount);
            pos.qty = newQty;
            if (newQty === 0) {
                pos.avgPrice = 0;
                pos.dcaRemainder = 0;
            }
            savePosition(pos);
            if (newQty === 0) logTradeInfo('Short position fully closed', { pair: fill.pair });
        }
        return;
    }
    // Long mode (original logic)
    if (fill.side === "bid") {
        if (!pos) {
            pos = { pair: fill.pair, qty: 0, avgPrice: 0, dcaCount: 0, openOrderIds: [] };
        }
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
                logTradeInfo("DCA increment", {
                    pair: fill.pair,
                    oldCount,
                    newCount: pos.dcaCount,
                    increments,
                    fillAmount: fill.amount
                });
            } else {
                pos.dcaRemainder = totalForDca;
            }
        }
        savePosition(pos);
    } else { // exit side ask reducing position
        if (!pos) return; // nothing to reduce
        const oldQty = pos.qty;
        const newQty = Math.max(0, oldQty - fill.amount);
        pos.qty = newQty;
        if (newQty === 0) {
            pos.avgPrice = 0;
            pos.dcaRemainder = 0;
        }
        savePosition(pos);
        if (newQty === 0) {
            logTradeInfo("Position fully closed", { pair: fill.pair });
        }
    }
}

export interface OrderSummaryStats {
    totalTrades: number;
    totalFilledQty: number;
    avgSlippagePct: number;
    maxSlippagePct: number;
}
