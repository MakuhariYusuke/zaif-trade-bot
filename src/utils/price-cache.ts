import fs from "fs";
import { getEventBus } from "../application/events/bus";
import { buildErrorEventMeta } from "../application/errors";
import path from "path";

let CACHE_FILE = process.env.PRICE_CACHE_FILE || path.resolve(process.cwd(), "price_cache.json");
function getMaxEntries(): number { return Number(process.env.PRICE_CACHE_MAX || 5000); }

export interface PricePoint {
    ts: number;      // ms
    price: number;
}

// In-memory LRU (by ts order) with lazy file load
let loaded = false;
let loadedForFile = CACHE_FILE;
const lru = new Map<number, PricePoint>(); // key: ts

function ensureLoaded(){
    const curr = process.env.PRICE_CACHE_FILE || CACHE_FILE;
    if (curr !== CACHE_FILE){
        // file path changed (e.g., between tests) -> reset
        CACHE_FILE = curr;
        loaded = false;
        lru.clear();
    }
    // If already loaded for this file, but the file was removed externally, reset to avoid stale in-memory state
    if (loaded && loadedForFile === CACHE_FILE) {
        if (!fs.existsSync(CACHE_FILE)) {
            lru.clear();
            loaded = false;
        } else {
            return;
        }
    }
    try {
        if (!fs.existsSync(CACHE_FILE)) { loaded = true; loadedForFile = CACHE_FILE; return; }
        const arr = JSON.parse(fs.readFileSync(CACHE_FILE, "utf8"));
        if (Array.isArray(arr)) {
            const sorted = (arr as PricePoint[]).slice().sort((a,b)=> a.ts - b.ts);
            for (const p of sorted){
                lru.set(p.ts, p);
            }
            // trim to MAX_ENTRIES if oversized
            while (lru.size > getMaxEntries()){
                const k = lru.keys().next().value as number | undefined;
                if (k === undefined) break; lru.delete(k);
            }
        }
    } catch (err) {
        console.error("Failed to load price cache:", err);
        try { getEventBus().publish({ type: 'EVENT/ERROR', code: 'CACHE_ERROR', ...buildErrorEventMeta({ requestId: null, pair: null, side: null, amount: null, price: null }, err) } as any); } catch {}
    } finally { loaded = true; loadedForFile = CACHE_FILE; }
}

function toArray(): PricePoint[] {
    return Array.from(lru.values());
}

/**
 * Load the price cache from file.
 * @returns {PricePoint[]} The list of price points.
 */
export function loadPriceCache(): PricePoint[] {
    ensureLoaded();
    return toArray();
}

/**
 * Append new price points to the cache, maintaining size limit.
 * @param {PricePoint[]} prices - The new price points to append.
 */
export function appendPriceSamples(prices: PricePoint[]) {
    ensureLoaded();
    // merge incoming, keep newest
    for (const p of prices){ lru.set(p.ts, p); }
    // keep only the newest MAX_ENTRIES
    const arr = toArray().sort((a,b)=> a.ts - b.ts);
    const start = Math.max(0, arr.length - getMaxEntries());
    lru.clear();
    for (let i=start; i<arr.length; i++){
        const p = arr[i];
        lru.set(p.ts, p);
    }
    // ensure dir and write synchronously for immediate test visibility
    try {
        const dir = path.dirname(CACHE_FILE);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(CACHE_FILE, JSON.stringify(toArray()));
    } catch (err) {
        console.error("Failed to write price cache:", err);
        try { getEventBus().publish({ type: 'EVENT/ERROR', code: 'CACHE_ERROR', ...buildErrorEventMeta({ requestId: null, pair: null, side: null, amount: null, price: null }, err) } as any); } catch {}
    }
}

/**
 * Get the most recent N prices from the cache.
 * @param {number} limit - The number of recent prices to retrieve.
 * @returns {number[]} An array of recent prices, most recent first.
 */
export function getPriceSeries(limit: number): number[] {
    ensureLoaded();
    let arr: PricePoint[] = toArray().slice();
    // ensure order by ts ascending, then take from end so newest-first is stable
    const out: number[] = [];
    arr.sort((a,b)=> a.ts - b.ts);
    for (let i=arr.length-1; i>=0 && out.length<limit; i--){
        out.push(arr[i].price);
    }
    return out;
}

/**
 * Test helper: reset the in-memory price cache and loaded flags.
 * Next operation will lazy-load from current PRICE_CACHE_FILE.
 */
export function resetPriceCache(){
    try { lru.clear(); } catch {}
    loaded = false;
    loadedForFile = process.env.PRICE_CACHE_FILE || CACHE_FILE;
}