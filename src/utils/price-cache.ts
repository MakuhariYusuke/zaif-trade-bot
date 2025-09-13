import fs from "fs";
import path from "path";

const CACHE_FILE = process.env.PRICE_CACHE_FILE || path.resolve(process.cwd(), "price_cache.json");
const MAX_ENTRIES = Number(process.env.PRICE_CACHE_MAX || 5000);

export interface PricePoint {
    ts: number;      // ms
    price: number;
}

/**
 * Load the price cache from file.
 * @returns {PricePoint[]} The list of price points.
 */
export function loadPriceCache(): PricePoint[] {
    try {
        if (!fs.existsSync(CACHE_FILE)) return [];
        const arr = JSON.parse(fs.readFileSync(CACHE_FILE, "utf8"));
        if (Array.isArray(arr)) return arr as PricePoint[];
    } catch (err) { console.error("Failed to load price cache:", err); }
    return [];
}

/**
 * Append new price points to the cache, maintaining size limit.
 * @param {PricePoint[]} prices - The new price points to append.
 */
export function appendPriceSamples(prices: PricePoint[]) {
    const cache = loadPriceCache();
    const merged = [...cache, ...prices];
    // sort & dedupe by ts
    const byTs = new Map<number, PricePoint>();
    for (const p of merged) byTs.set(p.ts, p);
    const list = Array.from(byTs.values()).sort((a, b) => a.ts - b.ts);
    if (list.length > MAX_ENTRIES) list.splice(0, list.length - MAX_ENTRIES);
    try { 
        fs.writeFileSync(CACHE_FILE, JSON.stringify(list)); 
    } catch (err) { 
        console.error("Failed to write price cache:", err); 
    }
}

/**
 * Get the most recent N prices from the cache.
 * @param {number} limit - The number of recent prices to retrieve.
 * @returns {number[]} An array of recent prices, most recent first.
 */
export function getPriceSeries(limit: number): number[] {
    const cache = loadPriceCache();
    const sliced = cache.slice(-limit);
    return sliced.map(p => p.price).reverse(); // reverse so that the latest comes first
}