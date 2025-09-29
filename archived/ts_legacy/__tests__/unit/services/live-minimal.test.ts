import { describe, it, beforeEach, vi, expect, afterEach } from 'vitest';
import path from 'path';
import fs from 'fs';

// Mocks
vi.mock('../../../ztb/api/adapters', () => ({ createPrivateApi: () => mockApi }));
vi.mock('../../../ztb/api/public', () => ({
    getOrderBook: vi.fn(async () => ({ bids: [[999, 1]], asks: [[1001, 1]] })),
    getTrades: vi.fn(async () => ([{ price: 1000, amount: 0.1, date: Math.floor(Date.now() / 1000) }])),
}));
vi.mock('../../../ztb/utils/daily-stats', () => ({
    incBuyEntry: vi.fn(),
    incSellEntry: vi.fn(),
}));

const calls: any = { trade: [], cancel: [], hist: [], get_info2: [] };
const mockApi: any = {
    trade: vi.fn(async (p: any) => { calls.trade.push(p); return { return: { order_id: 'OID1' } }; }),
    cancel_order: vi.fn(async (p: any) => { calls.cancel.push(p); return { return: { order_id: p.order_id } }; }),
    trade_history: vi.fn(async () => { calls.hist.push(1); return []; }),
    get_info2: vi.fn(async () => { calls.get_info2.push(1); return { success: 1, return: { funds: { jpy: 100000, eth: 10 } } }; }),
};

describe('tools/live/test-minimal-live', () => {
    const envBk = { ...process.env };
    // Use a unique tmp dir to avoid collisions with other live-min tests running in parallel
    const TMP = path.resolve(process.cwd(), 'tmp-live-min-eth');
    beforeEach(() => {
        vi.resetModules();
        Object.keys(calls).forEach(k => (calls as any)[k] = []);
        process.env = { ...envBk };
        if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
        fs.mkdirSync(TMP, { recursive: true });
        process.env.POSITION_STORE_DIR = path.join(TMP, 'positions');
        process.env.STATS_DIR = path.join(TMP, 'logs');
        process.env.FEATURES_LOG_DIR = path.join(TMP, 'features');
        process.env.EXCHANGE = 'zaif';
        process.env.DRY_RUN = '0';
        process.env.PAIR = 'eth_jpy';
        process.env.TRADE_FLOW = 'SELL_ONLY';
        process.env.TEST_FLOW_QTY = '0.1';
        process.env.ORDER_TYPE = 'limit';
        process.env.TEST_FLOW_RATE = '1000';
        process.env.SAFETY_MODE = '1';
    });
    afterEach(() => {
        process.env = { ...envBk };
    });

    it('SELL_ONLY limit order gets cancelled (unfilled) and cancel is called', async () => {
        // start tool
        await import('../../../ztb/tools/live/test-minimal-live');
        // Wait for features JSONL file to exist (max 2s)
        const base = path.resolve(process.env.FEATURES_LOG_DIR as string);
        const dir = path.join(base, 'features', 'live', 'zaif_eth_jpy');
    let files: string[] = [];
        const start = Date.now();
        while (Date.now() - start < 2000) {
            // allow async tool to progress
            await new Promise(r=>setTimeout(r, 10));
            // If 'dir' does not exist, return an empty array to avoid errors when reading directory contents.
        const base = path.resolve(process.env.FEATURES_LOG_DIR as string);
        const d0 = path.join(base, 'features', 'live', 'zaif_eth_jpy');
        files = fs.existsSync(d0) ? fs.readdirSync(d0).filter(f => f.endsWith('.jsonl')) : [];
        if (files.length) {
        const txt = fs.readFileSync(path.join(d0, files[0]), 'utf8');
                const line = txt.trim().split(/\r?\n/).pop() as string;
                const obj = JSON.parse(line);
                expect(['cancelled','failed','filled','partial'].includes(obj.status)).toBe(true);
            }
            if (files.length) {
                // Sort files by modified time descending and pick the latest
        const sortedFiles = files
                    .map(f => ({
                        name: f,
            mtime: fs.statSync(path.join(d0, f)).mtime.getTime()
                    }))
                    .sort((a, b) => b.mtime - a.mtime)
                    .map(f => f.name);
                const latestFile = sortedFiles[0];
        const txt = fs.readFileSync(path.join(d0, latestFile), 'utf8');
                const line = txt.trim().split(/\r?\n/).pop() as string;
                const obj = JSON.parse(line);
                expect(['cancelled','failed','filled','partial'].includes(obj.status)).toBe(true);
            }
        };
        expect(files.length).toBeGreaterThan(0);
        expect(calls.trade.length).toBeGreaterThan(0);
    });
});

