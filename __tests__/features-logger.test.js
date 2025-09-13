"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vitest_1 = require("vitest");
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const features_logger_1 = require("../src/utils/features-logger");
const TMP = path_1.default.resolve(process.cwd(), 'tmp-test-features');
function today() { return new Date().toISOString().slice(0, 10); }
(0, vitest_1.describe)('features-logger', () => {
    const pair = 'btc_jpy';
    const date = today();
    const root = path_1.default.join(TMP, 'logs');
    (0, vitest_1.beforeEach)(() => {
        if (fs_1.default.existsSync(TMP))
            fs_1.default.rmSync(TMP, { recursive: true, force: true });
        fs_1.default.mkdirSync(root, { recursive: true });
        process.env.FEATURES_LOG_DIR = root;
    });
    (0, vitest_1.it)('writes CSV header and JSON latest', () => {
        const s = { ts: Date.now(), pair, side: 'ask', rsi: 60, sma_short: 10, sma_long: 20, price: 100, qty: 0.001, pnl: 1.23, win: true, balance: { jpy: 100000, btc: 0.1 }, bestBid: 99, bestAsk: 101 };
        (0, features_logger_1.logFeatureSample)(s);
        const csv = fs_1.default.readFileSync(path_1.default.join(root, 'features', pair, `features-${date}.csv`), 'utf8');
        const lines = csv.trim().split(/\r?\n/);
        (0, vitest_1.expect)(lines[0]).toContain('ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win');
        const latest = JSON.parse(fs_1.default.readFileSync(path_1.default.join(root, 'features', `latest-${pair}.json`), 'utf8'));
        (0, vitest_1.expect)(latest.pair).toBe(pair);
        (0, vitest_1.expect)(latest.rsi).toBe(60);
    });
});
