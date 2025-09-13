"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vitest_1 = require("vitest");
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const daily_stats_1 = require("../src/utils/daily-stats");
const TMP = path_1.default.resolve(process.cwd(), 'tmp-test-stats');
function today() { return new Date().toISOString().slice(0, 10); }
(0, vitest_1.describe)('daily-stats', () => {
    const date = today();
    const pair = 'btc_jpy';
    const statsDir = path_1.default.join(TMP, 'logs');
    (0, vitest_1.beforeEach)(() => {
        if (fs_1.default.existsSync(TMP))
            fs_1.default.rmSync(TMP, { recursive: true, force: true });
        fs_1.default.mkdirSync(statsDir, { recursive: true });
        process.env.STATS_DIR = statsDir;
    });
    (0, vitest_1.it)('incBuyEntry and appendFillPnl update JSON', () => {
        const before = (0, daily_stats_1.loadDaily)(date, pair);
        (0, daily_stats_1.incBuyEntry)(date, pair);
        (0, daily_stats_1.appendFillPnl)(date, 123.45, pair);
        const after = (0, daily_stats_1.loadDaily)(date, pair);
        (0, vitest_1.expect)((after.buyEntries || 0)).toBe((before.buyEntries || 0) + 1);
        (0, vitest_1.expect)(after.realizedPnl).toBe((before.realizedPnl || 0) + 123.45);
    });
});
