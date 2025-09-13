"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vitest_1 = require("vitest");
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const child_process_1 = require("child_process");
const TMP = path_1.default.resolve(process.cwd(), 'tmp-test-ml');
function today() { return new Date().toISOString().slice(0, 10); }
(0, vitest_1.describe)('ml-simulate', () => {
    const pair = 'btc_jpy';
    const date = today();
    (0, vitest_1.beforeEach)(() => {
        if (fs_1.default.existsSync(TMP))
            fs_1.default.rmSync(TMP, { recursive: true, force: true });
        fs_1.default.mkdirSync(path_1.default.join(TMP, 'features', pair), { recursive: true });
        process.env.FEATURES_LOG_DIR = TMP;
    });
    (0, vitest_1.it)('computes winRate and pnl', () => {
        const csvPath = path_1.default.join(TMP, 'features', pair, `features-${date}.csv`);
        const header = 'ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win';
        const now = Date.now();
        const lines = [
            header,
            `${now - 2000},${pair},ask,70,9,26,100,0.001,,`,
            `${now - 1000},${pair},ask,75,9,26,105,0.001,5,1`
        ].join('\n');
        fs_1.default.writeFileSync(csvPath, lines);
        const mlPath = path_1.default.resolve(process.cwd(), 'src', 'tools', 'ml-simulate.ts').replace(/\\/g, '/');
        const cmd = `node -e "require('ts-node').register(); require('${mlPath}');" -- --pair ${pair} --params '{"SELL_RSI_OVERBOUGHT":65,"BUY_RSI_OVERSOLD":25,"SMA_SHORT":9,"SMA_LONG":26}'`;
        const out = (0, child_process_1.execSync)(cmd, { encoding: 'utf8' });
        const res = JSON.parse(out.trim());
        (0, vitest_1.expect)(res.trades).toBe(1);
        (0, vitest_1.expect)(res.winRate).toBe(1);
        (0, vitest_1.expect)(res.pnl).toBe(5);
    });
});
