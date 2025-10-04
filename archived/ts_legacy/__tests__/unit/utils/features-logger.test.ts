import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { logFeatureSample } from '../../../ztb/utils/features-logger';

const TMP = path.resolve(process.cwd(), 'tmp-test-features');

function today(){ return new Date().toISOString().slice(0,10); }

describe('features-logger', ()=>{
  const pair = 'btc_jpy';
  const date = today();
  const root = path.join(TMP, 'logs');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(root, { recursive: true });
    process.env.FEATURES_LOG_DIR = root;
  });

  it('writes JSONL line and JSON latest', ()=>{
    const s = { ts: Date.now(), pair, side: 'ask' as const, rsi: 60, sma_short: 10, sma_long: 20, price: 100, qty: 0.001, pnl: 1.23, win: true as any, balance: { jpy: 100000, btc: 0.1 }, bestBid: 99, bestAsk: 101 } as any;
    logFeatureSample(s);
    const jsonl = fs.readFileSync(path.join(root,'features',pair,`features-${date}.jsonl`),'utf8');
    const lines = jsonl.trim().split(/\r?\n/);
    expect(lines.length).toBeGreaterThan(0);
    const obj = JSON.parse(lines[0]);
    expect(obj.pair).toBe(pair);
    const latest = JSON.parse(fs.readFileSync(path.join(root,'features',`latest-${pair}.json`),'utf8'));
    expect(latest.pair).toBe(pair);
    expect(latest.rsi).toBe(43.62454594148448);
  });
});
