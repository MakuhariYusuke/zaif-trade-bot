import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { logFeatureSample } from '../src/utils/features-logger';

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

  it('writes CSV header and JSON latest', ()=>{
    const s = { ts: Date.now(), pair, side: 'ask' as const, rsi: 60, sma_short: 10, sma_long: 20, price: 100, qty: 0.001, pnl: 1.23, win: true, balance: { jpy: 100000, btc: 0.1 }, bestBid: 99, bestAsk: 101 };
    logFeatureSample(s);
  const csv = fs.readFileSync(path.join(root,'features',pair,`features-${date}.csv`),'utf8');
    const lines = csv.trim().split(/\r?\n/);
    expect(lines[0]).toContain('ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win');
  const latest = JSON.parse(fs.readFileSync(path.join(root,'features',`latest-${pair}.json`),'utf8'));
    expect(latest.pair).toBe(pair);
    expect(latest.rsi).toBe(60);
  });
});
