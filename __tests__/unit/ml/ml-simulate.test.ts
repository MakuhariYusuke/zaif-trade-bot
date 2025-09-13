import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

const TMP = path.resolve(process.cwd(), 'tmp-test-ml');

function todayStr(){ return new Date().toISOString().slice(0,10); }

describe('ml-simulate', ()=>{
  const pair = 'btc_jpy';
  const date = todayStr();
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(path.join(TMP,'features',pair), { recursive: true });
    process.env.FEATURES_LOG_DIR = TMP;
  });
  it('computes winRate and pnl', ()=>{
    const csvPath = path.join(TMP,'features',pair,`features-${date}.csv`);
    const header = 'ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win';
    const now = Date.now();
    const csvLines = [
      header,
      `${now-2000},${pair},ask,70,9,26,100,0.001,,`,
      `${now-1000},${pair},ask,75,9,26,105,0.001,5,1`
    ].join('\n');
    fs.writeFileSync(csvPath, csvLines);
    const mlPath = path.resolve(process.cwd(), 'src', 'tools', 'ml-simulate.ts').replace(/\\/g,'/');
    const cmd = `node -e "require('ts-node').register(); require('${mlPath}');" -- --pair ${pair} --params '{"SELL_RSI_OVERBOUGHT":65,"BUY_RSI_OVERSOLD":25,"SMA_SHORT":9,"SMA_LONG":26}'`;
    process.env.QUIET = '1';
    const out = execSync(cmd, { encoding: 'utf8' });
    const stdoutLines = out.trim().split(/\r?\n/).filter(Boolean);
    const jsonLine = [...stdoutLines].reverse().find(l => {
      const t = l.trim();
      return t.startsWith('{') && t.endsWith('}');
    }) || stdoutLines[stdoutLines.length-1];
    const res = JSON.parse(jsonLine);
    expect(res.trades).toBe(1);
    expect(res.winRate).toBe(1);
    expect(res.pnl).toBe(5);
  });

  it('counts trade even when only win flag is present', ()=>{
    const csvPath = path.join(TMP,'features',pair,`features-${date}.csv`);
    const header = 'ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win';
    const now = Date.now();
    const csvLines = [header, `${now-1000},${pair},ask,70,9,26,100,0.001,,1`].join('\n');
    fs.writeFileSync(csvPath, csvLines);
    const mlPath = path.resolve(process.cwd(), 'src', 'tools', 'ml-simulate.ts').replace(/\\/g,'/');
    const cmd = `node -e "require('ts-node').register(); require('${mlPath}');" -- --pair ${pair} --params '{"SELL_RSI_OVERBOUGHT":65,"BUY_RSI_OVERSOLD":25,"SMA_SHORT":9,"SMA_LONG":26}'`;
    process.env.QUIET = '1';
  const out = execSync(cmd, { encoding: 'utf8' });
  const lines = out.trim().split(/\r?\n/).filter(Boolean);
  const jsonLine = [...lines].reverse().find(l => { const t = l.trim(); return t.startsWith('{') && t.endsWith('}'); }) || lines[lines.length-1];
  const res = JSON.parse(jsonLine);
    expect(res.trades).toBe(1);
    expect(res.winRate).toBe(1);
  });

  it('returns zeros when no rows', ()=>{
    const dir = path.join(TMP,'features',pair);
    // ensure empty directory exists
    fs.mkdirSync(dir, { recursive: true });
    const mlPath = path.resolve(process.cwd(), 'src', 'tools', 'ml-simulate.ts').replace(/\\/g,'/');
    const cmd = `node -e "require('ts-node').register(); require('${mlPath}');" -- --pair ${pair} --params '{"SELL_RSI_OVERBOUGHT":65}'`;
    process.env.QUIET = '1';
    const out = execSync(cmd, { encoding: 'utf8' });
    const res = JSON.parse(out.trim());
    expect(res.trades).toBe(0);
    expect(res.winRate).toBe(0);
    expect(res.pnl).toBe(0);
  });
});
