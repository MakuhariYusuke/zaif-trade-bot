import fs from 'fs';
import path from 'path';
import { todayStr } from './toolkit';

export interface FeatureSample {
  ts: number;
  pair: string;
  side: 'bid' | 'ask';
  rsi?: number | null;
  sma_short?: number | null;
  sma_long?: number | null;
  price: number;
  qty: number;
  pnl?: number;
  win?: boolean;
  position?: { qty?: number; side?: string; avgPrice?: number };
  balance?: { jpy?: number; btc?: number; eth?: number; xrp?: number };
  bestBid?: number;
  bestAsk?: number;
  spread?: number;    // (ask - bid) / mid
  slippage?: number;  // (fillPrice - mid) / mid
  status?: 'cancelled' | 'failed' | 'filled' | 'partial' | string;
  depthBid?: number;  // best bid size
  depthAsk?: number;  // best ask size
  volumeRecent?: number; // recent traded volume (e.g., last 60s)
}

export function logFeatureSample(s: FeatureSample){
  const date = todayStr();
  const base = process.env.FEATURES_LOG_DIR ? path.resolve(process.env.FEATURES_LOG_DIR) : path.resolve(process.cwd(), 'logs');
  const src = process.env.FEATURES_SOURCE; // optional: 'paper' | 'live'
  const dir = src ? path.join(base, 'features', src, s.pair) : path.join(base, 'features', s.pair);
  try { if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true }); } catch {}
  const csvPath = path.join(dir, `features-${date}.csv`);
  // keep latest JSON at root features for easy consumption
  const jsonPath = path.join(base, 'features', `latest-${s.pair}.json`);
  const header = 'ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win,bal_jpy,bal_btc,bal_eth,bal_xrp,spread,slippage,status,depth_bid,depth_ask,volume_recent';
  // derive spread/slippage if not provided and bestBid/Ask present
  let spread = s.spread;
  let slippage = s.slippage;
  if ((spread == null || slippage == null) && s.bestAsk && s.bestBid){
    const mid = (s.bestAsk + s.bestBid) / 2;
    if (spread == null) spread = (s.bestAsk - s.bestBid) / mid;
    if (slippage == null) slippage = (s.price - mid) / mid;
  }
  const row = [
    s.ts,
    s.pair,
    s.side,
    s.rsi ?? '',
    s.sma_short ?? '',
    s.sma_long ?? '',
    s.price,
    s.qty,
    s.pnl ?? '',
    typeof s.win === 'boolean' ? (s.win ? 1 : 0) : '',
    s.balance?.jpy ?? '',
    s.balance?.btc ?? '',
    s.balance?.eth ?? '',
    s.balance?.xrp ?? '',
    spread ?? '',
    slippage ?? '',
    s.status ?? '',
    s.depthBid ?? '',
    s.depthAsk ?? '',
    s.volumeRecent ?? ''
  ].join(',');
  try {
    if (!fs.existsSync(csvPath)) fs.writeFileSync(csvPath, header + '\n');
    fs.appendFileSync(csvPath, row + '\n');
  } catch {}
  try {
    fs.writeFileSync(jsonPath, JSON.stringify(s, null, 2));
  } catch {}
}
