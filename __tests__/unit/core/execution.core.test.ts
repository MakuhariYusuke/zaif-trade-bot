import { describe, it, expect, beforeEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';
import os from 'os';

import { onExitFill, submitOrderWithRetry } from '../../../src/core/execution';
import { savePosition, loadPosition } from '../../../src/core/position-store';
import { cancelOrder, initMarket } from '../../../src/core/market';
import type { PrivateApi } from '../../../src/types/private';

function tmpDir(prefix: string){
  const d = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  return d;
}

describe('core/execution minimal coverage', () => {
  let tmp: string;
  const pair = 'btc_jpy';

  beforeEach(() => {
    tmp = tmpDir('exec-core-');
    const statsDir = path.join(tmp, 'logs');
    const posDir = path.join(tmp, 'positions');
    process.env.STATS_DIR = statsDir;
    process.env.POSITION_STORE_DIR = posDir;
    if (!fs.existsSync(statsDir)) fs.mkdirSync(statsDir, { recursive: true });
    if (!fs.existsSync(posDir)) fs.mkdirSync(posDir, { recursive: true });
  });

  it('submitOrderWithRetry DRY_RUN returns immediate filled summary', async () => {
    process.env.DRY_RUN = '1';
    const res = await submitOrderWithRetry({
      currency_pair: pair,
      side: 'bid',
      limitPrice: 100,
      amount: 0.1,
      primaryTimeoutMs: 200,
      retryTimeoutMs: 200,
    });
    expect(res.filledQty).toBeCloseTo(0.1);
    expect(res.avgFillPrice).toBeCloseTo(100);
    expect(res.repriceAttempts ?? 0).toBe(0);
  });

  it('onExitFill updates stats and reduces position qty (success path)', async () => {
    // prepare position
    savePosition({ pair, qty: 1, avgPrice: 100, dcaCount: 0, openOrderIds: [], side: 'long' });
    // exit half at profit
    onExitFill(pair, 110, 0.5);
    // position reduced
    const pos = loadPosition(pair)!;
    expect(pos.qty).toBeCloseTo(0.5);
    expect(pos.avgPrice).toBeCloseTo(100);
    // stats file updated
  const today = new Date().toISOString().slice(0,10);
  const statsFile1 = path.join(process.env.STATS_DIR!, `stats-${today}.jsonl`);
  const statsFile2 = path.join(process.env.STATS_DIR!, 'pairs', pair, `stats-${today}.jsonl`);
  // small delay to allow fs write
  await new Promise(r=>setTimeout(r, 10));
  const fpath = fs.existsSync(statsFile2) ? statsFile2 : statsFile1;
  expect(fs.existsSync(fpath)).toBe(true);
  const txt = fs.readFileSync(fpath, 'utf8');
  const last = txt.trim().split(/\r?\n/).filter(Boolean).pop()!;
  const agg = JSON.parse(last);
    // realized PnL increased by (110-100)*0.5=5 and filledCount incremented
    expect(agg.realizedPnl).toBeGreaterThanOrEqual(5 - 1e-9);
    expect(agg.filledCount).toBeGreaterThanOrEqual(1);
  });

  it('cancelOrder throws on API error (propagates)', async () => {
    const mockApi: PrivateApi = {
      get_info2: async () => ({ success: 1, return: { funds: {}, rights: { info: true, trade: true }, open_orders: 0, server_time: Date.now() } }),
      active_orders: async () => [],
      trade_history: async () => [],
      trade: async () => ({ success: 1, return: { order_id: 'OID' } }),
      cancel_order: async () => { throw new Error('cancel failed'); },
    };
    initMarket(mockApi);
    await expect(cancelOrder({ order_id: 123 })).rejects.toThrow('cancel failed');
  });
});
