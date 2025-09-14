import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('services/execution-service.onExitFill', () => {
  const TMP = path.resolve(process.cwd(), '.tmp-onExitFill');
  const STATS_DIR = path.join(TMP, 'logs');
  const STORE_FILE = path.join(TMP, 'positions.json');

  beforeEach(() => {
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
    process.env.STATS_DIR = STATS_DIR;
    process.env.POSITION_STORE_FILE = STORE_FILE;
  process.env.POSITION_STORE_DIR = path.join(TMP, 'positions');
  });

  it('appends realized PnL and reduces position qty', async () => {
    const pair = 'btc_jpy';
    const { onExitFill } = await import('../../../src/services/execution-service');
    const { savePosition, loadPosition } = await import('../../../src/services/position-store');
    const { loadDaily } = await import('../../../src/utils/daily-stats');
    // seed a long position: avg 100, qty 1
    savePosition({ pair, qty: 1, avgPrice: 100, dcaCount: 0, openOrderIds: [] });
    // wait for debounced write to flush
    await new Promise(r => setTimeout(r, 300));

  // exit half at 110
  await onExitFill(pair, 110, 0.4);
  await new Promise(r => setTimeout(r, 500));

    // stats check
    const today = new Date().toISOString().slice(0,10);
  const agg = loadDaily(today, pair);
    expect(agg.realizedPnl).toBeCloseTo((110-100)*0.4, 10);
    expect((agg.filledCount||0)).toBeGreaterThan(0);

    // position reduced but avgPrice preserved until zero
  const pos = loadPosition(pair);
    if (!pos) throw new Error(`Position for pair "${pair}" not found after exit fill`);
    expect(pos.qty).toBeCloseTo(1 - 0.4, 10);
    expect(pos.avgPrice).toBe(100);

  // exit rest -> position cleared
  await onExitFill(pair, 90, 0.6);
  await new Promise(r => setTimeout(r, 500));
    const pos2 = loadPosition(pair)!;
    expect(pos2.qty).toBe(0);
    expect(pos2.avgPrice).toBe(0);
  });
});
