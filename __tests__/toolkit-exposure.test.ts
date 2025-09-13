import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { computeExposureRatio, getExposureWarnPct, readFeatureCsvRows } from '../src/utils/toolkit';

describe('toolkit exposure helpers', () => {
  const envBackup = { ...process.env };
  beforeEach(() => { process.env = { ...envBackup }; });
  afterEach(() => { process.env = { ...envBackup }; });

  it('getExposureWarnPct respects ENV and clamps to [0,1]', () => {
    delete process.env.EXPOSURE_WARN_PCT;
    expect(getExposureWarnPct()).toBeCloseTo(0.05, 10);
    process.env.EXPOSURE_WARN_PCT = '0.2';
    expect(getExposureWarnPct()).toBeCloseTo(0.2, 10);
    process.env.EXPOSURE_WARN_PCT = '-1';
    expect(getExposureWarnPct()).toBeCloseTo(0.05, 10);
    process.env.EXPOSURE_WARN_PCT = '2';
    expect(getExposureWarnPct()).toBeCloseTo(1.0, 10);
  });

  it('computeExposureRatio returns ratios for bid/ask', () => {
    const funds:any = { jpy: 100000, btc: 2 };
    // bid notional / jpy
    expect(computeExposureRatio('bid', 1, 10000, funds, 'btc_jpy')).toBeCloseTo(0.1, 10);
    // ask qty / base
    expect(computeExposureRatio('ask', 0.5, 10000, funds, 'btc_jpy')).toBeCloseTo(0.25, 10);
  });
});

describe('toolkit readFeatureCsvRows', () => {
  const tmp = path.resolve(process.cwd(), 'tmp-test-ml-exposure');
  beforeEach(() => {
    if (fs.existsSync(tmp)) fs.rmSync(tmp, { recursive: true, force: true });
    fs.mkdirSync(tmp, { recursive: true });
  });
  it('reads rows from feature CSVs', () => {
    const dir = path.join(tmp);
    const hdr = 'ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win';
    const ts = Date.now();
    const csv = [hdr, `${ts},btc_jpy,ask,70,9,26,100,0.001,5,1`].join('\n');
    fs.writeFileSync(path.join(dir, 'features-2025-01-01.csv'), csv);
    const rows = readFeatureCsvRows(dir);
    expect(rows.length).toBe(1);
    expect(rows[0].pair).toBe('btc_jpy');
    expect(rows[0].pnl).toBe(5);
  });
});
