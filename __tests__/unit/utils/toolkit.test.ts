import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { clampAmountForSafety, computeExposureRatio, getExposureWarnPct, readFeatureCsvRows, getFeaturesRoot } from '../../../src/utils/toolkit';

describe('utils/toolkit', () => {
  const envBk = { ...process.env };
  beforeEach(()=>{ process.env = { ...envBk }; });
  afterEach(()=>{ process.env = { ...envBk }; });

  it('clampAmountForSafety clamps with SAFETY_MODE=1', ()=>{
    process.env.SAFETY_MODE = '1';
    process.env.SAFETY_CLAMP_PCT = '0.1';
    const funds:any = { jpy: 100000, btc: 2 };
    expect(clampAmountForSafety('bid', 20000, 1, funds, 'btc_jpy')).toBe(10000);
    expect(clampAmountForSafety('ask', 1.0, 1_000_000, funds, 'btc_jpy')).toBeCloseTo(0.2, 10);
  });

  it('exposure ratio and warn pct', ()=>{
    delete process.env.EXPOSURE_WARN_PCT;
    expect(getExposureWarnPct()).toBeCloseTo(0.05, 10);
    process.env.EXPOSURE_WARN_PCT = '0.2';
    expect(getExposureWarnPct()).toBeCloseTo(0.2, 10);
    const funds:any = { jpy: 100000, btc: 2 };
    expect(computeExposureRatio('bid', 1, 10000, funds, 'btc_jpy')).toBeCloseTo(0.1, 10);
    expect(computeExposureRatio('ask', 0.5, 10000, funds, 'btc_jpy')).toBeCloseTo(0.25, 10);
  });

  it('readFeatureCsvRows parses CSV files', ()=>{
    const tmp = path.resolve(process.cwd(), 'tmp-test-toolkit');
    if (fs.existsSync(tmp)) fs.rmSync(tmp, { recursive: true, force: true });
    fs.mkdirSync(tmp, { recursive: true });
    const hdr = 'ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win';
    const ts = Date.now();
    fs.writeFileSync(path.join(tmp, 'features-2025-01-01.csv'), [hdr, `${ts},btc_jpy,ask,70,9,26,100,0.001,1,1`].join('\n'));
    const rows = readFeatureCsvRows(tmp);
    expect(rows.length).toBe(1);
    expect(rows[0].win).toBe(1);
  });

  it('readFeatureCsvRows ignores empty/invalid lines and handles missing files', ()=>{
    const tmp = path.resolve(process.cwd(), 'tmp-test-toolkit-empty');
    if (fs.existsSync(tmp)) fs.rmSync(tmp, { recursive: true, force: true });
    fs.mkdirSync(tmp, { recursive: true });
    fs.writeFileSync(path.join(tmp, 'features-2025-01-02.csv'), 'ts,pair,side,price,qty\n\n\n');
    const rows = readFeatureCsvRows(tmp);
    expect(rows.length).toBe(0);
    const rows2 = readFeatureCsvRows(path.join(tmp, 'non-exist'));
    expect(rows2.length).toBe(0);
  });

  it('computeExposureRatio returns 0 when denominator is not positive', ()=>{
    const funds:any = { jpy: 0, btc: 0 };
    expect(computeExposureRatio('bid', 1, 10000, funds, 'btc_jpy')).toBe(0);
    expect(computeExposureRatio('ask', 1, 10000, funds, 'btc_jpy')).toBe(0);
  });

  it('getFeaturesRoot falls back to logs/features when env unset', ()=>{
    delete process.env.FEATURES_LOG_DIR;
    const root = getFeaturesRoot();
    expect(root.toLowerCase()).toContain('logs');
    expect(root.toLowerCase()).toContain('features');
  });
});
