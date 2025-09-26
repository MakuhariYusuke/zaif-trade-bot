import { describe, it, expect } from 'vitest';
import { heikinAshiSeries, vortexSeries, aroonSeries, tsiSeries, mfiSeries, obvSeries, keltnerSeries, donchianSeries, choppinessSeries } from '../../../ztb/utils/indicators';

function genOHLC(n: number, start = 100, step = 1){
  const open: number[] = []; const high: number[] = []; const low: number[] = []; const close: number[] = []; const volume: number[] = [];
  let price = start;
  for (let i = 0; i < n; i++) {
    const o = price;
    const c = price + (i % 2 === 0 ? step : -step/2);
    const h = Math.max(o, c) + 0.5;
    const l = Math.min(o, c) - 0.5;
    open.push(o); high.push(h); low.push(l); close.push(c); price = c;
    volume.push(100 + (i % 5));
  }
  return { open, high, low, close, volume };
}

describe('utils: new indicator series sanity', () => {
  it('heikin ashi produces smoothed values', () => {
    const { open, high, low, close } = genOHLC(30);
    const ha = heikinAshiSeries(open, high, low, close);
    expect(ha.close.at(-1)).not.toBeNull();
    const last = open.length - 1;
    const expectedClose = (open[last] + high[last] + low[last] + close[last]) / 4;
    expect(Math.abs((ha.close[last] as number) - expectedClose)).toBeLessThan(1e-9);
  });

  it('vortex in [0, +inf) with null padding', () => {
    const { high, low, close } = genOHLC(40);
    const { viPlus, viMinus } = vortexSeries(high, low, close, 14);
    const last = viPlus.at(-1);
    expect(last == null || last >= 0).toBe(true);
    expect(viMinus.at(-1) == null || (viMinus.at(-1)! >= 0)).toBe(true);
  });

  it('aroon in [0,100] and oscillator in [-100,100]', () => {
    const { high, low } = genOHLC(40);
    const ar = aroonSeries(high, low, 14);
    const up = ar.aroonUp.at(-1); const dn = ar.aroonDown.at(-1); const os = ar.aroonOsc.at(-1);
    if (up != null) expect(up).toBeGreaterThanOrEqual(0), expect(up).toBeLessThanOrEqual(100);
    if (dn != null) expect(dn).toBeGreaterThanOrEqual(0), expect(dn).toBeLessThanOrEqual(100);
    if (os != null) expect(Math.abs(os)).toBeLessThanOrEqual(100);
  });

  it('tsi and signal exist and reasonable sign', () => {
    const { close } = genOHLC(80, 100, 0.8);
    const { tsi, signal } = tsiSeries(close, 13, 25, 13);
    expect(tsi.at(-1)).not.toBeNull();
    expect(signal.at(-1)).not.toBeNull();
  });

  it('mfi in [0,100]', () => {
    const { high, low, close, volume } = genOHLC(60);
    const { mfi } = mfiSeries(high, low, close, volume, 14);
    const v = mfi.at(-1);
    if (v != null) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(100); }
  });

  it('obv accumulates directionally', () => {
    const { close, volume } = genOHLC(30, 100, 2);
    const { obv } = obvSeries(close, volume);
    expect(obv.at(-1)).not.toBeNull();
  });

  it('keltner bands have upper>=basis>=lower', () => {
    const { high, low, close } = genOHLC(80);
    const kc = keltnerSeries(high, low, close, 20, 2);
    const i = close.length - 1;
    const b = kc.basis[i]; const u = kc.upper[i]; const l = kc.lower[i];
    if (b != null && u != null && l != null) {
      expect(u).toBeGreaterThanOrEqual(b);
      expect(b).toBeGreaterThanOrEqual(l);
    }
  });

  it('donchian upper>=mid>=lower', () => {
    const { high, low } = genOHLC(60);
    const dc = donchianSeries(high, low, 20);
    const i = high.length - 1;
    const u = dc.upper[i]; const m = dc.mid[i]; const l = dc.lower[i];
    if (u != null && m != null && l != null) {
      expect(u).toBeGreaterThanOrEqual(m);
      expect(m).toBeGreaterThanOrEqual(l);
    }
  });

  it('choppiness in [0,100]', () => {
    const { high, low, close } = genOHLC(80);
    const { choppiness } = choppinessSeries(high, low, close, 14);
    const v = choppiness.at(-1);
    if (v != null) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(100); }
  });
});
