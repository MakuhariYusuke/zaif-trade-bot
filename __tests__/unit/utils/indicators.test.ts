import { describe, it, expect } from 'vitest';
import {
    sma,
    ema,
    wma,
    stddev,
    bollinger,
    bbWidth,
    rsi,
    macd,
    envelopes,
    deviationPct,
    roc,
    momentum,
    stochastic,
    ichimoku,
    dmiAdx,
    williamsR,
    cci,
    atr,
    donchianWidth,
    hma,
    kama,
    psarStep,
    fibPosition,
    fibonacciRetracement,
} from '../../../src/utils/indicators';

describe('utils/indicators', () => {
    // --- Moving Averages ---
    describe('Moving Averages', () => {
        const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        it('sma calculates simple moving average', () => {
            expect(sma(values, 5)).toBe(8);
            expect(sma([1, 2, 3, 4, 5], 5)).toBe(3);
            expect(sma(values, 11)).toBeNull();
            expect(sma(values, 0)).toBeNull();
            expect(sma([], 5)).toBeNull();
        });

        it('ema calculates exponential moving average', () => {
            const data = [2, 4, 6, 8, 10, 12];
            const e = ema(data, 5);
            expect(e).not.toBeNull();
            // Test with previous EMA seed yields similar value
            const prevEma = ema(data.slice(0, -1), 5);
            expect(prevEma).not.toBeNull();
            if (prevEma != null) {
                const e2 = ema(data, 5, prevEma)!;
                expect(Math.abs(e2 - (e as number))).toBeLessThan(0.5);
            }
            expect(ema(data, 7)).toBeNull();
            expect(ema([], 5)).toBeNull();
        });

        it('wma calculates weighted moving average', () => {
            expect(wma([1, 2, 3, 4, 5], 5)).toBeCloseTo(3.667, 3);
            expect(wma(values, 5)).toBeCloseTo(8.667, 3);
            expect(wma(values, 11)).toBeNull();
        });

        it('hma calculates Hull moving average', () => {
            const data = Array.from({ length: 30 }, (_, i) => i + 1);
            const h = hma(data, 20);
            expect(h).not.toBeNull();
            // For linear data, HMA should be near the tail
            expect(h!).toBeGreaterThan(28);
            expect(h!).toBeLessThan(30.5);
            expect(hma(data, 31)).toBeNull();
        });

        it('kama calculates Kaufman\'s Adaptive Moving Average', () => {
            const data = [10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14, 15];
            const k = kama(data, 10);
            expect(k).not.toBeNull();
            // In-bounds and stable with prevKama
            const min = Math.min(...data.slice(-10));
            const max = Math.max(...data.slice(-10));
            expect(k!).toBeGreaterThanOrEqual(min);
            expect(k!).toBeLessThanOrEqual(max);
            const prevKama = kama(data.slice(0, -1), 10);
            expect(prevKama).not.toBeNull();
            if (prevKama != null) {
                const k2 = kama(data, 10, 2, 30, prevKama)!;
                expect(Math.abs(k2 - k!)).toBeLessThan(1);
            }
            expect(kama(data, 15)).toBeNull();
        });
    });

    // --- Oscillators & Momentum ---
    describe('Oscillators & Momentum', () => {
        const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 9, 10, 11];

        it('rsi calculates Relative Strength Index', () => {
            const r = rsi(values, 14);
            expect(r.value).not.toBeNull();
            expect(r.value! >= 0 && r.value! <= 100).toBe(true);
            // Test rolling update
            const nextR = rsi([...values, 12], 14, r as any);
            expect(nextR.value).toBeGreaterThan(r.value!);
            expect(rsi(values, 20)).toEqual({ value: null });
        });

        it('macd calculates Moving Average Convergence Divergence', () => {
            const data = Array.from({ length: 40 }, (_, i) => 100 + Math.sin(i / 5) * 10);
            const res = macd(data, 12, 26, 9);
            expect(res.macd).not.toBeNull();
            expect(res.signal).not.toBeNull();
            expect(res.hist).not.toBeNull();
            // Test rolling update
            const prev = macd(data.slice(0, -1), 12, 26, 9);
            const next = macd(data, 12, 26, 9, { emaFast: prev.emaFast, emaSlow: prev.emaSlow, signal: typeof prev.signal === 'number' ? prev.signal : undefined });
            expect(next.macd).toBeCloseTo(res.macd!);
            // implementation differences can seed signal differently; check consistency
            expect(next.signal).not.toBeNull();
            expect(next.hist).not.toBeNull();
            expect(next.hist).toBeCloseTo(next.macd! - (next.signal as number));
        });

        it('stochastic calculates Stochastic Oscillator', () => {
            const h = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 14, 13];
            const l = [9, 10, 11, 10, 9, 8, 9, 10, 11, 12, 13, 14, 13, 12];
            const c = [9.5, 10.5, 11.5, 10.5, 9.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 13.5, 12.5];
            const st = stochastic(h, l, c, 14, 3, 3);
            expect(st.k! >= 0 && st.k! <= 100).toBe(true);
            expect(st.d! >= 0 && st.d! <= 100).toBe(true);
            expect(Math.abs((st.k as number) - (st.d as number))).toBeLessThan(20);
        });

        it('williamsR calculates Williams %R', () => {
            const h = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 14, 13];
            const l = [9, 10, 11, 10, 9, 8, 9, 10, 11, 12, 13, 14, 13, 12];
            const c = [9.5, 10.5, 11.5, 10.5, 9.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 13.5, 12.5];
            const wr = williamsR(h, l, c, 14)!;
            expect(wr <= 0 && wr >= -100).toBe(true);
        });

        it('cci calculates Commodity Channel Index', () => {
            const h = Array.from({ length: 25 }, (_, i) => 100 + i * 0.2);
            const l = Array.from({ length: 25 }, (_, i) => 98 + i * 0.2);
            const c = Array.from({ length: 25 }, (_, i) => 99 + i * 0.2);
            const val = cci(h, l, c, 20)!;
            expect(Number.isFinite(val)).toBe(true);
            expect(Math.abs(val)).toBeLessThan(300);
        });

        it('roc calculates Rate of Change', () => {
            expect(roc([100, 101, 102, 103, 104, 105], 5)).toBeCloseTo(5, 2);
            expect(roc([100], 5)).toBeNull();
        });

        it('momentum calculates momentum', () => {
            expect(momentum([100, 101, 102, 103, 104, 105], 5)).toBe(5);
            expect(momentum([100], 5)).toBeNull();
        });

        it('deviationPct calculates price deviation percentage', () => {
            expect(deviationPct(110, 100)).toBeCloseTo(10, 10);
            expect(deviationPct(95, 100)).toBeCloseTo(-5, 10);
            expect(deviationPct(100, 0)).toBeNull();
            expect(deviationPct(100, null)).toBeNull();
        });
    });

    // --- Volatility & Bands ---
    describe('Volatility & Bands', () => {
        const values = Array.from({ length: 30 }, (_, i) => 100 + Math.sin(i / 3) * 5);
        const h = values.map(v => v + 1);
        const l = values.map(v => v - 1);
        const c = values;

        it('stddev calculates standard deviation', () => {
            expect(stddev([2, 4, 4, 4, 5, 5, 7, 9], 8)).toBeCloseTo(2, 2);
            expect(stddev([1, 1, 1, 1], 4)).toBe(0);
        });

        it('bollinger calculates Bollinger Bands', () => {
            const bb = bollinger(c, 20, 2);
            expect(bb.basis).not.toBeNull();
            expect(bb.upper).not.toBeNull();
            expect(bb.lower).not.toBeNull();
            expect(bb.upper! > bb.basis!).toBe(true);
            expect(bb.lower! < bb.basis!).toBe(true);
        });

        it('bbWidth calculates Bollinger Band Width', () => {
            const width = bbWidth(c, 20, 2);
            expect(width).not.toBeNull();
            expect(width).toBeGreaterThan(0);
        });

        it('bollinger returns nulls when not enough data', () => {
            const bb = bollinger([1, 2, 3], 20, 2);
            expect(bb).toEqual({ basis: null, upper: null, lower: null });
        });

        it('envelopes calculates moving average envelopes', () => {
            const env = envelopes(100, 2.5);
            expect(env.upper!).toBeCloseTo(102.5, 10);
            expect(env.lower!).toBeCloseTo(97.5, 10);
            expect(envelopes(null, 2.5)).toEqual({ upper: null, lower: null });
        });

        it('atr calculates Average True Range', () => {
            const res = atr(h, l, c, 14);
            expect(res).not.toBeNull();
            expect(res).toBeGreaterThan(0);
        });

        it('donchianWidth calculates Donchian Channel width', () => {
            const width = donchianWidth(h, l, c, 20);
            expect(width).not.toBeNull();
            expect(width).toBeGreaterThan(0);
        });

        it('donchianWidth returns null when close is zero', () => {
            const width = donchianWidth([2, 3, 4, 5, 6], [1, 2, 3, 4, 5], [0, 0, 0, 0, 0], 5);
            expect(width).toBeNull();
        });
    });

    // --- Other Indicators ---
    describe('Other Indicators', () => {
        const h = Array.from({ length: 60 }, (_, i) => 100 + i * 0.1 + Math.sin(i / 3));
        const l = Array.from({ length: 60 }, (_, i) => 100 + i * 0.1 - Math.sin(i / 3));
        const c = Array.from({ length: 60 }, (_, i) => 100 + i * 0.1);

        it('ichimoku calculates Ichimoku Cloud values', () => {
            const ich = ichimoku(h, l, c, 9, 26, 52);
            expect(ich.tenkan).not.toBeNull();
            expect(ich.kijun).not.toBeNull();
            expect(ich.spanA).not.toBeNull();
            expect(ich.spanB).not.toBeNull();
            expect(ich.chikou).not.toBeNull();
        });

        it('dmiAdx calculates DMI and ADX', () => {
            const dmi = dmiAdx(h, l, c, 14);
            expect(dmi.adx).not.toBeNull();
            expect(dmi.plusDi).not.toBeNull();
            expect(dmi.minusDi).not.toBeNull();
        });

        it('dmiAdx handles no-move series as zeros', () => {
            const arr = Array.from({ length: 20 }, () => 100);
            const dmi = dmiAdx(arr, arr, arr, 14);
            expect(dmi).toEqual({ adx: 0, plusDi: 0, minusDi: 0 });
        });

        it('psarStep calculates one step of Parabolic SAR', () => {
            const prevUp = { sar: 100, ep: 105, af: 0.02, uptrend: true };
            const nextUp = psarStep(prevUp, 106, 101);
            expect(nextUp.uptrend).toBe(true);
            expect(nextUp.sar).toBeCloseTo(100.1);
            expect(nextUp.ep).toBe(106);
            expect(nextUp.af).toBe(0.04);

            const prevDown = { sar: 110, ep: 105, af: 0.02, uptrend: false };
            const nextDown = psarStep(prevDown, 108, 104);
            expect(nextDown.uptrend).toBe(false);
            expect(nextDown.sar).toBeCloseTo(109.9);
            expect(nextDown.ep).toBe(104);
            expect(nextDown.af).toBe(0.04);
        });

        it('fibPosition calculates Fibonacci position', () => {
            const pos = fibPosition(h, l, c, 50);
            expect(pos).not.toBeNull();
            expect(pos! >= 0 && pos! <= 1).toBe(true);
        });

        it('fibonacciRetracement calculates Fibonacci retracement levels', () => {
            const { levels } = fibonacciRetracement(h, l, 50);
            expect(levels).not.toBeNull();
            const keys = Object.keys(levels!);
            expect(keys).toContain('0.000');
            expect(keys).toContain('0.382');
            expect(keys).toContain('0.618');
            expect(keys).toContain('1.000');
            expect(levels!['0.000'] > levels!['1.000']).toBe(true);
        });
    });

    // --- Edge cases & guards ---
    describe('Edge Cases & Guards', () => {
        it('stochastic returns nulls when insufficient length', () => {
            const st = stochastic([1, 2], [1, 2], [1, 2], 5, 3, 3);
            expect(st).toEqual({ k: null, d: null });
        });

        it('ichimoku returns nulls when insufficient window', () => {
            const ich = ichimoku([1, 2, 3], [1, 2, 3], [1, 2, 3], 9, 26, 52);
            expect(ich).toEqual({ tenkan: null, kijun: null, spanA: null, spanB: null, chikou: 3 });
        });

        it('macd returns nulls before slow+signal seed', () => {
            const data = Array.from({ length: 20 }, (_, i) => i + 1);
            const res = macd(data, 12, 26, 9);
            expect(res.macd).toBeNull();
            expect(res.signal).toBeNull();
            expect(res.hist).toBeNull();
            // Fast EMA may be seeded while slow EMA is not
            expect(res.emaSlow).toBeUndefined();
        });

        it('roc returns null if previous value is zero', () => {
            expect(roc([0, 1], 1)).toBeNull();
        });
    });
});
