// Pure indicator functions (no side effects)
// Lightweight implementations with optional previous-state to enable rolling updates.

export type Num = number | null;

// ---- Series ----

import { wilderSmooth } from './utils';

/**
 * Simple memoization helper for expensive computations
 */
function memoize<T extends (...args: any[]) => any>(fn: T, keyFn?: (...args: Parameters<T>) => string): T {
  const cache = new Map<string, ReturnType<T>>();
  return ((...args: Parameters<T>) => {
    const key = keyFn ? keyFn(...args) : JSON.stringify(args);
    if (cache.has(key)) return cache.get(key)!;
    const result = fn(...args);
    cache.set(key, result);
    return result;
  }) as T;
}

/**
 * Average True Range (ATR) Series
 * Calculates the ATR over a specified period for each point in the series.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param period - The lookback period for ATR calculation (default: 14).
 * @returns An array containing the ATR values, with nulls for initial periods without enough data.
 */
function _atrSeries(high: number[], low: number[], close: number[], period: number): Array<number | null> {
  const n = close.length;
  const out: Array<number | null> = new Array(n).fill(null);
  if (n <= period) return out;
  const trArr: number[] = new Array(n - 1);
  for (let i = 1; i < n; i++) {
    trArr[i - 1] = Math.max(
      high[i] - low[i],
      Math.abs(high[i] - close[i - 1]),
      Math.abs(low[i] - close[i - 1])
    );
  }
  let currentAtr = trArr.slice(0, period).reduce((a, b) => a + b, 0) / period;
  out[period] = currentAtr;
  for (let i = period; i < trArr.length; i++) {
    currentAtr = wilderSmooth(currentAtr, trArr[i], period);
    out[i + 1] = currentAtr;
  }
  return out;
}

export const atrSeries = memoize(_atrSeries, (high, low, close, period) => `${high.length}-${low.length}-${close.length}-${period}-${high[0]}-${close[close.length-1]}`);

/**
 * Heikin-Ashi Candles
 * Transforms standard OHLC data into Heikin-Ashi format.
 * @param open - Array of open prices.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @returns An object containing arrays for Heikin-Ashi open, high, low, and close prices.
 */
export function heikinAshiSeries(open: number[], high: number[], low: number[], close: number[]): { open: Array<number | null>; high: Array<number | null>; low: Array<number | null>; close: Array<number | null> } {
  if (
    open.length !== high.length ||
    open.length !== low.length ||
    open.length !== close.length
  ) {
    throw new Error('Input arrays for Heikin-Ashi must have the same length.');
  }
  const n = open.length;
  const haO: Array<number | null> = new Array(n).fill(null);
  const haH: Array<number | null> = new Array(n).fill(null);
  const haL: Array<number | null> = new Array(n).fill(null);
  const haC: Array<number | null> = new Array(n).fill(null);
  if (n === 0) return { open: haO, high: haH, low: haL, close: haC };
  for (let i = 0; i < n; i++) {
    const o = open[i], h = high[i], l = low[i], c = close[i];
    const cAvg = (o + h + l + c) / 4;
    haC[i] = cAvg;
    if (i === 0) {
      haO[i] = (o + c) / 2;
    } else {
      const prevO = haO[i - 1] as number;
      const prevC = haC[i - 1] as number;
      haO[i] = (prevO + prevC) / 2;
    }
    haH[i] = Math.max(h, haO[i] as number, haC[i] as number);
    haL[i] = Math.min(l, haO[i] as number, haC[i] as number);
  }
  return { open: haO, high: haH, low: haL, close: haC };
}

export function keltnerSeries(high: number[], low: number[], close: number[], period = 20, mult = 2): { basis: Array<number | null>; upper: Array<number | null>; lower: Array<number | null> } {
  const n = close.length;
  const basis = rollingEma(close, period);
  const atrArr = atrSeries(high, low, close, period);
  const upper: Array<number | null> = new Array(n).fill(null);
  const lower: Array<number | null> = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    const b = basis[i];
    const a = atrArr[i];
    if (b == null || a == null) continue;
    upper[i] = b + mult * a;
    lower[i] = b - mult * a;
  }
  return { basis, upper, lower };
}

/**
 * Donchian Channel
 * Calculates the Donchian channel (upper, lower, mid) over a specified period.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param period - The lookback period for Donchian channel calculation (default: 20).
 * @returns An object containing arrays for the upper, lower, and mid Donchian channels.
 */
export function donchianSeries(high: number[], low: number[], period = 20): { upper: Array<number | null>; lower: Array<number | null>; mid: Array<number | null> } {
  const n = Math.min(high.length, low.length);
  const upper = rollingMax(high, period);
  const lower = rollingMin(low, period);
  const mid: Array<number | null> = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    if (upper[i] != null && lower[i] != null) mid[i] = ((upper[i] as number) + (lower[i] as number)) / 2;
  }
  return { upper, lower, mid };
}

/**
 * MACD (Moving Average Convergence Divergence) indicator
 * @param values
 * @param fast
 * @param slow
 * @param signal
 * @returns
 */
export function macdSeries(values: number[], fast = 12, slow = 26, signal = 9): { macd: Array<number | null>; signal: Array<number | null>; hist: Array<number | null> } {
  const n = values.length;
  const macdArr: Array<number | null> = new Array(n).fill(null);
  const signalArr: Array<number | null> = new Array(n).fill(null);
  const histArr: Array<number | null> = new Array(n).fill(null);
  if (n === 0) return { macd: macdArr, signal: signalArr, hist: histArr };
  const ef = rollingEma(values, fast);
  const es = rollingEma(values, slow);
  const kSig = 2 / (signal + 1);
  let prevSig: number | null = null;
  for (let i = 0; i < n; i++) {
    const f = ef[i];
    const s = es[i];
    if (f == null || s == null) { continue; }
    const m = f - s;
    macdArr[i] = m;
    if (prevSig == null) {
      // seed when we have enough MACD history for signal
      const start = i - signal + 1;
      if (start >= 0) {
        const macdHist = macdArr.slice(start, i + 1) as number[];
        if (macdHist.every(v => v != null)) {
          const seed = macdHist.reduce((a, b) => a + b, 0) / signal;
          prevSig = seed;
          signalArr[i] = seed;
          histArr[i] = m - seed;
        }
      }
    } else {
      prevSig = m * kSig + prevSig * (1 - kSig);
      signalArr[i] = prevSig;
      histArr[i] = m - prevSig;
    }
  }
  return { macd: macdArr, signal: signalArr, hist: histArr };
}

/**
 * Bollinger Bands indicator
 * @param values
 * @param period
 * @param k
 * @returns
 */
export function bollingerSeries(values: number[], period = 20, k = 2): { basis: Array<number | null>; upper: Array<number | null>; lower: Array<number | null>; bandwidth: Array<number | null>; percentB: Array<number | null> } {
  const n = values.length;
  const basis: Array<number | null> = new Array(n).fill(null);
  const upper: Array<number | null> = new Array(n).fill(null);
  const lower: Array<number | null> = new Array(n).fill(null);
  const bandwidth: Array<number | null> = new Array(n).fill(null);
  const percentB: Array<number | null> = new Array(n).fill(null);
  const { mean, sd } = rollingMeanStd(values, period);
  for (let i = 0; i < n; i++) {
    const m = mean[i];
    const s = sd[i];
    if (m == null || s == null) continue;
    basis[i] = m;
    upper[i] = m + k * s;
    lower[i] = m - k * s;
    if (Math.abs(m) > 1e-12) bandwidth[i] = ((upper[i]! - lower[i]!) / m) * 100;
    const denom = upper[i]! - lower[i]!;
    if (denom !== 0) percentB[i] = (values[i] - lower[i]!) / denom;
  }
  return { basis, upper, lower, bandwidth, percentB };
}

/**
 * SuperTrend indicator series
 * @param high
 * @param low
 * @param close
 * @param period
 * @param multiplier
 * @returns
 */
export function supertrendSeries(high: number[], low: number[], close: number[], period = 10, multiplier = 3): { line: Array<number | null>; dir: Array<'up' | 'down' | null> } {
  const n = close.length;
  const line: Array<number | null> = new Array(n).fill(null);
  const dir: Array<'up' | 'down' | null> = new Array(n).fill(null);
  if (n === 0) return { line, dir };

  const atrArr = atrSeries(high, low, close, period);
  const upperBand: Array<number | null> = new Array(n).fill(null);
  const lowerBand: Array<number | null> = new Array(n).fill(null);

  for (let i = period; i < n; i++) {
    const atrVal = atrArr[i];
    if (atrVal == null) continue;

    const mid = (high[i] + low[i]) / 2;
    const ub = mid + multiplier * atrVal;
    const lb = mid - multiplier * atrVal;

    const prevUpper = upperBand[i - 1];
    const prevLower = lowerBand[i - 1];
    const prevClose = close[i - 1];

    upperBand[i] = (prevUpper != null && (ub >= prevUpper || prevClose > prevUpper)) ? prevUpper : ub;
    lowerBand[i] = (prevLower != null && (lb <= prevLower || prevClose < prevLower)) ? prevLower : lb;

    // Initialize direction and line if this is the first valid calculation
    if (dir[i - 1] === null) {
      // For the first calculation, set initial bands
      if (i === period) {
        upperBand[i - 1] = ub;
        lowerBand[i - 1] = lb;
      }
      dir[i - 1] = 'up'; // Set initial direction to 'up'
      line[i - 1] = lowerBand[i - 1];
    }

    const prevDir = dir[i - 1];

    if (prevDir === 'up' && close[i] < (lowerBand[i] as number)) {
      dir[i] = 'down';
    } else if (prevDir === 'down' && close[i] > (upperBand[i] as number)) {
      dir[i] = 'up';
    } else {
      dir[i] = prevDir;
    }

    line[i] = dir[i] === 'up' ? lowerBand[i] : upperBand[i];
  }
  return { line, dir };
}

// Helper functions needed

function rollingEma(values: number[], period: number): Array<number | null> {
  const out: Array<number | null> = new Array(values.length).fill(null);
  if (values.length === 0 || period <= 0) return out;
  const k = 2 / (period + 1);
  if (values.length < period) return out;
  let emaVal = sma(values.slice(0, period), period);
  if (emaVal == null) return out;
  out[period - 1] = emaVal;
  for (let i = period; i < values.length; i++) {
    emaVal = values[i] * k + emaVal * (1 - k);
    out[i] = emaVal;
  }
  return out;
}

function rollingMeanStd(values: number[], period: number): { mean: Array<number | null>; sd: Array<number | null> } {
  const mean: Array<number | null> = new Array(values.length).fill(null);
  const sd: Array<number | null> = new Array(values.length).fill(null);
  if (period <= 0 || values.length < period) return { mean, sd };
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    sumSq += values[i] * values[i];
    if (i >= period) {
      sum -= values[i - period];
      sumSq -= values[i - period] * values[i - period];
    }
    if (i >= period - 1) {
      const m = sum / period;
      const variance = Math.max(0, sumSq / period - m * m);
      mean[i] = m;
      sd[i] = Math.sqrt(variance);
    }
  }
  return { mean, sd };
}

function rollingMax(values: number[], period: number): Array<number | null> {
  const out: Array<number | null> = new Array(values.length).fill(null);
  if (period <= 0) return out;
  const deque: number[] = [];
  for (let i = 0; i < values.length; i++) {
    while (deque.length && deque[0] <= i - period) deque.shift();
    while (deque.length && values[deque[deque.length - 1]] <= values[i]) deque.pop();
    deque.push(i);
    if (i >= period - 1) out[i] = values[deque[0]];
  }
  return out;
}

function rollingMin(values: number[], period: number): Array<number | null> {
  const out: Array<number | null> = new Array(values.length).fill(null);
  if (period <= 0) return out;
  const deque: number[] = [];
  for (let i = 0; i < values.length; i++) {
    while (deque.length && deque[0] <= i - period) deque.shift();
    while (deque.length && values[deque[deque.length - 1]] >= values[i]) deque.pop();
    deque.push(i);
    if (i >= period - 1) out[i] = values[deque[0]];
  }
  return out;
}

function sma(values: number[], period: number): Num {
  if (!Array.isArray(values) || period <= 0 || values.length < period) return null;
  const window = values.slice(-period);
  const sum = window.reduce((acc, val) => acc + val, 0);
  return sum / period;
}