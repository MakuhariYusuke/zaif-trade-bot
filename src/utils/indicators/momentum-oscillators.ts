// Pure indicator functions (no side effects)
// Lightweight implementations with optional previous-state to enable rolling updates.

export type Num = number | null;

// ---- Momentum/Oscillators ----

/**
 * Relative Strength Index (RSI) (Wilder's)
 * @param values - Array of price values.
 * @param period - The period over which to calculate RSI (default: 14).
 * @param prev - Previous state for rolling calculation.
 * @returns An object containing the RSI value and average gain/loss for the next rolling update.
 */
export function rsi(values: number[], period: number = 14, prev?: { avgGain: number; avgLoss: number }): { value: Num; avgGain?: number; avgLoss?: number } {
  if (!Array.isArray(values) || values.length < 2) return { value: null };

  const len = values.length;
  const change = values[len - 1] - values[len - 2];
  const gain = Math.max(0, change);
  const loss = Math.max(0, -change);

  if (prev?.avgGain != null && prev?.avgLoss != null) {
    const avgGain = wilderSmooth(prev.avgGain, gain, period);
    const avgLoss = wilderSmooth(prev.avgLoss, loss, period);
    const rs = avgLoss === 0 ? Infinity : avgGain / avgLoss;
    const value = 100 - 100 / (1 + rs);
    return { value, avgGain, avgLoss };
  }

  if (values.length < period + 1) return { value: null };

  let totalGain = 0, totalLoss = 0;
  for (let i = 1; i <= period; i++) {
    const ch = values[i] - values[i - 1];
    totalGain += Math.max(0, ch);
    totalLoss += Math.max(0, -ch);
  }

  let avgGain = totalGain / period;
  let avgLoss = totalLoss / period;

  for (let i = period + 1; i < len; i++) {
    const ch = values[i] - values[i - 1];
    const g = Math.max(0, ch);
    const l = Math.max(0, -ch);
    avgGain = wilderSmooth(avgGain, g, period);
    avgLoss = wilderSmooth(avgLoss, l, period);
  }

  const rs = avgLoss === 0 ? Infinity : avgGain / avgLoss;
  const value = 100 - 100 / (1 + rs);
  return { value, avgGain, avgLoss };
}

/**
 * Moving Average Convergence Divergence (MACD)
 * @param values - Array of price values.
 * @param fast - Fast EMA period (default: 12).
 * @param slow - Slow EMA period (default: 26).
 * @param signal - Signal line EMA period (default: 9).
 * @param prev - Previous state for rolling calculation.
 * @returns An object containing MACD line, signal line, histogram, and latest EMA values for rolling updates.
 */
export function macd(values: number[], fast: number = 12, slow: number = 26, signal: number = 9, prev?: { emaFast?: number; emaSlow?: number; signal?: number }): { macd: Num; signal: Num; hist: Num; emaFast?: number; emaSlow?: number } {
  if (values.length === 0) return { macd: null, signal: null, hist: null };

  const ef = ema(values, fast, prev?.emaFast);
  const es = ema(values, slow, prev?.emaSlow);

  if (ef == null || es == null) {
    return { macd: null, signal: null, hist: null, emaFast: ef ?? undefined, emaSlow: es ?? undefined };
  }

  const macdLine = ef - es;

  // For a full calculation, we need enough data for the slow EMA plus the signal EMA.
  if (values.length < slow + signal - 1 && prev?.signal == null) {
    return { macd: macdLine, signal: null, hist: null, emaFast: ef, emaSlow: es };
  }

  // To calculate the signal line, we need a history of MACD values.
  const macdValues: number[] = [];
  if (values.length >= slow + signal - 1) {
    for (let i = slow - 1; i < values.length; i++) {
      const slice = values.slice(0, i + 1);
      const fastEma = ema(slice, fast);
      const slowEma = ema(slice, slow);
      if (fastEma != null && slowEma != null) {
        macdValues.push(fastEma - slowEma);
      }
    }
  }

  const sig = prev?.signal != null
    ? (macdLine * (2 / (signal + 1))) + (prev.signal * (1 - (2 / (signal + 1))))
    : ema(macdValues, signal);

  const hist = sig != null ? macdLine - sig : null;

  return { macd: macdLine, signal: sig, hist, emaFast: ef, emaSlow: es };
}

/**
 * Stochastic Oscillator
 * Calculates the Stochastic Oscillator values %K and %D.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param kPeriod - The lookback period for %K calculation (default: 14).
 * @param dPeriod - The smoothing period for %D (SMA of %K) (default: 3).
 * @param smooth - The smoothing period for %K (SMA of Raw %K) (default: 3).
 * @returns An object with %K and %D values, which may be null if there is insufficient data.
 */
export function stochastic(high: number[], low: number[], close: number[], kPeriod: number = 14, dPeriod: number = 3, smooth: number = 3): { k: Num; d: Num } {
  const requiredLength = kPeriod + smooth - 1;
  if (close.length < requiredLength || high.length < requiredLength || low.length < requiredLength) {
    return { k: null, d: null };
  }

  const rawKValues: number[] = [];
  for (let i = 0; i < smooth; i++) {
    const end = close.length - i;
    const start = end - kPeriod;
    const windowHigh = high.slice(start, end);
    const windowLow = low.slice(start, end);
    const currentClose = close[end - 1];

    const highestHigh = Math.max(...windowHigh);
    const lowestLow = Math.min(...windowLow);

    const rawK = highestHigh === lowestLow ? 0 : ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;
    rawKValues.unshift(rawK); // Add to the beginning to maintain order
  }

  const k = sma(rawKValues, smooth);
  if (k === null || rawKValues.length < dPeriod) {
    return { k, d: null };
  }

  return { k, d: sma(rawKValues.slice(-dPeriod), dPeriod) };
}

/**
 * Williams %R
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param period - The lookback period for %R calculation (default: 14).
 * @returns The Williams %R value, or null if there is insufficient data.
 */
export function williamsR(high: number[], low: number[], close: number[], period: number = 14): Num {
  if (high.length < period || low.length < period || close.length < period) return null;
  const hh = Math.max(...high.slice(-period));
  const ll = Math.min(...low.slice(-period));
  const c = close[close.length - 1];
  return hh === ll ? -50 : -100 * ((hh - c) / (hh - ll));
}

/**
 * Commodity Channel Index (CCI)
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param period - The lookback period for CCI calculation (default: 20).
 * @returns The CCI value, or null if there is insufficient data.
 */
export function cci(high: number[], low: number[], close: number[], period: number = 20): Num {
  if (high.length < period || low.length < period || close.length < period) return null;

  const typicalPrices: number[] = [];
  for (let i = high.length - period; i < high.length; i++) {
    typicalPrices.push((high[i] + low[i] + close[i]) / 3);
  }

  const smaTp = typicalPrices.reduce((a, b) => a + b, 0) / period;
  const meanDeviation = typicalPrices.reduce((a, b) => a + Math.abs(b - smaTp), 0) / period;

  if (meanDeviation === 0) return 0;

  const lastTp = typicalPrices[typicalPrices.length - 1];
  return (lastTp - smaTp) / (0.015 * meanDeviation);
}

/**
 * Momentum
 * @param values - Array of price values.
 * @param period - The lookback period for momentum calculation (default: 10).
 * @returns The momentum value (price difference), or null if there is insufficient data.
 */
export function momentum(values: number[], period: number = 10): Num {
  if (values.length <= period) return null;
  const prev = values[values.length - 1 - period];
  const cur = values[values.length - 1];
  return cur - prev;
}

/**
 * Rate of Change (ROC)
 * @param values - Array of price values.
 * @param period - The lookback period for ROC calculation (default: 14).
 * @returns The ROC value in percent, or null if there is insufficient data.
 */
export function roc(values: number[], period: number = 14): Num {
  if (values.length < period + 1) return null;
  const prev = values[values.length - 1 - period];
  const cur = values[values.length - 1];
  if (prev === 0) return null;
  return ((cur - prev) / prev) * 100;
}

/**
 * Calculates envelope bands above and below a moving average.
 * @param ma - The moving average value.
 * @param pct - Percentage for the envelope width (default: 1.5).
 * @returns An object containing the upper and lower band values.
 */
export function envelopes(ma: Num, pct: number = 1.5): { upper: Num; lower: Num } {
  if (ma == null) return { upper: null, lower: null };
  const f = pct / 100;
  return { upper: ma * (1 + f), lower: ma * (1 - f) };
}

/**
 * Price Deviation Percentage (from a moving average)
 * @param price - The current price.
 * @param movingAverage - The moving average value.
 * @returns The percentage by which the price deviates from the moving average, or null if the moving average is invalid or zero.
 */
export function deviationPct(price: number, movingAverage: Num): Num {
  if (movingAverage == null || movingAverage === 0) return null;
  return ((price - movingAverage) / movingAverage) * 100;
}

/**
 * Fibonacci Retracement Position
 * Calculates the position (0 to 1) of the latest close price within the recent high-low window.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param lookback - Number of periods to look back for the high/low window (default: 100).
 * @returns Position of close price within the range (0 = lowest, 1 = highest), or null if not enough data or range is zero.
 */
export function fibPosition(high: number[], low: number[], close: number[], lookback: number = 100): Num {
  if (high.length < lookback || low.length < lookback || close.length === 0) return null;
  const windowHigh = high.slice(-lookback);
  const windowLow = low.slice(-lookback);
  const h = Math.max(...windowHigh);
  const l = Math.min(...windowLow);
  const c = close[close.length - 1];
  const diff = h - l;
  if (diff <= 0) return c === h ? 1 : (c === l ? 0 : null);
  return (c - l) / diff;
}

/**
 * State object for the Parabolic SAR calculation.
 */
export type PsarState = { sar: number; ep: number; af: number; uptrend: boolean };

/**
 * Parabolic SAR (one-step calculation, requires previous state)
 * @param prev - Previous SAR state object.
 * @param high - Current high price.
 * @param low - Current low price.
 * @param step - Acceleration factor step (default: 0.02).
 * @param max - Maximum acceleration factor (default: 0.2).
 * @returns The new state object containing the next SAR value, EP, AF, and trend direction.
 */
export function psarStep(
  prev: PsarState,
  high: number,
  low: number,
  step = 0.02,
  max = 0.2
): PsarState {
  let { sar, ep, af, uptrend } = prev;
  const prevSar = sar;

  if (uptrend) {
    sar = prevSar + af * (ep - prevSar);
    if (high > ep) {
      ep = high;
      af = Math.min(max, af + step);
    }
    // SAR should not be higher than the low of the last two periods
    sar = Math.min(sar, low);
    if (low < sar) {
      uptrend = false;
      sar = ep; // Switch to the highest point of the uptrend
      ep = low;
      af = step;
    }
  } else { // Downtrend
    sar = prevSar + af * (ep - prevSar);
    if (low < ep) {
      ep = low;
      af = Math.min(max, af + step);
    }
    // SAR should not be lower than the high of the last two periods
    sar = Math.max(sar, high);
    if (high > sar) {
      uptrend = true;
      sar = ep; // Switch to the lowest point of the downtrend
      ep = high;
      af = step;
    }
  }
  return { sar, ep, af, uptrend };
}

/**
 * SuperTrend indicator
 * Returns the latest SuperTrend value and direction (1 for uptrend, -1 for downtrend).
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param period - ATR lookback period (default: 10).
 * @param multiplier - ATR multiplier for band width (default: 3).
 * @returns An object containing the latest SuperTrend value, direction, and band values.
 */
export function supertrend(
  high: number[],
  low: number[],
  close: number[],
  period: number = 10,
  multiplier: number = 3
): { value: Num; dir: 1 | -1 | null; upper?: Num; lower?: Num } {
  const n = close.length;
  if (n <= period) return { value: null, dir: null };

  const atrValue = atr(high, low, close, period);
  if (atrValue === null) return { value: null, dir: null };

  const upperBand: number[] = new Array(n);
  const lowerBand: number[] = new Array(n);
  const st: number[] = new Array(n);
  let direction = 1;

  for (let i = period; i < n; i++) {
    const currentAtr = atr(high.slice(0, i + 1), low.slice(0, i + 1), close.slice(0, i + 1), period);
    if (currentAtr === null) continue;

    const mid = (high[i] + low[i]) / 2;
    const ub = mid + multiplier * currentAtr;
    const lb = mid - multiplier * currentAtr;

    upperBand[i] = (i > 0 && (ub < upperBand[i - 1] || close[i - 1] > upperBand[i - 1])) ? ub : (i > 0 ? upperBand[i - 1] : ub);
    lowerBand[i] = (i > 0 && (lb > lowerBand[i - 1] || close[i - 1] < lowerBand[i - 1])) ? lb : (i > 0 ? lowerBand[i - 1] : lb);

    if (i > 0) {
      if (st[i - 1] === upperBand[i - 1] && close[i] > upperBand[i]) {
        direction = 1;
      } else if (st[i - 1] === lowerBand[i - 1] && close[i] < lowerBand[i]) {
        direction = -1;
      }
    }

    st[i] = direction === 1 ? lowerBand[i] : upperBand[i];
  }

  const lastIndex = n - 1;
  const val = st[lastIndex];
  const dir = val === lowerBand[lastIndex] ? 1 : -1;

  return { value: val, dir, upper: upperBand[lastIndex], lower: lowerBand[lastIndex] };
}

/**
 * True Strength Index (TSI) Series
 * Calculates the TSI and its signal line over the entire series.
 * @param close - Array of closing prices.
 * @param short - The short EMA period (default: 13).
 * @param long - The long EMA period (default: 25).
 * @param signal - The signal line EMA period (default: 13).
 * @returns An object containing arrays for TSI and its signal line.
 */
export function tsiSeries(close: number[], short = 13, long = 25, signal = 13): { tsi: Array<number | null>; signal: Array<number | null> } {
  const n = close.length;
  const tsi: Array<number | null> = new Array(n).fill(null);
  const sig: Array<number | null> = new Array(n).fill(null);
  if (n <= 1) return { tsi, signal: sig };
  const mtm: Array<number | null> = new Array(n).fill(null);
  const absMtm: Array<number | null> = new Array(n).fill(null);
  for (let i = 1; i < n; i++) {
    const d = close[i] - close[i - 1];
    mtm[i] = d;
    absMtm[i] = Math.abs(d);
  }
  const e1 = emaSeriesSkipNulls(mtm, short);
  const e2 = emaSeriesSkipNulls(e1, long);
  const a1 = emaSeriesSkipNulls(absMtm, short);
  const a2 = emaSeriesSkipNulls(a1, long);
  for (let i = 0; i < n; i++) {
    const num = e2[i];
    const den = a2[i];
    if (num != null && den != null && den !== 0) tsi[i] = 100 * (num / den);
  }
  const s = emaSeriesSkipNulls(tsi, signal);
  for (let i = 0; i < n; i++) sig[i] = s[i];
  return { tsi, signal: sig };
}

/**
 * Vortex Indicator
 * Calculates the Vortex Indicator (+VI and -VI) over a specified period.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param period - The lookback period for VI calculation (default: 14).
 * @returns An object containing arrays for +VI and -VI.
 */
export function vortexSeries(high: number[], low: number[], close: number[], period = 14): { viPlus: Array<number | null>; viMinus: Array<number | null> } {
  const n = close.length;
  const viPlus: Array<number | null> = new Array(n).fill(null);
  const viMinus: Array<number | null> = new Array(n).fill(null);
  if (n <= 1) return { viPlus, viMinus };
  let sumTR = 0, sumVMp = 0, sumVMm = 0;
  const trArr: number[] = new Array(n).fill(0);
  const vmpArr: number[] = new Array(n).fill(0);
  const vmmArr: number[] = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    const tr = Math.max(high[i] - low[i], Math.abs(high[i] - close[i - 1]), Math.abs(low[i] - close[i - 1]));
    trArr[i] = tr;
    vmpArr[i] = Math.abs(high[i] - low[i - 1]);
    vmmArr[i] = Math.abs(low[i] - high[i - 1]);
  }
  for (let i = 1; i < n; i++) {
    sumTR += trArr[i];
    sumVMp += vmpArr[i];
    sumVMm += vmmArr[i];
    if (i > period) {
      sumTR -= trArr[i - period];
      sumVMp -= vmpArr[i - period];
      sumVMm -= vmmArr[i - period];
    }
    if (i >= period) {
      viPlus[i] = sumTR === 0 ? 0 : sumVMp / sumTR;
      viMinus[i] = sumTR === 0 ? 0 : sumVMm / sumTR;
    }
  }
  return { viPlus, viMinus };
}

/**
 * Aroon Indicator
 * Calculates Aroon Up, Aroon Down, and Aroon Oscillator over a specified period.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param period - The lookback period for Aroon calculation (default: 14).
 * @returns An object containing arrays for Aroon Up, Aroon Down, and Aroon Oscillator.
 */
export function aroonSeries(high: number[], low: number[], period = 14): { aroonUp: Array<number | null>; aroonDown: Array<number | null>; aroonOsc: Array<number | null> } {
  const n = Math.min(high.length, low.length);
  const up: Array<number | null> = new Array(n).fill(null);
  const down: Array<number | null> = new Array(n).fill(null);
  const osc: Array<number | null> = new Array(n).fill(null);
  if (period <= 0 || n === 0) return { aroonUp: up, aroonDown: down, aroonOsc: osc };
  const argMax = rollingArgMax(high, period);
  const argMin = rollingArgMin(low, period);
  for (let i = 0; i < n; i++) {
    if (i >= period - 1) {
      const idxH = argMax[i];
      const idxL = argMin[i];
      if (idxH == null || idxL == null) {
        up[i] = null;
        down[i] = null;
        osc[i] = null;
      } else {
        const upVal = ((period - 1 - (i - idxH)) / (period - 1)) * 100;
        const dnVal = ((period - 1 - (i - idxL)) / (period - 1)) * 100;
        up[i] = Math.max(0, Math.min(100, upVal));
        down[i] = Math.max(0, Math.min(100, dnVal));
        if (up[i] != null && down[i] != null) {
          osc[i] = up[i]! - down[i]!;
        } else {
          osc[i] = null;
        }
      }
    }
  }
  return { aroonUp: up, aroonDown: down, aroonOsc: osc };
}

// Helper functions needed
function wilderSmooth(prevValue: number, currentValue: number, period: number): number {
  return (prevValue * (period - 1) + currentValue) / period;
}

function ema(values: number[], period: number, prevEma?: number): Num {
  if (period <= 0 || values.length === 0) return null;
  const k = 2 / (period + 1);

  if (prevEma != null) {
    // Rolling calculation
    const lastValue = values[values.length - 1];
    return (lastValue * k) + (prevEma * (1 - k));
  }

  // Full calculation
  if (values.length < period) {
    return null; // Not enough data for even one EMA value
  }

  // Seed with SMA of the first 'period' values
  let currentEma = sma(values.slice(0, period), period);
  if (currentEma == null) {
    return null;
  }

  // Calculate subsequent EMA values up to the end of the array
  for (let i = period; i < values.length; i++) {
    currentEma = (values[i] * k) + (currentEma * (1 - k));
  }

  return currentEma;
}

function sma(values: number[], period: number): Num {
  if (!Array.isArray(values) || period <= 0 || values.length < period) return null;
  const window = values.slice(-period);
  const sum = window.reduce((acc, val) => acc + val, 0);
  return sum / period;
}

function atr(high: number[], low: number[], close: number[], period: number = 14): Num {
  const n = close.length;
  if (n <= period) return null;

  const trArr: number[] = [];
  for (let i = 1; i < n; i++) {
    const tr = Math.max(
      high[i] - low[i],
      Math.abs(high[i] - close[i - 1]),
      Math.abs(low[i] - close[i - 1])
    );
    trArr.push(tr);
  }

  let currentAtr = trArr.slice(0, period).reduce((a, b) => a + b, 0) / period;

  for (let i = period; i < trArr.length; i++) {
    currentAtr = wilderSmooth(currentAtr, trArr[i], period);
  }

  return currentAtr;
}

function emaSeriesSkipNulls(values: Array<number | null>, period: number): Array<number | null> {
  const n = values.length;
  const out: Array<number | null> = new Array(n).fill(null);
  const k = 2 / (period + 1);
  let seedIdx = -1;
  for (let i = 0, cnt = 0, sum = 0; i < n; i++) {
    const v = values[i];
    if (v == null) { continue; }
    cnt++;
    sum += v;
    if (cnt === period) {
      const seed = sum / period;
      out[i] = seed;
      seedIdx = i;
      break;
    }
  }
  if (seedIdx === -1) return out;
  let prev: number;
  if (typeof out[seedIdx] === 'number') {
    prev = out[seedIdx] as number;
  } else {
    return out;
  }
  for (let i = seedIdx + 1; i < n; i++) {
    const v = values[i];
    if (v == null) { out[i] = null; continue; }
    prev = v * k + prev * (1 - k);
    out[i] = prev;
  }
  return out;
}

function rollingArgMax(values: number[], period: number): Array<number | null> {
  const n = values.length;
  const out: Array<number | null> = new Array(n).fill(null);
  const dq: number[] = [];
  for (let i = 0; i < n; i++) {
    while (dq.length && dq[0] <= i - period) dq.shift();
    while (dq.length && values[dq[dq.length - 1]] <= values[i]) dq.pop();
    dq.push(i);
    if (i >= period - 1) out[i] = dq[0];
  }
  return out;
}

function rollingArgMin(values: number[], period: number): Array<number | null> {
  const n = values.length;
  const out: Array<number | null> = new Array(n).fill(null);
  const dq: number[] = [];
  for (let i = 0; i < n; i++) {
    while (dq.length && dq[0] <= i - period) dq.shift();
    while (dq.length && values[dq[dq.length - 1]] >= values[i]) dq.pop();
    dq.push(i);
    if (i >= period - 1) out[i] = dq[0];
  }
  return out;
}