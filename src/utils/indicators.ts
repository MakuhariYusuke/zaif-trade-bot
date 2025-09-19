// Pure indicator functions (no side effects)
// Lightweight implementations with optional previous-state to enable rolling updates.

export type Num = number | null;

// ---- Series helpers ----

/**
 * A helper function for Wilder's smoothing (a specific type of EMA with alpha = 1/period).
 * @param prevValue The previous smoothed value.
 * @param currentValue The current raw value.
 * @param period The smoothing period.
 * @returns The new smoothed value.
 */
function wilderSmooth(prevValue: number, currentValue: number, period: number): number {
  return (prevValue * (period - 1) + currentValue) / period;
}

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

function atrSeries(high: number[], low: number[], close: number[], period: number): Array<number | null> {
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

/**
 * Simple Moving Average (SMA)
 * 
 * Calculates the simple moving average for the given period.
 * @param values - Array of price values.
 * @param period - The lookback period for the SMA calculation. 
 * @returns The simple moving average value, or null if there is insufficient data.
 */
export function sma(values: number[], period: number): Num {
  if (!Array.isArray(values) || period <= 0 || values.length < period) return null;
  const window = values.slice(-period);
  const sum = window.reduce((acc, val) => acc + val, 0);
  return sum / period;
}

/**
 * Exponential Moving Average (EMA)
 * 
 * Returns the latest EMA value. Can perform a full calculation or a rolling update.
 * Note: The full calculation seeds with SMA for the first value.
 * @param values - Array of price values (oldest to newest).
 * @param period - The lookback period for the EMA calculation.
 * @param prevEma - The previous EMA value for a rolling calculation.
 * @returns The latest EMA value, or null if there is insufficient data.
 */
export function ema(values: number[], period: number, prevEma?: number): Num {
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

/**
 * Weighted Moving Average (WMA)
 * WMA gives more weight to recent prices.
 * @param values - Array of price values.
 * @param period - The lookback period for the WMA calculation.
 * @returns The latest WMA value, or null if there is insufficient data.
 */
export function wma(values: number[], period: number): Num {
  if (period <= 0 || values.length < period) return null;

  const den = (period * (period + 1)) / 2;
  const window = values.slice(-period);

  const num = window.reduce((acc, val, i) => acc + val * (i + 1), 0);
  return num / den;
}

/**
 * Standard Deviation (STDDEV)
 * Calculates the standard deviation over the specified period.
 * @param values - Array of price values.
 * @param period - The lookback period for the STDDEV calculation.
 * @returns The standard deviation value, or null if there is insufficient data.
 */
export function stddev(values: number[], period: number): Num {
  if (period <= 0 || values.length < period) return null;
  const window = values.slice(-period);
  const mean = sma(window, period);
  if (mean === null) return null;

  const sumSqDiff = window.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0);
  return Math.sqrt(sumSqDiff / period);
}

/**
 * Calculates Bollinger Bands for a given array of price values.
 * @param values - Array of price values.
 * @param period - The lookback period for the moving average and standard deviation (default: 20).
 * @param k - The standard deviation multiplier for the bands (default: 2).
 * @returns An object containing basis, upper, and lower bands. All properties are null if there is insufficient data.
 */
export function bollinger(values: number[], period: number = 20, k: number = 2): { basis: Num; upper: Num; lower: Num } {
  const basis = sma(values, period);
  const sd = stddev(values, period);
  if (basis == null || sd == null) return { basis: null, upper: null, lower: null };
  return { basis, upper: basis + k * sd, lower: basis - k * sd };
}

/**
 * Bollinger Band Width (BBW)
 * Calculates the width of the Bollinger Bands as a percentage of the middle band.
 * @param values - Array of price values.
 * @param period - The lookback period for Bollinger Bands (default: 20).
 * @param k - The standard deviation multiplier (default: 2).
 * @returns BBW value in percent, or null if data is insufficient or the basis is zero.
 */
export function bbWidth(values: number[], period: number = 20, k: number = 2): Num {
  const { basis, upper, lower } = bollinger(values, period, k);
  if (basis == null || upper == null || lower == null || Math.abs(basis) < 1e-12) return null;
  return ((upper - lower) / basis) * 100;
}

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
 * Ichimoku Cloud
 * Calculates the Ichimoku indicator components.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param tenkan - The period for the Conversion Line (Tenkan-sen) (default: 9).
 * @param kijun - The period for the Base Line (Kijun-sen) (default: 26).
 * @param senkouB - The period for Leading Span B (Senkou Span B) (default: 52).
 * @returns An object containing the values for Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, and Chikou Span.
 */
export function ichimoku(high: number[], low: number[], close: number[], tenkan: number = 9, kijun: number = 26, senkouB: number = 52): { tenkan: Num; kijun: Num; spanA: Num; spanB: Num; chikou: Num } {
  const conv = (p: number): Num => {
    if (high.length < p || low.length < p) return null;
    const h = Math.max(...high.slice(-p));
    const l = Math.min(...low.slice(-p));
    return (h + l) / 2;
  };
  const t = conv(tenkan);
  const k = conv(kijun);
  const b = conv(senkouB);
  const a = (t != null && k != null) ? (t + k) / 2 : null;
  // Chikou Span: closing price shifted back by the Kijun period
  const chikou = close.length >= kijun ? close[close.length - kijun] : null;
  return { tenkan: t, kijun: k, spanA: a, spanB: b, chikou };
}

/**
 * DMI/ADX (Directional Movement Index / Average Directional Index)
 * Calculates +DI, -DI, and ADX using Wilder's smoothing.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param period - The lookback period for DMI and ADX (default: 14).
 * @returns An object containing the latest ADX, +DI, and -DI values.
 */
export function dmiAdx(high: number[], low: number[], close: number[], period: number = 14): { adx: Num; plusDi: Num; minusDi: Num } {
  const n = close.length;
  // ADX requires at least (period * 2 - 1) data points for the initial calculation.
  if (n < period * 2) {
    return { adx: null, plusDi: null, minusDi: null };
  }

  const trArr: number[] = [];
  const plusDmArr: number[] = [];
  const minusDmArr: number[] = [];

  for (let i = 1; i < n; i++) {
    const h = high[i], l = low[i], prevH = high[i - 1], prevL = low[i - 1], prevC = close[i - 1];
    const tr = Math.max(h - l, Math.abs(h - prevC), Math.abs(l - prevC));
    trArr.push(tr);
    const upMove = h - prevH;
    const downMove = prevL - l;
    plusDmArr.push(upMove > downMove && upMove > 0 ? upMove : 0);
    minusDmArr.push(downMove > upMove && downMove > 0 ? downMove : 0);
  }

  let atr = trArr.slice(0, period).reduce((sum, val) => sum + val, 0);
  let sPlusDm = plusDmArr.slice(0, period).reduce((sum, val) => sum + val, 0);
  let sMinusDm = minusDmArr.slice(0, period).reduce((sum, val) => sum + val, 0);

  const dxArr: number[] = [];

  for (let i = period; i < trArr.length; i++) {
    atr = wilderSmooth(atr, trArr[i], period);
    sPlusDm = wilderSmooth(sPlusDm, plusDmArr[i], period);
    sMinusDm = wilderSmooth(sMinusDm, minusDmArr[i], period);

    const plusDi = atr === 0 ? 0 : (sPlusDm / atr) * 100;
    const minusDi = atr === 0 ? 0 : (sMinusDm / atr) * 100;
    const diSum = plusDi + minusDi;
    const dx = diSum === 0 ? 0 : (Math.abs(plusDi - minusDi) / diSum) * 100;
    dxArr.push(dx);
  }

  if (dxArr.length < period) {
    const plusDi = atr === 0 ? 0 : (sPlusDm / atr) * 100;
    const minusDi = atr === 0 ? 0 : (sMinusDm / atr) * 100;
    return { adx: null, plusDi, minusDi };
  }

  let adx = dxArr.slice(0, period).reduce((sum, val) => sum + val, 0) / period;
  for (let i = period; i < dxArr.length; i++) {
    adx = wilderSmooth(adx, dxArr[i], period);
  }

  const finalPlusDi = atr === 0 ? 0 : (sPlusDm / atr) * 100;
  const finalMinusDi = atr === 0 ? 0 : (sMinusDm / atr) * 100;

  return { adx, plusDi: finalPlusDi, minusDi: finalMinusDi };
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
 * Average True Range (ATR)
 * This implementation uses Wilder's smoothing (an EMA with alpha = 1/period).
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param period - The lookback period for ATR calculation (default: 14).
 * @returns The ATR value, or null if there is insufficient data.
 */
export function atr(high: number[], low: number[], close: number[], period: number = 14): Num {
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

/**
 * Donchian Channel Width
 * Calculates the Donchian channel width as a percentage of the latest close price.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param period - The lookback period for the Donchian channel (default: 20).
 * @returns Donchian channel width in percent, or null if not enough data or close is zero.
 */
export function donchianWidth(high: number[], low: number[], close: number[], period: number = 20): Num {
  if (high.length < period || low.length < period || close.length === 0) return null;
  const hh = Math.max(...high.slice(-period));
  const ll = Math.min(...low.slice(-period));
  const c = close[close.length - 1];
  if (c == null || c === 0) return null;
  return ((hh - ll) / c) * 100;
}

/**
 * Hull Moving Average (HMA)
 * A fast and smooth moving average that reduces lag.
 * @param values - Array of price values.
 * @param period - Lookback period (default: 20).
 * @returns The latest Hull Moving Average value, or null if not enough data.
 */
export function hma(values: number[], period: number = 20): Num {
  const halfP = Math.floor(period / 2);
  const sqrtP = Math.floor(Math.sqrt(period));
  if (values.length < period + sqrtP - 1) return null;

  const wma1 = wma(values, halfP);
  const wma2 = wma(values, period);

  if (wma1 === null || wma2 === null) return null;

  const diffValues: number[] = [];
  for (let i = 0; i < sqrtP; i++) {
    const end = values.length - i;
    const w1 = wma(values.slice(0, end), halfP);
    const w2 = wma(values.slice(0, end), period);
    if (w1 === null || w2 === null) return null; // Should not happen given initial check
    diffValues.unshift(2 * w1 - w2);
  }

  return wma(diffValues, sqrtP);
}

/** 
 * Kaufman's Adaptive Moving Average (KAMA)
 * @param values - Array of price values.
 * @param period - Lookback period for Efficiency Ratio (default: 10).
 * @param fast - Fast EMA period for scaling constant (default: 2).
 * @param slow - Slow EMA period for scaling constant (default: 30).
 * @param prevKama - Previous KAMA value for rolling calculation (optional).
 * @returns The latest KAMA value, or null if insufficient data.
 */
export function kama(values: number[], period = 10, fast = 2, slow = 30, prevKama?: number): Num {
  if (values.length <= period) return null;

  const change = Math.abs(values[values.length - 1] - values[values.length - 1 - period]);
  let volatility = 0;
  for (let i = values.length - period; i < values.length; i++) {
    volatility += Math.abs(values[i] - values[i - 1]);
  }

  const er = volatility === 0 ? 1 : change / volatility;
  const fastSC = 2 / (fast + 1);
  const slowSC = 2 / (slow + 1);
  const sc = Math.pow(er * (fastSC - slowSC) + slowSC, 2);

  let prev: number;
  if (prevKama != null) {
    prev = prevKama;
  } else {
    // Full calculation
    let currentKama = sma(values.slice(0, period), period) ?? values[0];
    for (let i = period; i < values.length; i++) {
      const pChange = Math.abs(values[i] - values[i - period]);
      let pVol = 0;
      for (let j = i - period + 1; j <= i; j++) {
        pVol += Math.abs(values[j] - values[j - 1]);
      }
      const pEr = pVol === 0 ? 1 : pChange / pVol;
      const pSc = Math.pow(pEr * (fastSC - slowSC) + slowSC, 2);
      currentKama = currentKama + pSc * (values[i] - currentKama);
    }
    return currentKama;
  }

  return prev + sc * (values[values.length - 1] - prev);
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

/**
 * Rolling Arg Min
 * Calculates the index of the minimum value in a rolling window.
 * @param values - Array of numeric values.
 * @param period - The size of the rolling window.
 * @returns An array of indices corresponding to the minimum values in each window.
 */
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

/**
 * Aroon Series
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

/**
 * Calculates the Exponential Moving Average (EMA) series, skipping null values in the input array.
 * @param values - Array of numeric values (may contain nulls).
 * @param period - The lookback period for EMA calculation.
 * @returns An array of EMA values, with nulls for positions where insufficient data is available.
 */
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
 * Money Flow Index (MFI) Series
 * Calculates the MFI over a specified period for each point in the series.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices. 
 * @param volume - Array of volume data.
 * @param period - The lookback period for MFI calculation (default: 14).
 * @returns An array containing the MFI values, with nulls for initial periods without enough data.
 */
export function mfiSeries(high: number[], low: number[], close: number[], volume: number[], period = 14): { mfi: Array<number | null> } {
  const n = Math.min(high.length, low.length, close.length, volume.length);
  const mfi: Array<number | null> = new Array(n).fill(null);
  if (n <= 1) return { mfi };
  const tp: number[] = new Array(n);
  for (let i = 0; i < n; i++) tp[i] = (high[i] + low[i] + close[i]) / 3;
  const pos: number[] = new Array(n).fill(0);
  const neg: number[] = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    const mf = tp[i] * volume[i];
    if (tp[i] > tp[i - 1]) pos[i] = mf; else if (tp[i] < tp[i - 1]) neg[i] = mf;
  }
  let sumPos = 0, sumNeg = 0;
  for (let i = 0; i < n; i++) {
    sumPos += pos[i]; sumNeg += neg[i];
    if (i >= period) { 
      sumPos -= pos[i - period]; 
      sumNeg -= neg[i - period]; 
    }
    if (i >= period) {
      if (sumNeg === 0 && sumPos === 0) mfi[i] = 50;
      else if (sumNeg === 0) mfi[i] = 100;
      else if (sumPos === 0) mfi[i] = 0;
      else mfi[i] = 100 - (100 / (1 + (sumPos / sumNeg)));
    }
  }
  return { mfi };
}

/**
 * On-Balance Volume (OBV) Series
 * Calculates the OBV for each point in the series.
 * @param close - Array of closing prices.
 * @param volume - Array of volume data.
 * @returns An array containing the OBV values, with null for the first entry.
 */
export function obvSeries(close: number[], volume: number[]): { obv: Array<number | null> } {
  const n = Math.min(close.length, volume.length);
  const obv: Array<number | null> = new Array(n).fill(null);
  if (n === 0) return { obv };
  let cur = 0;
  obv[0] = 0;
  for (let i = 1; i < n; i++) {
    if (close[i] > close[i - 1]) cur += volume[i];
    else if (close[i] < close[i - 1]) cur -= volume[i];
    obv[i] = cur;
  }
  return { obv };
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

function rollingSum(values: number[], period: number): Array<number | null> {
  const n = values.length;
  const out: Array<number | null> = new Array(n).fill(null);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    if (i >= period - 1) out[i] = sum;
  }
  return out;
}

/** 
 * Choppiness Index
 * Measures the market trendiness vs. choppiness over a specified period.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param period - The lookback period for Choppiness Index calculation (default: 14).
 * @returns An object containing an array for the Choppiness Index.
 */
export function choppinessSeries(high: number[], low: number[], close: number[], period = 14): { choppiness: Array<number | null> } {
  const n = close.length;
  const out: Array<number | null> = new Array(n).fill(null);
  if (n <= 1) return { choppiness: out };
  const tr: number[] = new Array(n).fill(0);
  tr[0] = 0;
  for (let i = 1; i < n; i++) tr[i] = Math.max(high[i] - low[i], Math.abs(high[i] - close[i - 1]), Math.abs(low[i] - close[i - 1]));
  const trSum = rollingSum(tr, period);
  const maxH = rollingMax(high, period);
  const minL = rollingMin(low, period);
  const logN = Math.log10(period);
  for (let i = 0; i < n; i++) {
    const s = trSum[i], h = maxH[i], l = minL[i];
    if (i >= period && s != null && h != null && l != null && h > l && s > 0) {
      out[i] = 100 * (Math.log10(s / (h - l)) / logN);
    }
  }
  return { choppiness: out };
}


// ---- Series implementations ----

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
          const seed = macdHist.reduce((a, b) => a + (b as number), 0) / signal;
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

/** Ichimoku Cloud indicator
 * @param high 
 * @param low
 * @param close
 * @param tenkan
 * @param kijun
 * @param senkouB
 * @returns
 */
export function ichimokuSeries(high: number[], low: number[], close: number[], tenkan = 9, kijun = 26, senkouB = 52): { tenkan: Array<number | null>; kijun: Array<number | null>; senkouA: Array<number | null>; senkouB: Array<number | null>; chikou: Array<number | null>; priceVsCloud: Array<'above' | 'in' | 'below' | null> } {
  const n = close.length;
  const tArr: Array<number | null> = new Array(n).fill(null);
  const kArr: Array<number | null> = new Array(n).fill(null);
  const sA: Array<number | null> = new Array(n).fill(null);
  const sB: Array<number | null> = new Array(n).fill(null);
  const cArr: Array<number | null> = new Array(n).fill(null);
  const classArr: Array<'above' | 'in' | 'below' | null> = new Array(n).fill(null);

  const hhTen = rollingMax(high, tenkan);
  const llTen = rollingMin(low, tenkan);
  const hhKij = rollingMax(high, kijun);
  const llKij = rollingMin(low, kijun);
  const hhB = rollingMax(high, senkouB);
  const llB = rollingMin(low, senkouB);

  for (let i = 0; i < n; i++) {
    const ht = hhTen[i], lt = llTen[i];
    const hk = hhKij[i], lk = llKij[i];
    const hb = hhB[i], lb = llB[i];
    if (ht != null && lt != null) tArr[i] = (ht + lt) / 2;
    if (hk != null && lk != null) kArr[i] = (hk + lk) / 2;
    if (tArr[i] != null && kArr[i] != null) sA[i] = (tArr[i]! + kArr[i]!) / 2;
    if (hb != null && lb != null) sB[i] = (hb + lb) / 2;
    if (i - kijun >= 0) cArr[i - kijun] = close[i];
    const c = close[i];
    const a = sA[i];
    const b = sB[i];
    if (a != null && b != null) {
      const up = Math.max(a, b);
      const down = Math.min(a, b);
      classArr[i] = c > up ? 'above' : (c < down ? 'below' : 'in');
    }
  }
  return { tenkan: tArr, kijun: kArr, senkouA: sA, senkouB: sB, chikou: cArr, priceVsCloud: classArr };
}

/** DMI/ADX (Directional Movement Index / Average Directional Index) indicator series
 * @param high 
 * @param low
 * @param close
 * @param period
 * @returns
 */
export function dmiAdxSeries(high: number[], low: number[], close: number[], period = 14): { plusDi: Array<number | null>; minusDi: Array<number | null>; adx: Array<number | null> } {
  const n = close.length;
  const plusDi: Array<number | null> = new Array(n).fill(null);
  const minusDi: Array<number | null> = new Array(n).fill(null);
  const adx: Array<number | null> = new Array(n).fill(null);
  if (n === 0) return { plusDi, minusDi, adx };

  const trArr: number[] = [];
  const plusDmArr: number[] = [];
  const minusDmArr: number[] = [];
  for (let i = 1; i < n; i++) {
    const tr = Math.max(high[i] - low[i], Math.abs(high[i] - close[i - 1]), Math.abs(low[i] - close[i - 1]));
    trArr.push(tr);
    const upMove = high[i] - high[i - 1];
    const downMove = low[i - 1] - low[i];
    plusDmArr.push(upMove > downMove && upMove > 0 ? upMove : 0);
    minusDmArr.push(downMove > upMove && downMove > 0 ? downMove : 0);
  }

  if (trArr.length < period) return { plusDi, minusDi, adx };

  let atrSum = trArr.slice(0, period).reduce((s, v) => s + v, 0);
  let sPlus = plusDmArr.slice(0, period).reduce((s, v) => s + v, 0);
  let sMinus = minusDmArr.slice(0, period).reduce((s, v) => s + v, 0);

  // First DI available at index (period)
  let diIndex = period;
  for (let i = period; i < trArr.length; i++) {
    atrSum = wilderSmooth(atrSum, trArr[i], period);
    sPlus = wilderSmooth(sPlus, plusDmArr[i], period);
    sMinus = wilderSmooth(sMinus, minusDmArr[i], period);
    const atrVal = atrSum;
    const pdi = atrVal === 0 ? 0 : (sPlus / atrVal) * 100;
    const mdi = atrVal === 0 ? 0 : (sMinus / atrVal) * 100;
    plusDi[i + 1] = pdi;
    minusDi[i + 1] = mdi;
  }

  // DX and ADX
  const dxArr: number[] = [];
  for (let i = diIndex; i < trArr.length; i++) {
    const pdi = plusDi[i + 1];
    const mdi = minusDi[i + 1];
    if (pdi == null || mdi == null) continue;
    const diSum = pdi + mdi;
    const dx = diSum === 0 ? 0 : (Math.abs(pdi - mdi) / diSum) * 100;
    dxArr.push(dx);
    if (dxArr.length === period) {
      adx[i + 1] = dxArr.reduce((s, v) => s + v, 0) / period;
    } else if (dxArr.length > period) {
      const prevAdx = adx[i] as number; // previous filled value
      adx[i + 1] = wilderSmooth(prevAdx, dx, period);
    }
  }
  return { plusDi, minusDi, adx };
}

/** SuperTrend indicator series
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

    const prevDir = dir[i - 1];
    const prevLine = line[i - 1];

    if (prevDir === 'up' && close[i] < (lowerBand[i] as number)) {
      dir[i] = 'down';
    } else if (prevDir === 'down' && close[i] > (upperBand[i] as number)) {
      dir[i] = 'up';
    } else {
      dir[i] = i > 0 ? prevDir : 'up';
    }

    line[i] = dir[i] === 'up' ? lowerBand[i] : upperBand[i];
  }
  return { line, dir };
}
