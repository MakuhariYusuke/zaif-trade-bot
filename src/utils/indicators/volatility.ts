// Pure indicator functions (no side effects)
// Lightweight implementations with optional previous-state to enable rolling updates.

export type Num = number | null;

// ---- Volatility ----

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
 * Average True Range (ATR)
 * Calculates the ATR value for the most recent period.
 * @param high - Array of high prices.
 * @param low - Array of low prices.
 * @param close - Array of close prices.
 * @param period - The lookback period for ATR calculation (default: 14).
 * @returns The ATR value, or null if there is insufficient data or invalid period.
 */
export function atr(high: number[], low: number[], close: number[], period: number = 14): Num {
  if (period <= 0 || high.length < period + 1 || low.length < period + 1 || close.length < period + 1) return null;
  if (high.length !== low.length || high.length !== close.length) return null;

  // Calculate True Range for each period
  const trArr: number[] = [];
  for (let i = 1; i < Math.min(high.length, low.length, close.length); i++) {
    const tr = Math.max(
      high[i] - low[i],
      Math.abs(high[i] - close[i - 1]),
      Math.abs(low[i] - close[i - 1])
    );
    trArr.push(tr);
  }

  if (trArr.length < period) return null;

  // Calculate initial ATR
  let atr = trArr.slice(0, period).reduce((sum, tr) => sum + tr, 0) / period;

  // Apply Wilder's smoothing for remaining values
  for (let i = period; i < trArr.length; i++) {
    atr = wilderSmooth(atr, trArr[i], period);
  }

  return atr;
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

// Helper functions needed
function sma(values: number[], period: number): Num {
  if (!Array.isArray(values) || period <= 0 || values.length < period) return null;
  const window = values.slice(-period);
  const sum = window.reduce((acc, val) => acc + val, 0);
  return sum / period;
}

function wilderSmooth(prevValue: number, currentValue: number, period: number): number {
  return (prevValue * (period - 1) + currentValue) / period;
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