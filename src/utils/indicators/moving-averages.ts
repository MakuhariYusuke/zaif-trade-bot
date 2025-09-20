// Pure indicator functions (no side effects)
// Lightweight implementations with optional previous-state to enable rolling updates.

export type Num = number | null;

// ---- Moving Averages ----

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