// Pure indicator functions (no side effects)
// Lightweight implementations with optional previous-state to enable rolling updates.

export type Num = number | null;

// ---- Trend ----

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

    const prevDir = dir[i - 1];
    const prevLine = line[i - 1];

    if (prevDir === 'up' && close[i] < (lowerBand[i] as number)) {
      dir[i] = 'down';
    } else if (prevDir === 'down' && close[i] > (upperBand[i] as number)) {
      dir[i] = 'up';
    } else {
      dir[i] = prevDir || 'up'; // Initialize first direction as 'up' if null
    }

    line[i] = dir[i] === 'up' ? lowerBand[i] : upperBand[i];
  }
  return { line, dir };
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

// Helper functions needed
function wilderSmooth(prevValue: number, currentValue: number, period: number): number {
  return (prevValue * (period - 1) + currentValue) / period;
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