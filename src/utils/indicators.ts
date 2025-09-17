// Pure indicator functions (no side effects)
// Lightweight implementations with optional previous-state to enable rolling updates.

export type Num = number | null | undefined;

/**
 * Simple Moving Average (SMA)
 */
export function sma(values: number[], period: number): Num {
  if (!Array.isArray(values) || period <= 0 || values.length < period) return null;
  let s = 0; for (let i = values.length - period; i < values.length; i++) s += values[i];
  return s / period;
}

/**
 * Exponential Moving Average (EMA)
 */
export function ema(values: number[], period: number, prevEma?: number): Num {
  if (period <= 0 || values.length === 0) return null;
  const k = 2 / (period + 1);
  if (prevEma != null && values.length >= 1) return (values[values.length - 1] - prevEma) * k + prevEma;
  // seed with SMA of first period
  if (values.length < period) return null;
  const seed = sma(values.slice(0, period), period);
  if (seed == null) return null;
  let e = seed;
  for (let i = period; i < values.length; i++) e = (values[i] - e) * k + e;
  return e;
}

/**
 * Weighted Moving Average (WMA)
 */
export function wma(values: number[], period: number): Num {
  if (period <= 0 || values.length < period) return null;
  let num = 0, den = 0, w = 1;
  for (let i = values.length - period; i < values.length; i++) { num += values[i] * w; den += w; w++; }
  return num / den;
}

/**
 * Standard Deviation (STDDEV)
 */
export function stddev(values: number[], period: number): Num {
  if (period <= 0 || values.length < period) return null;
  const window = values.slice(values.length - period);
  const m = window.reduce((a, b) => a + b, 0) / period;
  const v = window.reduce((a, b) => a + Math.pow(b - m, 2), 0) / period;
  return Math.sqrt(v);
}

/** Bollinger Bands */
export function bollinger(values: number[], period = 20, k = 2): { basis: Num; upper: Num; lower: Num } {
  const basis = sma(values, period);
  const sd = stddev(values, period);
  if (basis == null || sd == null) return { basis: null, upper: null, lower: null };
  return { basis, upper: basis + k * sd, lower: basis - k * sd };
}

/**
 * Bollinger Band Width (BBW) in percent
 */
export function bbWidth(values: number[], period = 20, k = 2): Num {
  const bb = bollinger(values, period, k);
  if (bb.basis == null || bb.upper == null || bb.lower == null || bb.basis === 0) return null;
  return ((bb.upper - bb.lower) / bb.basis) * 100;
}

/**
 * Relative Strength Index (RSI) (Wilder's)
 */
export function rsi(values: number[], period = 14, prev?: { avgGain: number; avgLoss: number; value: number }): { value: Num; avgGain?: number; avgLoss?: number } {
  if (!Array.isArray(values) || values.length < 2) return { value: null };
  const len = values.length;
  const change = values[len - 1] - values[len - 2];
  const gain = Math.max(0, change);
  const loss = Math.max(0, -change);
  if (prev && prev.avgGain != null && prev.avgLoss != null) {
    const avgGain = (prev.avgGain * (period - 1) + gain) / period;
    const avgLoss = (prev.avgLoss * (period - 1) + loss) / period;
    const rs = avgLoss === 0 ? Infinity : avgGain / avgLoss;
    const value = 100 - 100 / (1 + rs);
    return { value, avgGain, avgLoss };
  }
  if (values.length < period + 1) return { value: null };
  let g = 0, l = 0;
  for (let i = len - period; i < len; i++) {
    const ch = values[i] - values[i - 1];
    g += Math.max(0, ch); l += Math.max(0, -ch);
  }
  const avgGain = g / period; const avgLoss = l / period;
  const rs = avgLoss === 0 ? Infinity : avgGain / avgLoss;
  const value = 100 - 100 / (1 + rs);
  return { value, avgGain, avgLoss };
}

/**
 * Moving Average Convergence Divergence (MACD)
 */
export function macd(values: number[], fast = 12, slow = 26, signal = 9, prev?: { emaFast?: number; emaSlow?: number; signal?: number }): { macd: Num; signal: Num; hist: Num; emaFast?: number; emaSlow?: number } {
  if (values.length === 0) return { macd: null, signal: null, hist: null };
  const efPrev = prev?.emaFast; const esPrev = prev?.emaSlow; const sigPrev = prev?.signal;
  const ef = ema(values, fast, efPrev);
  const es = ema(values, slow, esPrev);
  if (ef == null || es == null) return { macd: null, signal: null, hist: null, emaFast: ef ?? undefined, emaSlow: es ?? undefined };
  const line = ef - es;
  const sigArr = values.map(() => 0); // dummy length holder
  // MACD signal line: EMA of MACD line (uses previous signal value if available, otherwise returns MACD line if enough data)
    const sig = sigPrev != null ? (line - sigPrev) * (2 / (signal + 1)) + sigPrev : (values.length < slow + signal ? null : line);
  const hist = sig == null ? null : (line - sig);
  return { macd: line, signal: sig, hist, emaFast: ef, emaSlow: es };
}
   

/**
 * Envelope
 */
export function envelopes(ma: number | null, pct = 1.5): { upper: Num; lower: Num } {
  if (ma == null) return { upper: null, lower: null };
  const f = pct / 100;
  return { upper: ma * (1 + f), lower: ma * (1 - f) };
}

/**
 * Price Deviation Percentage (from moving average)
 */
export function deviationPct(price: number, ma: Num): Num {
  if (ma == null || ma === 0) return null;
  return ((price - ma) / ma) * 100;
}

/**
 * Rate of Change (ROC) in percent
 */
export function roc(values: number[], period = 14): Num {
  if (values.length <= period) return null;
  const prev = values[values.length - 1 - period];
  const cur = values[values.length - 1];
  if (prev === 0) return null;
  return ((cur - prev) / prev) * 100;
}

/** Momentum (difference) */
export function momentum(values: number[], period = 10): Num {
  if (values.length <= period) return null;
  const prev = values[values.length - 1 - period];
  const cur = values[values.length - 1];
  return cur - prev;
}

/** Stochastic Oscillator */
export function stochastic(high: number[], low: number[], close: number[], kPeriod = 14, dPeriod = 3, smooth = 3): { k: Num; d: Num } {
  if (close.length < kPeriod || high.length < kPeriod || low.length < kPeriod) return { k: null, d: null };
  const h = Math.max(...high.slice(-kPeriod));
  const l = Math.min(...low.slice(-kPeriod));
  const c = close[close.length - 1];
  const rawK = h === l ? 0 : ((c - l) / (h - l)) * 100;
  // smoothing %K
  const kArr: number[] = [];
  for (let i = close.length - kPeriod - smooth + 1; i <= close.length - kPeriod; i++) {
    if (i < 0) continue;
    const h = Math.max(...high.slice(i, i + kPeriod));
    const l = Math.min(...low.slice(i, i + kPeriod));
    const c = close[i + kPeriod - 1];
    const rawK = h === l ? 0 : ((c - l) / (h - l)) * 100;
    kArr.push(rawK);
  }
  kArr.push(rawK);
  const k = kArr.reduce((a, b) => a + b, 0) / kArr.length;
  // %D as SMA of last dPeriod of %K
  const dArr = kArr.slice(-dPeriod);
  const d = dArr.reduce((a, b) => a + b, 0) / dArr.length;
  return { k, d };
}

/**
 * Ichimoku (values only: tenkan, kijun, spanA, spanB, chikou)
 * @returns {
 *   tenkan: Num,
 *   kijun: Num,
 *   spanA: Num,
 *   spanB: Num,
 *   chikou: Num
 * }
 * tenkan: (9) Conversion Line
 * kijun: (26) Base Line
 * senkou span A: (26) Leading Span A
 * senkou span B: (52) Leading Span B
 * chikou: (closing price shifted back 26 periods)
 */
export function ichimoku(high: number[], low: number[], close: number[], tenkan = 9, kijun = 26, senkouB = 52): { tenkan: Num; kijun: Num; spanA: Num; spanB: Num; chikou: Num } {
  const conv = (p: number): Num => {
    if (high.length < p || low.length < p) return null;
    const h = Math.max(...high.slice(-p));
    const l = Math.min(...low.slice(-p));
    return (h + l) / 2;
  };
  const t = conv(tenkan); const k = conv(kijun);
  const b = conv(senkouB);
  const a = (t != null && k != null) ? (t + k) / 2 : null;
  const chikou = close.length ? close[close.length - 1] : null;
  return { tenkan: t, kijun: k, spanA: a, spanB: b, chikou };
}

/** DMI/ADX (simplified) */
export function dmiAdx(high: number[], low: number[], close: number[], period = 14): { adx: Num; plusDi: Num; minusDi: Num } {
  const n = close.length;
  if (n < period + 1) return { adx: null, plusDi: null, minusDi: null };
  let trSum = 0, plusDmSum = 0, minusDmSum = 0;
for (let i = n - period; i < n; i++) {
    const h = high[i], l = low[i], prevH = high[i - 1], prevL = low[i - 1], prevC = close[i - 1];
    trSum += Math.max(h - l, Math.abs(h - prevC), Math.abs(l - prevC));
    const upMove = h - prevH;
    const downMove = prevL - l;
    if (upMove > downMove && upMove > 0) plusDmSum += upMove;
    if (downMove > upMove && downMove > 0) minusDmSum += downMove;
}
  if (trSum === 0) return { adx: 0, plusDi: 0, minusDi: 0 };
  const plusDi = 100 * (plusDmSum / trSum);
  const minusDi = 100 * (minusDmSum / trSum);
  const diSum = plusDi + minusDi;
  if (diSum === 0) return { adx: null, plusDi: null, minusDi: null };
  const dx = Math.abs(plusDi - minusDi) / diSum * 100;
  // ADX as simple mean of DX over period (approx)
  const adx = dx; // simplified single-window approximation
  return { adx, plusDi, minusDi };
}

/** 
 * Williams %R 
 */
export function williamsR(high: number[], low: number[], close: number[], period = 14): Num {
  if (high.length < period || low.length < period || close.length < period) return null;
  const hh = Math.max(...high.slice(-period));
  const ll = Math.min(...low.slice(-period));
  const c = close[close.length - 1];
  if (hh === ll) return 0;
  return -100 * ((hh - c) / (hh - ll));
}

// CCI
export function cci(high: number[], low: number[], close: number[], period = 20): Num {
  if (high.length < period || low.length < period || close.length < period) return null;
  const tp = high.map((h, i) => (h + low[i] + close[i]) / 3);
  const last = tp.slice(-period);
  const avg = last.reduce((a, b) => a + b, 0) / period;
  const md = last.reduce((a, b) => a + Math.abs(b - avg), 0) / period;
  if (md === 0) return 0;
  const curTp = tp[tp.length - 1];
  return (curTp - avg) / (0.015 * md);
}

/** Average True Range (ATR) (simple SMA of True Range) */
export function atr(high: number[], low: number[], close: number[], period = 14): Num {
  const n = close.length;
  if (n < period + 1) return null;
  const trs: number[] = [];
  for (let i = n - period; i < n; i++) {
    const tr = Math.max(high[i] - low[i], Math.abs(high[i] - close[i - 1]), Math.abs(low[i] - close[i - 1]));
    trs.push(tr);
  }
  const s = trs.reduce((a, b) => a + b, 0);
  return s / period;
}

/** Donchian channel width (% of close) */
export function donchianWidth(high: number[], low: number[], close: number[], period = 20): Num {
  if (high.length < period || low.length < period || close.length === 0) return null;
  const hh = Math.max(...high.slice(-period));
  const ll = Math.min(...low.slice(-period));
  const c = close[close.length - 1];
  if (c === 0) return null;
  return ((hh - ll) / c) * 100;
}

/** Hull Moving Average (HMA) */
export function hma(values: number[], period = 20): Num {
  if (values.length < period) return null;
  const wmaP = (arr: number[], p: number) => wma(arr, p);
  const half = Math.floor(period / 2);
  const sqrtP = Math.max(1, Math.floor(Math.sqrt(period)));
  const wma1 = wma(values, half);
  // Construct rolling window for final WMA approximation
  const tempArr: number[] = [];
  for (let i = values.length - sqrtP; i < values.length; i++) {
    if (i < half) continue;
    const wma1i = wma(values.slice(i - half, i + 1), half);
    const wma2i = wma(values.slice(i - period, i + 1), period);
    if (wma1i == null || wma2i == null) continue;
    tempArr.push(2 * wma1i - wma2i);
  }
  if (tempArr.length < sqrtP) return null;
  return wmaP(tempArr, sqrtP);
}

/** KAMA (simplified; requires previous KAMA value optionally)
 * @param values - Array of price values
 * @param period - Lookback period (default: 10)
 * @param fast - Fast EMA period (default: 2)
 * @param slow - Slow EMA period (default: 30)
 * @param prevKama - Previous KAMA value (optional)
 * @returns KAMA value or null
 */
export function kama(values: number[], period = 10, fast = 2, slow = 30, prevKama?: number): Num {
  if (values.length <= period) return null;
  const change = Math.abs(values[values.length - 1] - values[values.length - 1 - period]);
  let volatility = 0;
  for (let i = values.length - period + 1; i < values.length; i++) volatility += Math.abs(values[i] - values[i - 1]);
  const er = volatility === 0 ? 0 : change / volatility;
  const fastSC = 2 / (fast + 1);
  const slowSC = 2 / (slow + 1);
  const sc = Math.pow(er * (fastSC - slowSC) + slowSC, 2);
  let prev: number;
  if (prevKama != null) {
    prev = prevKama;
  } else {
    const smaVal = sma(values.slice(0, period), period);
    if (smaVal != null) {
      prev = smaVal as number;
    } else {
      const idx = values.length - 1 - period;
      prev = idx >= 0 ? values[idx] : values[values.length - 1];
    }
  }
  return prev + sc * (values[values.length - 1] - prev);
}

/**
 * Parabolic SAR (one-step, needs previous state)
 * @param prev - Previous SAR state object:
 *   sar: number - Previous SAR value
 *   ep: number - Extreme point (highest high or lowest low)
 *   af: number - Acceleration factor
 *   uptrend: boolean - True if in uptrend, false if in downtrend
 * @param high - Current high price
 * @param low - Current low price
 * @param step - Acceleration factor step (default: 0.02)
 * @param max - Maximum acceleration factor (default: 0.2)
 * @returns Updated SAR state object:
 *   sar: number - New SAR value
 *   ep: number - Updated extreme point
 *   af: number - Updated acceleration factor
 *   uptrend: boolean - Updated trend direction
 */
export function psarStep(prev: { sar: number; ep: number; af: number; uptrend: boolean }, high: number, low: number, step = 0.02, max = 0.2) {
  let { sar, ep, af, uptrend } = prev;
  if (uptrend) {
    sar = sar + af * (ep - sar);
    if (high > ep) { ep = high; af = Math.min(max, af + step); }
    if (low < sar) { uptrend = false; sar = ep; ep = low; af = step; }
  } else {
    sar = sar + af * (ep - sar);
    if (low < ep) { ep = low; af = Math.min(max, af + step); }
    if (high > sar) { uptrend = true; sar = ep; ep = high; af = step; }
  }
  return { sar, ep, af, uptrend };
}

/** Fibonacci position (0..1 within recent high-low window) */
export function fibPosition(high: number[], low: number[], close: number[], lookback = 100): Num {
  if (high.length === 0 || low.length === 0 || close.length === 0) return null;
  const h = Math.max(...high.slice(-lookback));
  const l = Math.min(...low.slice(-lookback));
  const c = close[close.length - 1];
  const diff = h - l; if (!(diff > 0)) return null;
  return (c - l) / diff;
}

// (removed) fibonacciRetracement: not used anywhere; keep API minimal
/**
 * Fibonacci retracement levels
 */
export function fibonacciRetracement(high: number[], low: number[], lookback = 100): { levels: Record<string, number> | null } {
  if (high.length === 0 || low.length === 0) return { levels: null };
  const h = Math.max(...high.slice(-lookback));
  const l = Math.min(...low.slice(-lookback));
  const diff = h - l;
  if (!(diff > 0)) return { levels: null };
  const ratios: Record<string, number> = {
    '0.000': h,
    '0.236': h - 0.236 * diff,
    '0.382': h - 0.382 * diff,
    '0.500': h - 0.5 * diff,
    '0.618': h - 0.618 * diff,
    '0.786': h - 0.786 * diff,
    '1.000': l,
  };
  return { levels: ratios };
}
