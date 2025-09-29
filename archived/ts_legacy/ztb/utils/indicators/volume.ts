// Pure indicator functions (no side effects)
// Lightweight implementations with optional previous-state to enable rolling updates.

export type Num = number | null;

// ---- Volume ----

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