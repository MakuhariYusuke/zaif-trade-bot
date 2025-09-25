// Pure indicator functions (no side effects)
// Lightweight implementations with optional previous-state to enable rolling updates.

export type Num = number | null;

// ---- Utils ----

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
 * A helper function for Wilder's smoothing (a specific type of EMA with alpha = 1/period).
 * @param prevValue The previous smoothed value.
 * @param currentValue The current raw value.
 * @param period The smoothing period.
 * @returns The new smoothed value.
 */
export function wilderSmooth(prevValue: number, currentValue: number, period: number): number {
  return (prevValue * (period - 1) + currentValue) / period;
}

/**
 * Calculates the rolling maximum value over a specified window period.
 * @param values - Array of numeric values.
 * @param period - The size of the rolling window.
 * @returns An array where each element is the maximum value in the window, or null if not enough data.
 */
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

// Memoized versions for performance
export const rollingMaxMemo = memoize(rollingMax, (values, period) => `${period}-${JSON.stringify(values)}`);
export const rollingMinMemo = memoize(rollingMin, (values, period) => `${period}-${JSON.stringify(values)}`);

// Export rollingArgMax and rollingArgMin
export const rollingArgMax = memoize(_rollingArgMax, (values: number[], period: number) => `${period}-${JSON.stringify(values)}`);
export const rollingArgMin = memoize(_rollingArgMin, (values: number[], period: number) => `${period}-${JSON.stringify(values)}`);

/**
 * Calculates the rolling sum over a specified window period.
 * @param values - Array of numeric values.
 * @param period - The size of the rolling window.
 * @returns An array where each element is the sum of values in the window, or null if not enough data.
 */
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
 * Rolling Arg Max
 * Calculates the index of the maximum value in a rolling window.
 * @param values - Array of numeric values.
 * @param period - The size of the rolling window.
 * @returns An array of indices corresponding to the maximum values in each window.
 */
function _rollingArgMax(values: number[], period: number): Array<number | null> {
  const n = values.length;
  const out: Array<number | null> = new Array(n).fill(null);
  if (period <= 0) {
    // For period <= 0, return indices directly
    for (let i = 0; i < n; i++) out[i] = i;
    return out;
  }
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
function _rollingArgMin(values: number[], period: number): Array<number | null> {
  const n = values.length;
  const out: Array<number | null> = new Array(n).fill(null);
  if (period <= 0) {
    // For period <= 0, return indices directly
    for (let i = 0; i < n; i++) out[i] = i;
    return out;
  }
  const dq: number[] = [];
  for (let i = 0; i < n; i++) {
    while (dq.length && dq[0] <= i - period) dq.shift();
    while (dq.length && values[dq[dq.length - 1]] >= values[i]) dq.pop();
    dq.push(i);
    if (i >= period - 1) out[i] = dq[0];
  }
  return out;
}