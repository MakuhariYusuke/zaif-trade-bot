export type MACD = { macd: number|null, signal: number|null, hist: number|null }[];
export type ADX = { plusDI: number|null, minusDI: number|null, adx: number|null }[];
export type BBands = { basis: number|null, upper: number|null, lower: number|null, bandwidth: number|null, percentB: number|null }[];
export type Ichimoku = { tenkan: number|null, kijun: number|null, senkouA: number|null, senkouB: number|null, chikou: number|null, priceVsCloud?: ('above'|'in'|'below')|null }[];
export type SuperTrend = { line: number|null, dir: 'up'|'down'|null }[];

export interface IndicatorSnapshot {
  macd?: MACD[number];
  adx?: ADX[number];
  bbands?: BBands[number];
  ichimoku?: Ichimoku[number];
  supertrend?: SuperTrend[number];
}
