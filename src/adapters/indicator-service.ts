import { ema, rsi, sma, macd, bollinger, stochastic, ichimoku, dmiAdx, envelopes, deviationPct, roc, momentum, williamsR, cci, atr, donchianWidth, bbWidth, hma, kama, psarStep, fibPosition } from '../utils/indicators';
import BaseService from './base-service';

export interface IndicatorSnapshot {
  ts: number;
  price: number;
  sma_short?: number|null;
  sma_long?: number|null;
  ema_short?: number|null;
  ema_long?: number|null;
  wma_short?: number|null; // reserved if needed in future
  hma_long?: number|null;
  kama?: number|null;
  rsi?: number|null;
  roc?: number|null;
  mom?: number|null;
  willr?: number|null;
  cci?: number|null;
  macd?: number|null; macdSignal?: number|null; macdHist?: number|null;
  bb_basis?: number|null; bb_upper?: number|null; bb_lower?: number|null; bb_width?: number|null;
  don_width?: number|null;
  atr?: number|null;
  stoch_k?: number|null; stoch_d?: number|null;
  ichi_tenkan?: number|null; ichi_kijun?: number|null; ichi_spanA?: number|null; ichi_spanB?: number|null; ichi_chikou?: number|null;
  adx?: number|null; plusDi?: number|null; minusDi?: number|null;
  env_upper?: number|null; env_lower?: number|null;
  dev_pct?: number|null;
  psar?: number|null;
  fib_pos?: number|null;
}

export interface IndicatorServiceOptions {
  smaShort?: number; smaLong?: number;
  emaShort?: number; emaLong?: number;
  wmaShort?: number; // placeholder
  hmaLong?: number;
  kamaPeriod?: number; kamaFast?: number; kamaSlow?: number;
  rsiPeriod?: number;
  rocPeriod?: number; momentumPeriod?: number; willrPeriod?: number; cciPeriod?: number;
  macdFast?: number; macdSlow?: number; macdSignal?: number;
  bbPeriod?: number; bbK?: number;
  bbWidthK?: number;
  donPeriod?: number;
  stochK?: number; stochD?: number; stochSmooth?: number;
  ichiTenkan?: number; ichiKijun?: number; ichiSenkouB?: number;
  dmiPeriod?: number; atrPeriod?: number;
  envPct?: number;
  devMaPeriod?: number;
  approxFromBidAsk?: boolean;
}

export class IndicatorService extends BaseService {
  private prices: number[] = [];
  private highs: number[] = [];
  private lows: number[] = [];
  private closes: number[] = [];
  private opts: Required<IndicatorServiceOptions>;
  private prev = { rsi: undefined as any, macd: undefined as any };
  private prevKama: number | undefined;
  private psarState: { sar: number; ep: number; af: number; uptrend: boolean } | undefined;

  constructor(opts: IndicatorServiceOptions = {}){
    super();
    this.opts = {
      smaShort: opts.smaShort ?? 9,
      smaLong: opts.smaLong ?? 26,
      emaShort: opts.emaShort ?? 12,
      emaLong: opts.emaLong ?? 26,
  wmaShort: opts.wmaShort ?? 10,
  hmaLong: opts.hmaLong ?? 55,
  kamaPeriod: opts.kamaPeriod ?? 10,
  kamaFast: opts.kamaFast ?? 2,
  kamaSlow: opts.kamaSlow ?? 30,
      rsiPeriod: opts.rsiPeriod ?? 14,
  rocPeriod: opts.rocPeriod ?? 14,
  momentumPeriod: opts.momentumPeriod ?? 10,
  willrPeriod: opts.willrPeriod ?? 14,
  cciPeriod: opts.cciPeriod ?? 20,
      macdFast: opts.macdFast ?? 12,
      macdSlow: opts.macdSlow ?? 26,
      macdSignal: opts.macdSignal ?? 9,
      bbPeriod: opts.bbPeriod ?? 20,
      bbK: opts.bbK ?? 2,
  bbWidthK: opts.bbWidthK ?? 2,
  donPeriod: opts.donPeriod ?? 20,
      stochK: opts.stochK ?? 14,
      stochD: opts.stochD ?? 3,
      stochSmooth: opts.stochSmooth ?? 3,
      ichiTenkan: opts.ichiTenkan ?? 9,
      ichiKijun: opts.ichiKijun ?? 26,
      ichiSenkouB: opts.ichiSenkouB ?? 52,
  dmiPeriod: opts.dmiPeriod ?? 14,
  atrPeriod: opts.atrPeriod ?? 14,
      envPct: opts.envPct ?? 1.5,
      devMaPeriod: opts.devMaPeriod ?? 20,
      approxFromBidAsk: opts.approxFromBidAsk ?? true,
    };
  }

  update(ts: number, price: number, bestBid?: number, bestAsk?: number): IndicatorSnapshot {
    // optional lightweight category log every N (controlled by caller/env)
    try { this.clog('IND', 'DEBUG', 'update', { ts, price, bestBid, bestAsk }); } catch {}
    this.prices.push(price);
    this.closes.push(price);
    // approx OHLC from best bid/ask if provided
    const high = (this.opts.approxFromBidAsk && bestAsk != null) ? Math.max(price, bestAsk) : price;
    const low = (this.opts.approxFromBidAsk && bestBid != null) ? Math.min(price, bestBid) : price;
    this.highs.push(high);
    this.lows.push(low);
    // truncate to reasonable window to bound memory
    const maxLen = 300;
    const trim = (arr: any[]) => { while (arr.length > maxLen) arr.shift(); };
    [this.prices, this.highs, this.lows, this.closes].forEach(trim);

    const o: IndicatorSnapshot = { ts, price };
    o.sma_short = sma(this.prices, this.opts.smaShort) ?? null;
    o.sma_long = sma(this.prices, this.opts.smaLong) ?? null;
    o.ema_short = ema(this.prices, this.opts.emaShort, undefined) ?? null;
    o.ema_long = ema(this.prices, this.opts.emaLong, undefined) ?? null;
    o.hma_long = hma(this.prices, this.opts.hmaLong) ?? null;
    this.prevKama = kama(this.prices, this.opts.kamaPeriod, this.opts.kamaFast, this.opts.kamaSlow, this.prevKama) ?? this.prevKama;
    o.kama = this.prevKama ?? null;
    const r = rsi(this.prices, this.opts.rsiPeriod, this.prev.rsi);
    o.rsi = r.value ?? null; this.prev.rsi = r;
    o.roc = roc(this.prices, this.opts.rocPeriod) ?? null;
    o.mom = momentum(this.prices, this.opts.momentumPeriod) ?? null;
    const m = macd(this.prices, this.opts.macdFast, this.opts.macdSlow, this.opts.macdSignal, this.prev.macd);
    o.macd = m.macd ?? null; o.macdSignal = m.signal ?? null; o.macdHist = m.hist ?? null; this.prev.macd = m;
    const bb = bollinger(this.prices, this.opts.bbPeriod, this.opts.bbK);
    o.bb_basis = bb.basis ?? null; o.bb_upper = bb.upper ?? null; o.bb_lower = bb.lower ?? null;
    o.bb_width = bbWidth(this.prices, this.opts.bbPeriod, this.opts.bbWidthK) ?? null;
    o.atr = atr(this.highs, this.lows, this.closes, this.opts.atrPeriod) ?? null;
    o.don_width = donchianWidth(this.highs, this.lows, this.closes, this.opts.donPeriod) ?? null;
    o.willr = williamsR(this.highs, this.lows, this.closes, this.opts.willrPeriod) ?? null;
    o.cci = cci(this.highs, this.lows, this.closes, this.opts.cciPeriod) ?? null;
    const st = stochastic(this.highs, this.lows, this.closes, this.opts.stochK, this.opts.stochD, this.opts.stochSmooth);
    o.stoch_k = st.k ?? null; o.stoch_d = st.d ?? null;
    const ich = ichimoku(this.highs, this.lows, this.closes, this.opts.ichiTenkan, this.opts.ichiKijun, this.opts.ichiSenkouB);
    o.ichi_tenkan = ich.tenkan ?? null; o.ichi_kijun = ich.kijun ?? null; o.ichi_spanA = ich.spanA ?? null; o.ichi_spanB = ich.spanB ?? null; o.ichi_chikou = ich.chikou ?? null;
    const dmi = dmiAdx(this.highs, this.lows, this.closes, this.opts.dmiPeriod);
    o.adx = dmi.adx ?? null; o.plusDi = dmi.plusDi ?? null; o.minusDi = dmi.minusDi ?? null;
    const env = envelopes(o.sma_long ?? o.sma_short ?? null, this.opts.envPct);
    o.env_upper = env.upper ?? null; o.env_lower = env.lower ?? null;
    const dev = deviationPct(price, o.sma_long ?? o.sma_short ?? null);
    o.dev_pct = dev ?? null;
    // PSAR one-step update requires previous state
    try {
      const lastH = this.highs[this.highs.length - 1];
      const lastL = this.lows[this.lows.length - 1];
      if (lastH != null && lastL != null) {
        if (!this.psarState) this.psarState = { sar: this.closes[0] ?? price, ep: lastH, af: 0.02, uptrend: true };
        this.psarState = psarStep(this.psarState, lastH, lastL);
        o.psar = this.psarState.sar;
      }
    } catch {}
    o.fib_pos = fibPosition(this.highs, this.lows, this.closes, 100) ?? null;
  try { this.clog('IND', 'DEBUG', 'snapshot', { ts: o.ts, price: o.price, rsi: o.rsi, smaS: o.sma_short, emaS: o.ema_short, macd: o.macd, dev: o.dev_pct }); } catch {}
  return o;
  }
}
