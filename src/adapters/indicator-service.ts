import { ema, rsi, sma, macd, bollinger, stochastic, ichimoku, dmiAdx, envelopes, deviationPct, roc, momentum, williamsR, cci, atr, donchianWidth, bbWidth, hma, kama, psarStep, fibPosition, supertrend, heikinAshiSeries, vortexSeries, aroonSeries, tsiSeries, mfiSeries, obvSeries, keltnerSeries, donchianSeries, choppinessSeries } from '../utils/indicators';
import BaseService from './base-service';

interface IndicatorSnapshot {
  ts: number;
  price: number;
  // trend / price-based averages
  sma_short?: number | null;
  sma_long?: number | null;
  ema_short?: number | null;
  ema_long?: number | null;
  wma_short?: number | null; // reserved if needed in future
  hma_long?: number | null;
  kama?: number | null;
  rsi?: number | null;
  roc?: number | null;
  mom?: number | null;
  willr?: number | null;
  cci?: number | null;
  macd?: number | null; macdSignal?: number | null; macdHist?: number | null;
  bb_basis?: number | null; bb_upper?: number | null; bb_lower?: number | null; bb_width?: number | null;
  don_width?: number | null;
  atr?: number | null;
  stoch_k?: number | null; stoch_d?: number | null;
  ichi_tenkan?: number | null; ichi_kijun?: number | null; ichi_spanA?: number | null; ichi_spanB?: number | null; ichi_chikou?: number | null;
  adx?: number | null; plusDi?: number | null; minusDi?: number | null;
  env_upper?: number | null; env_lower?: number | null;
  dev_pct?: number | null;
  psar?: number | null;
  fib_pos?: number | null;
  supertrend?: number | null;
  supertrend_dir?: 1 | -1 | null;
  // aliases for conventional names (kept for compatibility with tests)
  rsi14?: number | null;
  atr14?: number | null;
  // new indicators
  ha_open?: number | null; ha_high?: number | null; ha_low?: number | null; ha_close?: number | null;
  viPlus?: number | null; viMinus?: number | null;
  aroonUp?: number | null; aroonDown?: number | null; aroonOsc?: number | null;
  tsi?: number | null; tsiSignal?: number | null;
  mfi?: number | null; obv?: number | null;
  kc_basis?: number | null; kc_upper?: number | null; kc_lower?: number | null;
  don_upper?: number | null; don_lower?: number | null; don_mid?: number | null;
  choppiness?: number | null;
}

interface IndicatorServiceOptions {
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
  supertrendPeriod?: number; supertrendMult?: number;
  envPct?: number;
  devMaPeriod?: number;
  approxFromBidAsk?: boolean;
  // new indicator options
  vortexPeriod?: number;
  aroonPeriod?: number;
  tsiShort?: number; tsiLong?: number; tsiSignal?: number;
  mfiPeriod?: number; obvEnabled?: boolean;
  kcPeriod?: number; kcMult?: number;
  donchianPeriod?: number;
  choppinessPeriod?: number;
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

  constructor(opts: IndicatorServiceOptions = {}) {
    super();
    const envNum = (k: string, def: number) => { const v = Number(process.env[k]); return Number.isFinite(v) && v > 0 ? v : def; };
    const envListNum = (k: string): number[] | null => {
      const s = process.env[k]; if (!s) return null; const parts = s.split(',').map(x => Number(x.trim())).filter(x => Number.isFinite(x)); return parts.length ? parts : null;
    };
    const envBool = (k: string, def: boolean) => { const v = process.env[k]; if (v == null) return def; return v === '1' || v?.toLowerCase?.() === 'true'; };
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
      supertrendPeriod: opts.supertrendPeriod ?? 10,
      supertrendMult: opts.supertrendMult ?? 3,
      envPct: opts.envPct ?? 1.5,
      devMaPeriod: opts.devMaPeriod ?? 20,
      approxFromBidAsk: opts.approxFromBidAsk ?? true,
      vortexPeriod: opts.vortexPeriod ?? envNum('IND_VORTEX', 14),
      aroonPeriod: opts.aroonPeriod ?? envNum('IND_AROON', 14),
      ...(() => { const a = envListNum('IND_TSI'); const [s, l, g] = a ?? []; return { tsiShort: opts.tsiShort ?? (s || 13), tsiLong: opts.tsiLong ?? (l || 25), tsiSignal: opts.tsiSignal ?? (g || 13) }; })(),
      mfiPeriod: opts.mfiPeriod ?? envNum('IND_MFI', 14),
      obvEnabled: opts.obvEnabled ?? envBool('IND_OBV', true),
      kcPeriod: opts.kcPeriod ?? (envListNum('IND_KC')?.[0] ?? 20),
      kcMult: opts.kcMult ?? (envListNum('IND_KC')?.[1] ?? 2),
      donchianPeriod: opts.donchianPeriod ?? envNum('IND_DONCHIAN', 20),
      choppinessPeriod: opts.choppinessPeriod ?? envNum('IND_CHOP', 14),
    };
  }

  update(ts: number, price: number, bestBid?: number, bestAsk?: number): IndicatorSnapshot {
    // optional lightweight category log every N (controlled by caller/env)
    try { this.clog('IND', 'DEBUG', 'update', { ts, price, bestBid, bestAsk }); } catch { }
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
    const stp = this.opts.supertrendPeriod, stm = this.opts.supertrendMult;
    const stres = supertrend(this.highs, this.lows, this.closes, stp, stm);
    o.supertrend = stres.value ?? null; o.supertrend_dir = stres.dir ?? null;
    // New: Heikin-Ashi (approximate open as previous close)
    try {
      const n = this.closes.length;
      if (n >= 1) {
        const openArr: number[] = new Array(n);
        for (let i = 0; i < n; i++) openArr[i] = i === 0 ? this.closes[0] : this.closes[i - 1];
        const ha = heikinAshiSeries(openArr, this.highs, this.lows, this.closes);
        o.ha_open = ha.open[n - 1] ?? null; o.ha_high = ha.high[n - 1] ?? null; o.ha_low = ha.low[n - 1] ?? null; o.ha_close = ha.close[n - 1] ?? null;
      }
    } catch {
      // once-per-cycle warn not needed here; HA can be approximated
    }
    // Vortex
    try {
      const vi = vortexSeries(this.highs, this.lows, this.closes, this.opts.vortexPeriod);
      o.viPlus = vi.viPlus[vi.viPlus.length - 1] ?? null;
      o.viMinus = vi.viMinus[vi.viMinus.length - 1] ?? null;
    } catch { }
    // Aroon
    try {
      const ar = aroonSeries(this.highs, this.lows, this.opts.aroonPeriod);
      o.aroonUp = ar.aroonUp[ar.aroonUp.length - 1] ?? null;
      o.aroonDown = ar.aroonDown[ar.aroonDown.length - 1] ?? null;
      o.aroonOsc = ar.aroonOsc[ar.aroonOsc.length - 1] ?? null;
    } catch { }
    // TSI
    try {
      const tsis = tsiSeries(this.closes, this.opts.tsiShort!, this.opts.tsiLong!, this.opts.tsiSignal!);
      o.tsi = tsis.tsi[tsis.tsi.length - 1] ?? null;
      o.tsiSignal = tsis.signal[tsis.signal.length - 1] ?? null;
    } catch { }
    // MFI (needs volume) -> missing in this service, emit WARN once per process
    try {
      const g: any = global as any;
      const warnKey = '__ind_warn_mfi_missing';
  if (!g[warnKey]) { g[warnKey] = 1; this.clog('IND', 'WARN', 'missing', { indicator: 'mfi', reason: 'no volume series' }); try { if (process.env.QUIET === '1') console.warn(`[WARN][IND] missing {"indicator":"mfi","reason":"no volume series"}`); } catch {} }
      o.mfi = null;
    } catch { }
    // OBV (needs volume)
    try {
      const g: any = global as any;
      const warnKey = '__ind_warn_obv_missing';
  if (this.opts.obvEnabled && !g[warnKey]) { g[warnKey] = 1; this.clog('IND', 'WARN', 'missing', { indicator: 'obv', reason: 'no volume series' }); try { if (process.env.QUIET === '1') console.warn(`[WARN][IND] missing {"indicator":"obv","reason":"no volume series"}`); } catch {} }
      o.obv = null;
    } catch { }
    // Keltner
    try {
      const kc = keltnerSeries(this.highs, this.lows, this.closes, this.opts.kcPeriod!, this.opts.kcMult!);
      o.kc_basis = kc.basis[kc.basis.length - 1] ?? null;
      o.kc_upper = kc.upper[kc.upper.length - 1] ?? null;
      o.kc_lower = kc.lower[kc.lower.length - 1] ?? null;
    } catch { }
    // Donchian Channels
    try {
      const dc = donchianSeries(this.highs, this.lows, this.opts.donchianPeriod!);
      o.don_upper = dc.upper[dc.upper.length - 1] ?? null;
      o.don_lower = dc.lower[dc.lower.length - 1] ?? null;
      o.don_mid = dc.mid[dc.mid.length - 1] ?? null;
    } catch { }
    // Choppiness
    try {
      const ch = choppinessSeries(this.highs, this.lows, this.closes, this.opts.choppinessPeriod!);
      o.choppiness = ch.choppiness[ch.choppiness.length - 1] ?? null;
    } catch { }
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
    } catch { }
    o.fib_pos = fibPosition(this.highs, this.lows, this.closes, 100) ?? null;
    try { this.clog('IND', 'DEBUG', 'snapshot', { ts: o.ts, price: o.price, rsi: o.rsi, smaS: o.sma_short, emaS: o.ema_short, macd: o.macd, dev: o.dev_pct }); } catch { }
    // add aliases expected by some consumers/tests
    o.rsi14 = o.rsi ?? null;
    o.atr14 = o.atr ?? null;
    return o;
  }
}
