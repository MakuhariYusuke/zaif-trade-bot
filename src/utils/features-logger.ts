import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { todayStr } from './toolkit';
import zlib from 'zlib';
import { IndicatorService } from '../adapters/indicator-service';
import { getPriceSeries } from './price-cache';
import { logDebug, logWarn } from './logger';

export interface FeatureSample {
  ts: number;
  pair: string;
  side: 'bid' | 'ask';
  rsi?: number | null;
  sma_short?: number | null;
  sma_long?: number | null;
  ema_short?: number | null;
  ema_long?: number | null;
  wma_short?: number | null;
  hma_long?: number | null;
  kama?: number | null;
  macd?: number | null;
  macdSignal?: number | null;
  macdHist?: number | null;
  bb_basis?: number | null;
  bb_upper?: number | null;
  bb_lower?: number | null;
  bb_width?: number | null;
  don_width?: number | null;
  atr?: number | null;
  stoch_k?: number | null;
  stoch_d?: number | null;
  ichi_tenkan?: number | null;
  ichi_kijun?: number | null;
  ichi_spanA?: number | null;
  ichi_spanB?: number | null;
  ichi_chikou?: number | null;
  adx?: number | null;
  plusDi?: number | null;
  minusDi?: number | null;
  env_upper?: number | null;
  env_lower?: number | null;
  dev_pct?: number | null;
  // aliases for conventional names
  rsi14?: number | null;
  atr14?: number | null;
  macd_hist?: number | null;
  roc?: number | null;
  mom?: number | null;
  willr?: number | null;
  cci?: number | null;
  psar?: number | null;
  fib_pos?: number | null;
  price: number;
  qty: number;
  pnl?: number;
  win?: boolean;
  position?: { qty?: number; side?: string; avgPrice?: number };
  balance?: { jpy?: number; btc?: number; eth?: number; xrp?: number };
  bestBid?: number;
  bestAsk?: number;
  spread?: number;    // (ask - bid) / mid
  slippage?: number;  // (fillPrice - mid) / mid
  status?: 'cancelled' | 'failed' | 'filled' | 'partial' | string;
  depthBid?: number;  // best bid size
  depthAsk?: number;  // best ask size
  volumeRecent?: number; // recent traded volume (e.g., last 60s)
}

// --- Buffered async logger with rotation (JSONL only) ---

type BufferEntry = { path: string; lines: string[] };
const buffers = new Map<string, BufferEntry>();
let flushTimer: NodeJS.Timeout | null = null;
let intervalTimer: NodeJS.Timeout | null = null;
// Treat explicit TEST_MODE=1 as test; otherwise allow interval when TEST_MODE is '0' even under vitest
const __IS_TEST__ = process.env.TEST_MODE === '1';
const FLUSH_INTERVAL_MS = Math.max(0, Number(process.env.FEATURES_FLUSH_INTERVAL_MS || '1000'));
const IND_LOG_EVERY_N = Math.max(0, Number(process.env.IND_LOG_EVERY_N || '0'));
const indicatorServices = new Map<string, IndicatorService>();

async function ensureDir(dir: string){
  try { await fsp.mkdir(dir, { recursive: true }); } catch {}
}

function scheduleFlush(delayMs = 300){
  if (flushTimer) return;
  flushTimer = setTimeout(async ()=>{
    flushTimer = null;
    await flushFeatureLogBuffers();
  }, delayMs).unref?.();
}

// background time-based flushing (disabled in tests)
try {
  if (!__IS_TEST__ && FLUSH_INTERVAL_MS > 0 && !intervalTimer) {
    intervalTimer = setInterval(async ()=>{
      try { await flushFeatureLogBuffers(); } catch {}
    }, FLUSH_INTERVAL_MS);
    intervalTimer.unref?.();
  }
} catch {}

export async function flushFeatureLogBuffers(){
  const entries = Array.from(buffers.values());
  buffers.clear();
  await Promise.all(entries.map(async ({ path: p, lines }) => {
    try {
      await ensureDir(path.dirname(p));
      const content = lines.join('');
      await fsp.appendFile(p, content, { encoding: 'utf8' });
    } catch {}
  }));
}

async function writeLatestJson(base: string, s: FeatureSample){
  const jsonPath = path.join(base, 'features', `latest-${s.pair}.json`);
  const isTest = process.env.TEST_MODE === '1' || !!process.env.VITEST_WORKER_ID;
  try {
    if (isTest) {
      // テストでは同期化して直後の読み取りを安定化
      try { fs.mkdirSync(path.dirname(jsonPath), { recursive: true }); } catch {}
      try { fs.writeFileSync(jsonPath, JSON.stringify(s, null, 2), { encoding: 'utf8' }); } catch {}
    } else {
      await ensureDir(path.dirname(jsonPath));
      await fsp.writeFile(jsonPath, JSON.stringify(s, null, 2), 'utf8');
    }
  } catch {}
}

async function gzipFile(inPath: string){
  const outPath = inPath + '.gz';
  if (fs.existsSync(outPath)) return;
  try {
    const src = fs.createReadStream(inPath);
    const dst = fs.createWriteStream(outPath);
    await new Promise<void>((resolve, reject)=>{
      src.pipe(zlib.createGzip()).pipe(dst).on('finish', ()=>resolve()).on('error', reject);
    });
    await fsp.unlink(inPath).catch(()=>{});
  } catch {}
}

async function rotateOldFeatureLogs(base: string){
  const days = Number(process.env.FEATURES_LOG_ROTATE_DAYS || '7');
  if (!(days > 0)) return;
  const cutoff = Date.now() - days * 86400_000;
  const root = path.join(base, 'features');
  const walk = async (dir: string) => {
    let entries: fs.Dirent[] = [];
    try { entries = await fsp.readdir(dir, { withFileTypes: true }) as any; } catch { return; }
    await Promise.all(entries.map(async e => {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) return walk(p);
      if (p.endsWith('.gz')) return;
      try {
        const st = await fsp.stat(p);
        if (st.mtimeMs < cutoff) await gzipFile(p);
      } catch {}
    }));
  };
  await walk(root);
}

function pushBuffer(filePath: string, line: string){
  const entry = buffers.get(filePath) || { path: filePath, lines: [] };
  entry.lines.push(line);
  buffers.set(filePath, entry);
  if (entry.lines.length >= Number(process.env.FEATURES_BUFFER_LINES || '200')) scheduleFlush(50);
  else scheduleFlush(300);
}

export function logFeatureSample(s: FeatureSample){
  const date = todayStr();
  const base = process.env.FEATURES_LOG_DIR ? path.resolve(process.env.FEATURES_LOG_DIR) : path.resolve(process.cwd(), 'logs');
  const src = process.env.FEATURES_SOURCE; // optional: 'paper' | 'live'
  const dir = src ? path.join(base, 'features', src, s.pair) : path.join(base, 'features', s.pair);
  const isTest = process.env.TEST_MODE === '1' || !!process.env.VITEST_WORKER_ID;
  // derive spread/slippage if not provided and bestBid/Ask present
  let spread = s.spread;
  let slippage = s.slippage;
  if ((spread == null || slippage == null) && s.bestAsk != null && s.bestBid != null){
    const mid = (s.bestAsk + s.bestBid) / 2;
    if (spread == null) spread = (s.bestAsk - s.bestBid) / mid;
    if (slippage == null) slippage = (s.price - mid) / mid;
  }
  // compute indicators
  try {
    const key = `${s.pair}:${src||'root'}`;
    let svc = indicatorServices.get(key);
    if (!svc) {
      svc = new IndicatorService({});
      indicatorServices.set(key, svc);
      try {
        const series = getPriceSeries(200);
        if (Array.isArray(series) && series.length > 0) {
          const n = Math.min(series.length, 200);
          const start = s.ts - n * 1000;
          for (let i = series.length - n; i < series.length; i++) {
            const t = start + (i - (series.length - n)) * 1000;
            const p = Number(series[i]);
            if (Number.isFinite(p)) svc.update(t, p);
          }
        }
      } catch {}
    }
    const ind = svc.update(s.ts, s.price, s.bestBid, s.bestAsk);
    if (IND_LOG_EVERY_N > 0) {
      const ctrKey = `__ind_ctr_${key}`;
      const g: any = global as any;
      g[ctrKey] = (g[ctrKey] || 0) + 1;
      if (g[ctrKey] % IND_LOG_EVERY_N === 0) {
        try { logDebug(`[IND] pair=${s.pair} rsi=${ind.rsi?.toFixed?.(2)} smaS=${ind.sma_short?.toFixed?.(2)} emaS=${ind.ema_short?.toFixed?.(2)} macd=${ind.macd?.toFixed?.(4)} dev=${ind.dev_pct?.toFixed?.(2)}%`); } catch {}
      }
    }
    s = {
      ...s,
      rsi: s.rsi ?? (ind.rsi ?? null),
      roc: (s as any).roc ?? (ind.roc ?? null),
      mom: (s as any).mom ?? (ind.mom ?? null),
      willr: (s as any).willr ?? (ind.willr ?? null),
      cci: (s as any).cci ?? (ind.cci ?? null),
      sma_short: s.sma_short ?? (ind.sma_short ?? null),
      sma_long: s.sma_long ?? (ind.sma_long ?? null),
      ema_short: (s as any).ema_short ?? (ind.ema_short ?? null),
      ema_long: (s as any).ema_long ?? (ind.ema_long ?? null),
      wma_short: (s as any).wma_short ?? (null as any),
      hma_long: (s as any).hma_long ?? (ind.hma_long ?? null),
      kama: (s as any).kama ?? (ind.kama ?? null),
      macd: (s as any).macd ?? (ind.macd ?? null),
      macdSignal: (s as any).macdSignal ?? (ind.macdSignal ?? null),
      macdHist: (s as any).macdHist ?? (ind.macdHist ?? null),
      bb_basis: (s as any).bb_basis ?? (ind.bb_basis ?? null),
      bb_upper: (s as any).bb_upper ?? (ind.bb_upper ?? null),
      bb_lower: (s as any).bb_lower ?? (ind.bb_lower ?? null),
      bb_width: (s as any).bb_width ?? (ind.bb_width ?? null),
      don_width: (s as any).don_width ?? (ind.don_width ?? null),
      atr: (s as any).atr ?? (ind.atr ?? null),
      stoch_k: (s as any).stoch_k ?? (ind.stoch_k ?? null),
      stoch_d: (s as any).stoch_d ?? (ind.stoch_d ?? null),
      ichi_tenkan: (s as any).ichi_tenkan ?? (ind.ichi_tenkan ?? null),
      ichi_kijun: (s as any).ichi_kijun ?? (ind.ichi_kijun ?? null),
      ichi_spanA: (s as any).ichi_spanA ?? (ind.ichi_spanA ?? null),
      ichi_spanB: (s as any).ichi_spanB ?? (ind.ichi_spanB ?? null),
      ichi_chikou: (s as any).ichi_chikou ?? (ind.ichi_chikou ?? null),
      adx: (s as any).adx ?? (ind.adx ?? null),
      plusDi: (s as any).plusDi ?? (ind.plusDi ?? null),
      minusDi: (s as any).minusDi ?? (ind.minusDi ?? null),
      env_upper: (s as any).env_upper ?? (ind.env_upper ?? null),
      env_lower: (s as any).env_lower ?? (ind.env_lower ?? null),
      dev_pct: (s as any).dev_pct ?? (ind.dev_pct ?? null),
      psar: (s as any).psar ?? (ind.psar ?? null),
      fib_pos: (s as any).fib_pos ?? (ind.fib_pos ?? null),
    } as FeatureSample;
    // Emit one WARN line per process as a sample of indicators
    const onceKey = `__ind_warn_once_${key}`;
    const g: any = global as any;
    if (!g[onceKey]) {
      g[onceKey] = 1;
    try { logWarn(`[INDICATOR] sample rsi14=${ind.rsi?.toFixed?.(2)} macd_hist=${ind.macdHist?.toFixed?.(4)} atr14=${ind.atr?.toFixed?.(2)} bb_width=${ind.bb_width?.toFixed?.(2)}% roc=${ind.roc?.toFixed?.(2)}%`); } catch {}
    }
  } catch {}
  const jsonlPath = path.join(dir, `features-${date}.jsonl`);
  try { if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true }); } catch {}
  // JSONL append (1行1レコード)
  const line = JSON.stringify({ ...s, rsi14: s.rsi ?? null, atr14: s.atr ?? null, macd_hist: s.macdHist ?? null, spread, slippage }) + '\n';
  if (isTest) {
    try { fs.appendFileSync(jsonlPath, line, { encoding: 'utf8' }); } catch {}
  } else {
    pushBuffer(jsonlPath, line);
  }
  // latest json for convenience (best-effort)
  writeLatestJson(base, s);
  // background rotation (best-effort, non-blocking)
  rotateOldFeatureLogs(base).catch(()=>{});
}

// Flush on process end
try {
  process.on('beforeExit', async ()=>{ await flushFeatureLogBuffers(); });
  process.on('exit', ()=>{});
} catch {}
