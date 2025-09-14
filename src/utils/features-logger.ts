import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { todayStr } from './toolkit';
import zlib from 'zlib';

export interface FeatureSample {
  ts: number;
  pair: string;
  side: 'bid' | 'ask';
  rsi?: number | null;
  sma_short?: number | null;
  sma_long?: number | null;
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

// --- Buffered async logger with rotation and JSONL option ---

type BufferEntry = { path: string; lines: string[] };
const buffers = new Map<string, BufferEntry>();
let flushTimer: NodeJS.Timeout | null = null;
let intervalTimer: NodeJS.Timeout | null = null;
// Treat explicit TEST_MODE=1 as test; otherwise allow interval when TEST_MODE is '0' even under vitest
const __IS_TEST__ = process.env.TEST_MODE === '1';
const FLUSH_INTERVAL_MS = Math.max(0, Number(process.env.FEATURES_FLUSH_INTERVAL_MS || '1000'));

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
  const format = String(process.env.FEATURES_LOG_FORMAT || 'csv').toLowerCase(); // csv|jsonl|both
  const isTest = process.env.TEST_MODE === '1' || !!process.env.VITEST_WORKER_ID;
  // derive spread/slippage if not provided and bestBid/Ask present
  let spread = s.spread;
  let slippage = s.slippage;
  if ((spread == null || slippage == null) && s.bestAsk && s.bestBid){
    const mid = (s.bestAsk + s.bestBid) / 2;
    if (spread == null) spread = (s.bestAsk - s.bestBid) / mid;
    if (slippage == null) slippage = (s.price - mid) / mid;
  }
  const header = 'ts,pair,side,rsi,sma_short,sma_long,price,qty,pnl,win,bal_jpy,bal_btc,bal_eth,bal_xrp,spread,slippage,status,depth_bid,depth_ask,volume_recent';
  const csvRow = [
    s.ts,
    s.pair,
    s.side,
    s.rsi ?? '',
    s.sma_short ?? '',
    s.sma_long ?? '',
    s.price,
    s.qty,
    s.pnl ?? '',
    typeof s.win === 'boolean' ? (s.win ? 1 : 0) : '',
    s.balance?.jpy ?? '',
    s.balance?.btc ?? '',
    s.balance?.eth ?? '',
    s.balance?.xrp ?? '',
    spread ?? '',
    slippage ?? '',
    s.status ?? '',
    s.depthBid ?? '',
    s.depthAsk ?? '',
    s.volumeRecent ?? ''
  ].join(',');
  const csvPath = path.join(dir, `features-${date}.csv`);
  const jsonlPath = path.join(dir, `features-${date}.jsonl`);
  try { if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true }); } catch {}
  // Ensure CSV header exists when CSV is enabled
  if (format === 'csv' || format === 'both'){
    if (isTest) {
      try {
        if (!fs.existsSync(csvPath)) {
          fs.writeFileSync(csvPath, header + '\n' + csvRow + '\n', { encoding: 'utf8' });
        } else {
          fs.appendFileSync(csvPath, csvRow + '\n', { encoding: 'utf8' });
        }
      } catch {}
    } else {
      try { if (!fs.existsSync(csvPath)) fs.writeFileSync(csvPath, header + '\n'); } catch {}
      pushBuffer(csvPath, csvRow + '\n');
    }
  }
  if (format === 'jsonl' || format === 'both'){
    const line = JSON.stringify({ ...s, spread, slippage }) + '\n';
    if (isTest) {
      try { fs.appendFileSync(jsonlPath, line, { encoding: 'utf8' }); } catch {}
    } else {
      pushBuffer(jsonlPath, line);
    }
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
