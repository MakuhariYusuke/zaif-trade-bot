import * as fs from 'fs';
import * as path from 'path';

export type RateDetails = {
  count: number;
  acquired: number;
  rejected: number;
  avgWaitMs: number;
  rejectRate: number;
  capacity: number;
  refillPerSec: number;
};

export type RatePayload = {
  window: number;
  avgWaitMs: number;
  rejectRate: number;
  byCategory: { PUBLIC: number; PRIVATE: number; EXEC: number };
  details: { [k: string]: RateDetails };
};

export type CacheCounter = { hits: number; misses: number; stale: number; hitRate: number };

export type LogEntry = {
  ts: string;
  level: string;
  category?: string;
  message: string;
  data?: any[];
};
export type EventMetrics = {
  windowMs: number;
  types: Record<string, {
    publishes: number;
    handlerCalls: number;
    errors: number;
    avgLatencyMs: number;
    p95LatencyMs: number;
    handlers: Record<string, { name: string; calls: number; errors: number; avgLatencyMs: number; p95LatencyMs: number }>;
  }>;
};

function parseArgs(argv: string[]) {
  const out: any = { file: undefined as string | undefined, lines: 4000 as number, watch: false as boolean, watchMs: 2000 as number, json: false as boolean };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if ((a === '--file' || a === '-f') && argv[i + 1]) { out.file = argv[++i]; continue; }
    if (a === '--lines' && argv[i + 1]) { out.lines = Math.max(100, Number(argv[++i]) || 4000); continue; }
    if (a === '--watch') {
      out.watch = true;
      if (argv[i + 1] && !argv[i + 1].startsWith('--')) { out.watchMs = Math.max(200, Number(argv[++i]) || 2000); }
      continue;
    }
    if (a === '--json') { out.json = true; continue; }
    if (a === '--help' || a === '-h') { out.help = true; }
  }
  return out;
}

function printHelp() {
  console.log('Usage: ts-node src/tools/metrics-dash.ts [--file <log>] [--lines N] [--watch [intervalMs]] [--json]');
  console.log('Notes: Prefer LOG_JSON=1 logs. Also attempts plain-text parsing as fallback.');
}

function findLatestLogFile(): string | undefined {
  const candidates: string[] = [];
  const root = process.cwd();
  const pushLogs = (dir: string) => {
    try {
      const full = path.resolve(root, dir);
      const items = fs.readdirSync(full);
      for (const it of items) {
        if (it.endsWith('.log')) candidates.push(path.join(full, it));
      }
    } catch { /* ignore */ }
  };
  pushLogs('logs');
  // scan known tmp folders
  try {
    const tmpDirs = fs.readdirSync(root).filter(d => d.startsWith('tmp-'));
    for (const d of tmpDirs) pushLogs(path.join(d, 'logs'));
  } catch { /* ignore */ }
  if (candidates.length === 0) return undefined;
  let best: string | undefined;
  let bestMtime = 0;
  for (const f of candidates) {
    try {
      const st = fs.statSync(f);
      const m = st.mtimeMs;
      if (m > bestMtime) { bestMtime = m; best = f; }
    } catch { /* ignore */ }
  }
  return best;
}

export function tailLines(content: string, maxLines: number): string[] {
  const lines = content.split(/\r?\n/);
  if (lines.length <= maxLines) return lines.filter(Boolean);
  return lines.slice(lines.length - maxLines).filter(Boolean);
}

export function tryParseJsonFromLine(line: string): LogEntry | null {
  // JSONL path
  const idx = line.indexOf('{');
  if (idx < 0) return null;
  const maybe = line.slice(idx);
  try {
    const obj = JSON.parse(maybe);
    if (obj && typeof obj === 'object' && obj.ts && obj.message) return obj as LogEntry;
  } catch { /* not JSON */ }
  return null;
}
// Fallback: parse plain-text lines like "[INFO][RATE] metrics { ... }"
export function tryParsePlainMetricsLine(line: string): { category: 'RATE'|'CACHE'|'EVENT'; payload: any } | null {
  const m = line.match(/\[(INFO|WARN|ERROR)\]\[(RATE|CACHE|EVENT)\]\s+metrics\s+(\{.*\})/);
  if (!m) return null;
  const cat = m[2] as 'RATE'|'CACHE'|'EVENT';
  const objStr = m[3];
  // Heuristic transform to JSON
  let s = objStr;
  s = s.replace(/([\{,\s])(\w+)\s*:/g, '$1"$2":'); // quote keys
  s = s.replace(/'/g, '"');
  try {
    const parsed = JSON.parse(s);
    return { category: cat, payload: parsed };
  } catch { return null; }
}

export function formatPercent(n: number) { return (Number(n) * 100).toFixed(1) + '%'; }

export function buildSparkline(values: number[], width = 30): string {
  if (!values.length) return '';
  const glyphs = ['▁','▂','▃','▄','▅','▆','▇','█'];
  const arr = values.slice(-width);
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const span = Math.max(1e-9, max - min);
  return arr.map(v => {
    const idx = Math.min(glyphs.length - 1, Math.max(0, Math.floor(((v - min) / span) * (glyphs.length - 1))))
    return glyphs[idx];
  }).join('');
}

export function printRate(payload: RatePayload, history?: number[]) {
  const c = colorFns();
  const avgColor = payload.avgWaitMs > 2000 ? c.red : payload.avgWaitMs > 500 ? c.yellow : c.green;
  const rejColor = payload.rejectRate > 0.10 ? c.red : payload.rejectRate > 0.05 ? c.yellow : c.green;
  console.log('=== RATE/METRICS ===');
  let p95 = 0;
  if (history && history.length) {
    const arr = history.slice().sort((a,b)=>a-b);
    const idx = Math.floor(0.95 * (arr.length - 1));
    p95 = arr[Math.max(0, idx)] || 0;
  }
  console.log(`window=${payload.window} avgWait=${avgColor(String(payload.avgWaitMs)+'ms')} p95=${avgColor(String(Math.round(p95))+'ms')} rejectRate=${rejColor(formatPercent(payload.rejectRate))}`);
  if (history && history.length) console.log(`avgWait spark: ${buildSparkline(history)}`);
  const heads = ['Category','acq','rej','avgWait','rejRate','cap','refill/s'];
  console.log(heads.join('\t'));
  for (const k of ['PUBLIC','PRIVATE','EXEC']) {
    const d = payload.details[k];
    if (!d) continue;
    const avgC = d.avgWaitMs > 2000 ? c.red : d.avgWaitMs > 500 ? c.yellow : c.green;
    const rejC = d.rejectRate > 0.10 ? c.red : d.rejectRate > 0.05 ? c.yellow : c.green;
    console.log([k, d.acquired, d.rejected, avgC(`${d.avgWaitMs}ms`), rejC(formatPercent(d.rejectRate)), d.capacity, d.refillPerSec].join('\t'));
  }
  console.log('');
}

export function printCache(payload: Record<string, CacheCounter>, history?: number[]) {
  const c = colorFns();
  console.log('=== CACHE/METRICS ===');
  if (history && history.length) console.log(`hitRate spark: ${buildSparkline(history)}`);
  const heads = ['Cache','hits','misses','stale','hitRate'];
  console.log(heads.join('\t'));
  for (const [name, ctn] of Object.entries(payload)) {
    const hr = ctn.hitRate;
    const hrC = hr < 0.50 ? c.red : hr < 0.80 ? c.yellow : c.green;
    const total = ctn.hits + ctn.misses;
    const staleRatio = total ? (ctn.stale / total) : 0;
    const staleStr = staleRatio > 0.2 ? c.red(formatPercent(staleRatio)) : staleRatio > 0.1 ? c.yellow(formatPercent(staleRatio)) : c.green(formatPercent(staleRatio));
    console.log([name, ctn.hits, ctn.misses, `${ctn.stale} (${staleStr})`, hrC(formatPercent(hr))].join('\t'));
  }
  console.log('');
}

export function avgEventLatency(ev: EventMetrics): number {
  const vals = Object.values(ev.types);
  if (!vals.length) return 0;
  const sum = vals.reduce((a, t) => a + Number(t.avgLatencyMs || 0), 0);
  return sum / vals.length;
}

export function printEvent(ev: EventMetrics, history?: number[]) {
  const c = colorFns();
  console.log('=== EVENT/METRICS ===');
  if (history && history.length) console.log(`avgLatency spark: ${buildSparkline(history)}`);
  console.log(`window=${Math.round(ev.windowMs/1000)}s types=${Object.keys(ev.types).length}`);
  const heads = ['Type','pub','calls','errors','avgMs','p95Ms','topHandler'];
  console.log(heads.join('\t'));
  const entries = Object.entries(ev.types).sort((a,b)=>b[1].handlerCalls - a[1].handlerCalls).slice(0, 8);
  for (const [type, s] of entries) {
    const errC = s.errors > 0 ? c.yellow : c.green;
    // pick most active handler name
    let top = '-';
    let maxCalls = -1;
    for (const [id, hs] of Object.entries(s.handlers || {})) {
      if (hs.calls > maxCalls) { maxCalls = hs.calls; top = hs.name || id; }
    }
    console.log([type, s.publishes, s.handlerCalls, errC(String(s.errors)), s.avgLatencyMs, s.p95LatencyMs, top].join('\t'));
  }
  console.log('');
}

function colorFns(){
  const disable = process.env.NO_COLOR === '1' || process.env.FORCE_COLOR === '0';
  const wrap = (code: string) => (s: string) => disable ? s : `\u001b[${code}m${s}\u001b[0m`;
  return {
    red: wrap('31'),
    yellow: wrap('33'),
    green: wrap('32'),
    dim: wrap('2'),
  };
}

export function parseEntriesFromContent(content: string, maxLines: number) {
  const lines = tailLines(content, maxLines);
  const entries: LogEntry[] = [];
  let ratePlain: any = null;
  let cachePlain: any = null;
  let eventPlain: any = null;
  for (const ln of lines) {
    const e = tryParseJsonFromLine(ln);
    if (e) entries.push(e);
    else {
      const p = tryParsePlainMetricsLine(ln);
      if (p) {
        if (p.category === 'RATE') ratePlain = p.payload;
        else if (p.category === 'CACHE') cachePlain = p.payload;
        else eventPlain = p.payload;
      }
    }
  }
  return { entries, ratePlain, cachePlain, eventPlain };
}

export function pickLatestMetrics(entries: LogEntry[], fallbacks: { ratePlain?: any; cachePlain?: any; eventPlain?: any }) {
  const latestRate = [...entries].reverse().find(e => e.category === 'RATE' && e.message === 'metrics');
  const latestCache = [...entries].reverse().find(e => e.category === 'CACHE' && e.message === 'metrics');
  const latestEvent = [...entries].reverse().find(e => e.category === 'EVENT' && e.message === 'metrics');
  const rate = latestRate ? (latestRate.data && latestRate.data[0]) : fallbacks.ratePlain;
  const cache = latestCache ? (latestCache.data && latestCache.data[0]) : fallbacks.cachePlain;
  const event = latestEvent ? (latestEvent.data && latestEvent.data[0]) : fallbacks.eventPlain;
  return { rate, cache, event } as { rate: RatePayload | undefined; cache: Record<string, CacheCounter> | undefined; event: EventMetrics | undefined };
}

export function clearScreen() { console.clear(); }

export async function runDash(opts: { file?: string; lines?: number; watch?: boolean; watchMs?: number; abortSignal?: AbortSignal }) {
  const file = opts.file || process.env.METRICS_LOG || findLatestLogFile();
  const lines = opts.lines ?? 4000;
  if (!file) throw new Error('No log file found. Specify with --file <path>.');
  if (!fs.existsSync(file)) throw new Error(`File not found: ${file}`);
  const rateHistory: number[] = [];
  const cacheHistory: number[] = [];
  const eventHistory: number[] = [];
  let showRate = true; // up/down toggle
  let sparkWidth = 30; // left/right to change
  const onKey = (chunk: Buffer) => {
    const str = chunk.toString();
    if (str === 'q' || str === 'Q' || str === '\u0003') { /* Ctrl+C */ process.emit('SIGINT'); return; }
    if (str === '\u001b[A' || str === '\u001b[B') { // up/down
      showRate = str === '\u001b[A' ? true : false;
    } else if (str === '\u001b[C') { // right
      sparkWidth = Math.min(60, sparkWidth + 2);
    } else if (str === '\u001b[D') { // left
      sparkWidth = Math.max(10, sparkWidth - 2);
    }
  };
  const loop = async () => {
    const content = fs.readFileSync(file, 'utf8');
    const { entries, ratePlain, cachePlain, eventPlain } = parseEntriesFromContent(content, lines);
    const { rate, cache, event } = pickLatestMetrics(entries, { ratePlain, cachePlain, eventPlain });
    clearScreen();
    console.log(`Source: ${file}`);
    console.log(`Now: ${new Date().toLocaleString()}`);
    if (!rate && !cache && !event) {
      console.error('No RATE/CACHE/EVENT metrics found in file.');
    } else {
      const nonTty = !process.stdin.isTTY;
      if (nonTty) {
        if (rate) { rateHistory.push(Number(rate.avgWaitMs || 0)); printRate(rate, rateHistory.slice(-sparkWidth)); }
        if (cache) {
          const avgHit = (() => {
            const vals = Object.values(cache);
            if (!vals.length) return 0;
            return vals.reduce((a, c) => a + Number(c.hitRate || 0), 0) / vals.length;
          })();
          cacheHistory.push(avgHit);
          printCache(cache, cacheHistory.slice(-sparkWidth));
        }
        if (event) {
          const avg = avgEventLatency(event);
          eventHistory.push(avg);
          printEvent(event, eventHistory.slice(-sparkWidth));
        }
      } else if (showRate && rate) { rateHistory.push(Number(rate.avgWaitMs || 0)); console.log('(↑/↓ 切替, ←/→ 幅, q 終了)'); printRate(rate, rateHistory.map(x=>x).slice(-sparkWidth)); }
      else if (!showRate && cache) {
        const avgHit = (() => {
          const vals = Object.values(cache);
          if (!vals.length) return 0;
          return vals.reduce((a, c) => a + Number(c.hitRate || 0), 0) / vals.length;
        })();
        cacheHistory.push(avgHit);
        console.log('(↑/↓ 切替, ←/→ 幅, q 終了)');
        printCache(cache, cacheHistory.slice(-sparkWidth));
      } else if (!showRate && event) {
        const avg = avgEventLatency(event);
        eventHistory.push(avg);
        console.log('(↑/↓ 切替, ←/→ 幅, q 終了)');
        printEvent(event, eventHistory.slice(-sparkWidth));
      }
    }
  };
  if (opts.watch) {
    const interval = Math.max(200, opts.watchMs ?? 2000);
    let alive = true;
    const onSig = () => { alive = false; };
    process.on('SIGINT', onSig);
    try {
      if (process.stdin.isTTY) {
        process.stdin.setRawMode(true);
        process.stdin.resume();
        process.stdin.on('data', onKey);
      }
      while (alive) {
        if (opts.abortSignal?.aborted) break;
        await loop();
        await new Promise(r => setTimeout(r, interval));
      }
    } finally {
      process.off('SIGINT', onSig);
      if (process.stdin.isTTY) {
        try { process.stdin.setRawMode(false); } catch {}
        process.stdin.off('data', onKey);
      }
    }
  } else {
    await loop();
  }
}

function readTradePhase() {
  try {
    const statePath = process.env.TRADE_STATE_FILE || 'trade-state.json';
    if (!fs.existsSync(statePath)) return null;
    const s = JSON.parse(fs.readFileSync(statePath, 'utf8'));
    const phase = typeof s.phase === 'number' ? s.phase : null;
    const totalSuccess = typeof s.totalSuccess === 'number' ? s.totalSuccess : null;
    return { phase, totalSuccess };
  } catch { return null; }
}

export function runOnceCollect(file?: string, lines?: number) {
  const f = file || process.env.METRICS_LOG || findLatestLogFile();
  const maxLines = lines ?? 4000;
  if (!f || !fs.existsSync(f)) {
    return { error: 'log_not_found', file: f || null } as any;
  }
  const content = fs.readFileSync(f, 'utf8');
  const { entries, ratePlain, cachePlain, eventPlain } = parseEntriesFromContent(content, maxLines);
  const { rate, cache, event } = pickLatestMetrics(entries, { ratePlain, cachePlain, eventPlain });
  const tradePhase = readTradePhase();
  return { rate, cache, events: event, tradePhase };
}

function runOnceJson(file?: string, lines?: number) {
  const res = runOnceCollect(file, lines);
  console.log(JSON.stringify(res));
}

function runCli() {
  const args = parseArgs(process.argv);
  if (args.help) { printHelp(); return; }
  if (args.json) { runOnceJson(args.file, args.lines); return; }
  runDash({ file: args.file, lines: args.lines, watch: args.watch, watchMs: args.watchMs }).catch(err => {
    console.error(err?.message || String(err));
    process.exitCode = 1;
  });
}

if (require.main === module) runCli();
