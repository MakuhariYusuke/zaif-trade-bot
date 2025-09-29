import fs from 'node:fs';

type TimerInfo = { id: NodeJS.Timeout; kind: 'timeout' | 'interval'; createdAt: number; stack?: string };
type StreamInfo = { path?: string; flags?: string; closed: boolean; createdAt: number; stack?: string; stream: fs.WriteStream };

const enabled = process.env.DETECT_LEAKS === '1';

let orig = {
  setTimeout: global.setTimeout,
  clearTimeout: global.clearTimeout,
  setInterval: global.setInterval,
  clearInterval: global.clearInterval,
  createWriteStream: fs.createWriteStream,
};

const timers = new Set<TimerInfo>();
const streams = new Set<StreamInfo>();

export function installLeakDetector() {
  if (!enabled) return;

  global.setTimeout = ((fn: any, ms?: number, ...args: any[]) => {
    const stack = new Error().stack;
    let handle: NodeJS.Timeout;
    const wrapped = (...a: any[]) => { try { fn(...a); } finally { timers.forEach(t => t.id === handle && timers.delete(t)); } };
    handle = orig.setTimeout(wrapped as any, ms as any, ...args) as NodeJS.Timeout;
    timers.add({ id: handle, kind: 'timeout', createdAt: Date.now(), stack });
    return handle;
  }) as any;

  global.clearTimeout = ((h: any) => {
    timers.forEach(t => t.id === h && timers.delete(t));
    return orig.clearTimeout(h as any);
  }) as any;

  global.setInterval = ((fn: any, ms?: number, ...args: any[]) => {
    const stack = new Error().stack;
    const handle = orig.setInterval(fn as any, ms as any, ...args) as NodeJS.Timeout;
    timers.add({ id: handle, kind: 'interval', createdAt: Date.now(), stack });
    return handle;
  }) as any;

  global.clearInterval = ((h: any) => {
    timers.forEach(t => t.id === h && timers.delete(t));
    return orig.clearInterval(h as any);
  }) as any;

  fs.createWriteStream = ((path: any, options?: any) => {
    const stack = new Error().stack;
    const ws = orig.createWriteStream(path as any, options);
    const info: StreamInfo = { path: typeof path === 'string' ? path : undefined, flags: (options as any)?.flags, closed: false, createdAt: Date.now(), stack, stream: ws };
    streams.add(info);
    const markClosed = () => { info.closed = true; streams.delete(info); };
    ws.once('close', markClosed);
    const origEnd = ws.end.bind(ws);
    ws.end = ((...args: any[]) => { try { return origEnd(...args); } finally { /* closed via 'close' event */ } }) as any;
    return ws;
  }) as any;
}

export function uninstallLeakDetector() {
  if (!enabled) return;
  global.setTimeout = orig.setTimeout;
  global.clearTimeout = orig.clearTimeout;
  global.setInterval = orig.setInterval;
  global.clearInterval = orig.clearInterval;
  fs.createWriteStream = orig.createWriteStream;
  timers.clear();
  streams.clear();
}

export function summarizeLeaks() {
  if (!enabled) return { timers: [], streams: [] as StreamInfo[] };
  const t = Array.from(timers);
  const s = Array.from(streams).filter(x => !x.closed);
  return { timers: t, streams: s };
}

/** afterEach用：残存があればWARN→strictならthrow／緩和ならcleanup */
export function assertNoLeaksAndCleanup() {
  if (!enabled) return;
  const { timers: t, streams: s } = summarizeLeaks();
  if (t.length === 0 && s.length === 0) return;

  const header = `[LEAK] timers=${t.length} streams=${s.length}`;
  // eslint-disable-next-line no-console
  console.warn(header);
  if (t.length) {
    for (const it of t) {
      // eslint-disable-next-line no-console
      console.warn(` - timer ${it.kind} createdAt=${new Date(it.createdAt).toISOString()}`);
      try { it.kind === 'interval' ? global.clearInterval(it.id) : global.clearTimeout(it.id); } catch {}
    }
  }
  if (s.length) {
    for (const st of s) {
      // eslint-disable-next-line no-console
      console.warn(` - stream path=${st.path ?? '(unknown)'} createdAt=${new Date(st.createdAt).toISOString()}`);
      try { st.stream.end(); } catch {}
    }
  }

  const strict = process.env.DETECT_LEAKS_STRICT !== '0';
  if (strict) {
    throw new Error(`${header} (DETECT_LEAKS_STRICT)`);
  }
}
