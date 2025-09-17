type Counter = { hits: number; misses: number; stale: number };

const counters: Record<string, Counter> = {};
let lastEmit = Date.now();
let intervalMs = Math.max(0, Number(process.env.CACHE_METRICS_INTERVAL_MS ?? 60000));

function get(key: string): Counter {
  if (!counters[key]) counters[key] = { hits: 0, misses: 0, stale: 0 };
  return counters[key];
}

export function cacheHit(name: string) {
  get(name).hits++;
  maybeEmit();
}

export function cacheMiss(name: string) {
  get(name).misses++;
  maybeEmit();
}

export function cacheStale(name: string) {
  get(name).stale++;
  maybeEmit();
}

export function setCacheMetricsInterval(ms: number){ intervalMs = Math.max(0, ms); }

function maybeEmit(){
  if (intervalMs <= 0) return;
  const now = Date.now();
  if (now - lastEmit < intervalMs) return;
  lastEmit = now;
  const payload: Record<string, Counter & { hitRate: number }>= {};
  for (const [k,v] of Object.entries(counters)){
    const total = v.hits + v.misses;
    payload[k] = { ...v, hitRate: Number((total ? v.hits/total : 0).toFixed(3)) };
  }
  try {
    const { log } = require('./logger');
    if (typeof log === 'function') log('INFO','CACHE','metrics', payload);
    else console.log('[INFO][CACHE] metrics', payload);
  } catch {
    console.log('[INFO][CACHE] metrics', payload);
  }
}

// test helper
export function __resetCacheMetrics(){ for (const k of Object.keys(counters)) delete counters[k]; lastEmit = Date.now(); }
