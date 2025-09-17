export type Priority = 'high' | 'normal' | 'low';
export type Category = 'API-PUBLIC' | 'API-PRIVATE' | 'EXEC' | 'API' | 'MISC' | string;
export type OpType = 'ORDER' | 'CANCEL' | 'QUERY' | 'POLL' | string | undefined;

export interface RateLimiterOptions {
  capacity?: number;
  refillPerSec?: number;
  reserveRatio?: number; // 0..1 reserved for high priority
}

export class RateLimiter {
  private capacity: number; // legacy shared capacity (fallback)
  private refillPerSec: number; // legacy shared refill (fallback)
  private reserveRatio: number;
  // per-category buckets
  private buckets: Record<'PUBLIC'|'PRIVATE'|'EXEC'|'MISC', { capacity: number; refillPerSec: number; tokens: number; lastRefill: number }>;
  private metricsIntervalMs: number;
  private samples: Array<{ ts: number; waitedMs: number; cat: 'PUBLIC'|'PRIVATE'|'EXEC'|'MISC'; outcome: 'acquired'|'rejected' }> = [];
  private lastMetricsEmit: number;

  constructor(opts: RateLimiterOptions = {}) {
    // env fallbacks: support both new and legacy names
    const envCapacity = process.env.RATE_CAPACITY ?? process.env.RATE_LIMIT_CAPACITY;
    const envRefill = process.env.RATE_REFILL ?? process.env.RATE_REFILL_PER_SEC;
    const envReserve = process.env.RATE_RESERVE_RATIO ?? process.env.RATE_PRIORITY_RESERVE;
    this.capacity = Math.max(1, opts.capacity ?? Number(envCapacity ?? 100));
    this.refillPerSec = Math.max(0, opts.refillPerSec ?? Number(envRefill ?? 10));
    const rr = opts.reserveRatio ?? Number(envReserve ?? 0.1);
    this.reserveRatio = rr >= 0 && rr < 1 ? rr : 0.1;
    const now = Date.now();
    const capPub = Number(process.env.RATE_CAPACITY_PUBLIC ?? this.capacity);
    const capPri = Number(process.env.RATE_CAPACITY_PRIVATE ?? this.capacity);
    const capExe = Number(process.env.RATE_CAPACITY_EXEC ?? this.capacity);
    const refPub = Number(process.env.RATE_REFILL_PUBLIC ?? this.refillPerSec);
    const refPri = Number(process.env.RATE_REFILL_PRIVATE ?? this.refillPerSec);
    const refExe = Number(process.env.RATE_REFILL_EXEC ?? this.refillPerSec);
    const norm = (n: number, def: number) => Number.isFinite(n) && n >= 0 ? n : def;
    this.buckets = {
      PUBLIC:  { capacity: Math.max(1, norm(capPub, this.capacity)),  refillPerSec: Math.max(0, norm(refPub, this.refillPerSec)), tokens: 0, lastRefill: now },
      PRIVATE: { capacity: Math.max(1, norm(capPri, this.capacity)),  refillPerSec: Math.max(0, norm(refPri, this.refillPerSec)), tokens: 0, lastRefill: now },
      EXEC:    { capacity: Math.max(1, norm(capExe, this.capacity)),  refillPerSec: Math.max(0, norm(refExe, this.refillPerSec)), tokens: 0, lastRefill: now },
      MISC:    { capacity: this.capacity, refillPerSec: this.refillPerSec, tokens: 0, lastRefill: now }
    };
    // start fully filled
    for (const k of Object.keys(this.buckets) as Array<keyof typeof this.buckets>) {
      this.buckets[k].tokens = this.buckets[k].capacity;
    }
    this.metricsIntervalMs = Math.max(0, Number(process.env.RATE_METRICS_INTERVAL_MS ?? 60000));
    this.lastMetricsEmit = Date.now();
  }

  private refill(cat: 'PUBLIC'|'PRIVATE'|'EXEC'|'MISC') {
    const b = this.buckets[cat];
    const now = Date.now();
    const elapsedMs = now - b.lastRefill;
    if (elapsedMs <= 0) return;
    const add = b.refillPerSec * (elapsedMs / 1000);
    if (add > 0) {
      b.tokens = Math.min(b.capacity, b.tokens + add);
      b.lastRefill = now;
    }
  }

  private reserved(capacity: number): number { return Math.floor(capacity * this.reserveRatio); }

  async acquire(priority: Priority = 'normal', maxWaitMs = 1000, _category?: Category, opType?: OpType): Promise<number> {
    const start = Date.now();
    const minSleep = 10;
    const mapCat = (c?: Category): 'PUBLIC'|'PRIVATE'|'EXEC'|'MISC' => {
      if (c === 'API-PUBLIC') return 'PUBLIC';
      if (c === 'API-PRIVATE') return 'PRIVATE';
      if (c === 'EXEC') return 'EXEC';
      return 'MISC';
    };
    const cat = mapCat(_category);
    while (true) {
      this.refill(cat);
      const b = this.buckets[cat];
      const resv = this.reserved(b.capacity);
      const t = b.tokens;
      const availableForNormal = Math.max(0, t - Math.min(resv, t));
      const canBorrowReserved = priority === 'high' && opType === 'ORDER';
      const canUse = canBorrowReserved ? (t >= 1) : (availableForNormal >= 1);
      if (canUse) {
        b.tokens = t - 1;
        const waited = Date.now() - start;
        this.recordSample('acquired', waited, _category);
        return waited;
      }
      const now = Date.now();
      if (now - start >= maxWaitMs) {
        this.recordSample('rejected', now - start, _category);
        throw Object.assign(new Error('rate_limited'), { code: 'RATE_LIMITED' });
      }
      // compute time needed for 1 token for this priority
      const needed = canBorrowReserved ? Math.max(0, 1 - t) : Math.max(0, 1 - availableForNormal);
      const waitMs = Math.max(minSleep, Math.ceil((needed / Math.max(1e-6, b.refillPerSec)) * 1000));
      const remaining = maxWaitMs - (now - start);
      await new Promise(r => setTimeout(r, Math.min(waitMs, Math.max(minSleep, remaining))));
    }
  }

  private recordSample(outcome: 'acquired'|'rejected', waitedMs: number, cat?: Category) {
    const mapCat = (c?: Category): 'PUBLIC'|'PRIVATE'|'EXEC'|'MISC' => {
      if (c === 'API-PUBLIC') return 'PUBLIC';
      if (c === 'API-PRIVATE') return 'PRIVATE';
      if (c === 'EXEC') return 'EXEC';
      return 'MISC';
    };
    this.samples.push({ ts: Date.now(), waitedMs, cat: mapCat(cat), outcome });
    if (this.samples.length > 50) this.samples.splice(0, this.samples.length - 50);
    this.maybeEmitMetrics();
  }

  private maybeEmitMetrics() {
    const interval = this.metricsIntervalMs;
    if (interval <= 0) return;
    const now = Date.now();
    if (now - this.lastMetricsEmit < interval) return;
    this.lastMetricsEmit = now;
    const window = this.samples.length;
    if (window === 0) return;
    const acquired = this.samples.filter(s => s.outcome === 'acquired');
    const rejected = this.samples.filter(s => s.outcome === 'rejected');
    const avgWaitMs = acquired.length ? Math.round(acquired.reduce((a, s) => a + s.waitedMs, 0) / acquired.length) : 0;
    const cats: Array<'PUBLIC'|'PRIVATE'|'EXEC'> = ['PUBLIC','PRIVATE','EXEC'];
    const byCategory = { PUBLIC: 0, PRIVATE: 0, EXEC: 0 } as Record<'PUBLIC'|'PRIVATE'|'EXEC', number>;
    const details: Record<string, { count: number; acquired: number; rejected: number; avgWaitMs: number; rejectRate: number; capacity: number; refillPerSec: number }>= {};
    for (const c of cats) {
      const list = this.samples.filter(s => s.cat === c);
      const acq = list.filter(s => s.outcome === 'acquired');
      const rej = list.filter(s => s.outcome === 'rejected');
      const avg = acq.length ? Math.round(acq.reduce((a, s) => a + s.waitedMs, 0) / acq.length) : 0;
      byCategory[c] = list.length;
      const b = this.buckets[c];
      details[c] = { count: list.length, acquired: acq.length, rejected: rej.length, avgWaitMs: avg, rejectRate: Number((rej.length / Math.max(1, list.length)).toFixed(3)), capacity: Math.round(b.capacity), refillPerSec: b.refillPerSec };
    }
    const payload = {
      window,
      avgWaitMs,
      rejectRate: Number((rejected.length / window).toFixed(3)),
      byCategory,
      details
    };
    // lazy import to avoid cycles at module load
    try {
      const { log } = require('../utils/logger');
      if (typeof log === 'function') log('INFO', 'RATE', 'metrics', payload);
      else console.log('[INFO][RATE] metrics', payload);
    } catch {
      console.log('[INFO][RATE] metrics', payload);
    }
  }
}

let globalLimiter: RateLimiter | null = null;
export function getRateLimiter(): RateLimiter {
  if (!globalLimiter) globalLimiter = new RateLimiter();
  return globalLimiter;
}
export function setRateLimiter(l: RateLimiter | null) { globalLimiter = l; }
