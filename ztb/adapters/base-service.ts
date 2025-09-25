import { sleep } from "../utils/toolkit";
import { logger as defaultLogger } from "../utils/logger";
import type { Logger } from "../utils/logger";
import type { PrivateApi } from "../types/private";
import { getEventBus } from "../application/events/bus";
import { buildErrorEventMeta, normalizeErrorCode } from "../application/errors";

function toPosInt(val: number | string | undefined | null, def: number) {
  const n = Number(val);
  if (Number.isFinite(n) && n >= 0) {
    return Math.floor(n);
  }
  return def;
}

export class BaseService {
  protected privateApi?: PrivateApi;
  protected logger?: Logger;

  init(api: PrivateApi) { this.privateApi = api; }
  dispose() { this.privateApi = undefined; this.logger = undefined; }
  setLogger(logger: Logger) { this.logger = logger; }

  /**
   * Executes the given async function with retry logic, circuit breaker, and rate limiter.
   * @param fn - The async function to execute.
   * @param label - Label for logging and error messages.
   * @param attempts - Maximum number of retry attempts.
   * @param backoffMs - Initial backoff in milliseconds.
   * @param contextMeta - Optional metadata for logging, circuit breaker, and rate limiter.
   *   {
   *     requestId?: string;
   *     pair?: string;
   *     side?: string;
   *     amount?: number;
   *     price?: number;
   *     category?: string;
   *     priority?: 'low' | 'normal' | 'high';
   *   }
   */
  async withRetry<T>(
    fn: () => Promise<T>,
    label: string,
    attempts?: number,
    backoffMs?: number,
    contextMeta?: {
      requestId?: string;
      pair?: string;
      side?: string;
      amount?: number;
      price?: number;
      category?: string;
      priority?: 'low' | 'normal' | 'high';
      opType?: 'ORDER' | 'CANCEL' | 'QUERY' | 'POLL' | string;
      [key: string]: any;
    }
  ): Promise<T> {
    const ATT = toPosInt(process.env.RETRY_ATTEMPTS, 3);
    const BO = toPosInt(process.env.RETRY_BACKOFF_MS, 50);
    const max = toPosInt(attempts ?? ATT, ATT);
    const backoff = toPosInt(backoffMs ?? BO, BO);
    let lastErr: any;
    const baseMeta = {
      requestId: contextMeta?.requestId ?? null,
      pair: contextMeta?.pair ?? null,
      side: contextMeta?.side ?? null,
      amount: contextMeta?.amount ?? null,
      price: contextMeta?.price ?? null,
    };
    const category = contextMeta?.category || 'API';
    const cat = String(category);
    // Circuit breaker gating for PUBLIC/PRIVATE/EXEC
    let allowed = true;
    let CircuitBreaker: any = undefined;
    let cb: any = undefined;
    try {
      const imported = await import('../application/circuit-breaker.js');
      CircuitBreaker = imported.CircuitBreaker;
      const pick = (c: string) => {
        const g = global as unknown as { [key: string]: InstanceType<typeof CircuitBreaker> };
        const key = `__cb_${c}`;
        if (!g[key]) g[key] = new CircuitBreaker();
        return g[key];
      };
      cb = cat === 'API-PUBLIC' ? pick('public') : (cat === 'API-PRIVATE' ? pick('private') : (cat === 'EXEC' ? pick('exec') : pick('misc')));
      // Only gate for PRIVATE/EXEC; PUBLIC and unknown just record metrics
      if (cat === 'API-PRIVATE' || cat === 'EXEC') {
        allowed = cb.allowRequest();
      } else {
        allowed = true;
      }
    } catch (e) {
      this.clog('CB', 'WARN', 'CircuitBreaker import failed', { ...baseMeta, category: cat, error: String(e) });
      // allowed remains true, but log the failure for troubleshooting
    }
    if (!allowed) {
      this.clog('CB', 'ERROR', 'blocked', { ...baseMeta, category: cat });
      const err: any = new Error('circuit_open');
      err.code = 'CIRCUIT_OPEN';
      err.cause = { code: 'CIRCUIT_OPEN' };
      try { getEventBus().publish({ type: 'EVENT/ERROR', code: 'CIRCUIT_OPEN', ...buildErrorEventMeta(baseMeta, err) } as any); } catch {}
      throw err;
    }

    // Rate limiter acquire token
  const priority = contextMeta?.priority ?? 'normal';
  const opType = contextMeta?.opType;
  // LONG_TESTS=1 の長時間スループット計測では 1000 件(≒100秒) 等を完了させるため
  // デフォルト待機時間を大幅に引き上げる。通常実行では短い待ちで速やかに RATE_LIMITED を返す。
  const LONG_MODE = String(process.env.LONG_TESTS || '0') === '1';
  const RATE_MAX_WAIT_DEFAULT = toPosInt(process.env.RATE_MAX_WAIT_MS, LONG_MODE ? 200_000 : 1000);
  const rateMaxWaitMs = toPosInt((contextMeta as any)?.rateMaxWaitMs, RATE_MAX_WAIT_DEFAULT);
  const IS_TEST = (process.env.TEST_MODE === '1') || !!process.env.VITEST_WORKER_ID;
  const TEST_FORCE_RATE = (global as any).__rateLimiterSet === true; // tests that call setRateLimiter expect it enabled
  const RATE_ENABLED = TEST_FORCE_RATE ? true : (String(process.env.RATE_LIMITER_ENABLED ?? (IS_TEST ? '0' : '1')) === '1');
    // simple consecutive wait detection per category (non-persistent)
    const g: any = global as any;
    const waitKey = `__rate_wait_consec_${cat}`;
  if (RATE_ENABLED) {
    try {
      // Cache getRateLimiter import in global object
      if (!g.__getRateLimiter) {
        g.__getRateLimiterPromise = import('../application/rate-limiter.js').then(mod => mod.getRateLimiter);
      }
      const getRateLimiter: typeof import('../application/rate-limiter.js').getRateLimiter =
        g.__getRateLimiter || (g.__getRateLimiter = await g.__getRateLimiterPromise);
      const wait = await getRateLimiter().acquire(priority, rateMaxWaitMs, cat, opType);
      const rateCat = 'RATE';
      if (wait > 500) {
        g[waitKey] = (g[waitKey] ?? 0) + 1;
        this.clog(rateCat, 'WARN', 'waited', { ...baseMeta, category: cat, priority, waitedMs: wait, consec: g[waitKey] });
      } else {
        g[waitKey] = 0;
        this.clog(rateCat, 'INFO', 'acquired', { ...baseMeta, category: cat, priority, waitedMs: wait });
      }
    } catch (e: any) {
      if (e?.code === 'RATE_LIMITED') {
        this.clog('RATE', 'ERROR', 'rejected', { ...baseMeta, category: cat, priority });
        const err: any = new Error('rate_limited');
        err.code = 'RATE_LIMITED';
        err.cause = { code: 'RATE_LIMITED' };
        try { getEventBus().publish({ type: 'EVENT/ERROR', code: 'RATE_LIMITED', ...buildErrorEventMeta(baseMeta, err) } as any); } catch {}
        throw err;
      }
    }
  }
    for (let i = 0; i < max; i++) {
      const start = Date.now();
      try {
        const res = await fn();
        try { if (cb && typeof cb.recordSuccess === 'function') cb.recordSuccess(Date.now() - start); } catch {}
        return res;
      } catch (e: any) {
        lastErr = e;
        try { if (cb && typeof cb.recordFailure === 'function') cb.recordFailure(e); } catch {}
        const code = e?.code || e?.cause?.code;
        const connReset = code === 'ECONNRESET' || /ECONNRESET/i.test(String(e?.message || ''));
        if (connReset) break;
        if (i < max - 1) {
          const amp = 0.1 + Math.random() * 0.1; // 0.10 - 0.20
          const sign = Math.random() < 0.5 ? -1 : 1;
          const factor = 1 + sign * amp; // 0.8-0.9 or 1.1-1.2
          const delay = Math.floor(backoff * Math.pow(2, i) * factor);
          // category WARN with required meta and retry count
          this.clog(category, 'WARN', 'retry', {
            ...baseMeta,
            retries: i + 1,
            cause: { code: code ?? null, message: e?.message }
          });
          await sleep(delay);
        }
      }
    }
    const wrapped = new Error(`${label} failed: ${lastErr?.message || String(lastErr)}`);
    (wrapped as any).cause = lastErr;
    // category ERROR on final failure with required meta
    this.clog(category, 'ERROR', 'failed', {
      ...baseMeta,
      retries: max,
      cause: { code: (lastErr?.code || lastErr?.cause?.code) ?? null, message: lastErr?.message }
    });
    try {
      const code = normalizeErrorCode(lastErr);
      getEventBus().publish({ type: 'EVENT/ERROR', code, ...buildErrorEventMeta(baseMeta, lastErr) } as any);
    } catch {}
    throw wrapped;
  }

  log(level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR', message: string, meta?: any) {
    const lg = this.logger ?? defaultLogger;
    switch (level) {
      case 'DEBUG': return lg.debug(message, meta);
      case 'INFO': return lg.info(message, meta);
      case 'WARN': return lg.warn(message, meta);
      case 'ERROR': return lg.error(message, meta);
    }
  }
  clog(category: string, level: 'DEBUG'|'INFO'|'WARN'|'ERROR', message: string, meta?: any) {
    const lg = this.logger ?? defaultLogger;
    if ((lg as any).log) return (lg as any).log(level, category, message, meta);
    return this.log(level, message, meta);
  }
  debug(message: string, meta?: any) { this.log('DEBUG', message, meta); }
  info(message: string, meta?: any) { this.log('INFO', message, meta); }
  warn(message: string, meta?: any) { this.log('WARN', message, meta); }
  error(message: string, meta?: any) { this.log('ERROR', message, meta); }
}

export default BaseService;

/**
 * Standalone helper function for retrying async operations.
 * This is a convenience wrapper around BaseService.withRetry for use outside of class context.
 * Prefer using BaseService.withRetry when you need logger or dependency injection.
 */
export async function withRetry<T>(fn: () => Promise<T>, label: string, attempts?: number, backoffMs?: number, contextMeta?: any): Promise<T> {
  const svc = new BaseService();
  return svc.withRetry(fn, label, attempts, backoffMs, contextMeta);
}
