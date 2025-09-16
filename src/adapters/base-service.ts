import { sleep } from "../utils/toolkit";
import { logger as defaultLogger } from "../utils/logger";
import type { Logger } from "../utils/logger";
import type { PrivateApi } from "../types/private";

function toPosInt(val: any, def: number) { const n = Number(val); return Number.isFinite(n) && n >= 0 ? Math.floor(n) : def; }

export class BaseService {
  protected privateApi?: PrivateApi;
  protected logger?: Logger;

  init(api: PrivateApi) { this.privateApi = api; }
  dispose() { this.privateApi = undefined; this.logger = undefined; }
  setLogger(logger: Logger) { this.logger = logger; }

  async withRetry<T>(fn: () => Promise<T>, label: string, attempts?: number, backoffMs?: number, contextMeta?: any): Promise<T> {
    const ATT = (() => { const n = toPosInt(process.env.RETRY_ATTEMPTS, 3); return n > 0 ? n : 3; })();
    const BO = (() => { const n = toPosInt(process.env.RETRY_BACKOFF_MS, 50); return n >= 0 ? n : 50; })();
    const max = toPosInt(attempts ?? ATT, ATT);
    const backoff = toPosInt(backoffMs ?? BO, BO);
    let lastErr: any;
    for (let i = 0; i < max; i++) {
      try { return await fn(); } catch (e: any) {
        lastErr = e;
        const code = e?.code || e?.cause?.code;
        const isConnReset = code === 'ECONNRESET' || /ECONNRESET/i.test(String(e?.message || ''));
        if (isConnReset) break;
        if (i < max - 1) {
          const amp = 0.1 + Math.random() * 0.1; // 0.10 - 0.20
          const sign = Math.random() < 0.5 ? -1 : 1;
          const factor = 1 + sign * amp; // 0.8-0.9 or 1.1-1.2
          const delay = Math.floor(backoff * Math.pow(2, i) * factor);
          await sleep(delay);
        }
      }
    }
    const wrapped = new Error(`${label} failed: ${lastErr?.message || String(lastErr)}`);
    (wrapped as any).cause = lastErr;
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

export async function withRetry<T>(fn: () => Promise<T>, label: string, attempts?: number, backoffMs?: number, contextMeta?: any): Promise<T> {
  const svc = new BaseService();
  return svc.withRetry(fn, label, attempts, backoffMs, contextMeta);
}
