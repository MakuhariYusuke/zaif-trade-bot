import crypto from 'crypto';

/** Common helpers and structure for Private API clients across exchanges. */
export abstract class BaseExchangePrivate {
  /** Optional display name for logging */
  protected readonly name: string = 'exchange';
  private nonceSeq = 0;

  protected now(): number { return Date.now(); }
  protected sleep(ms: number) { return new Promise(res => setTimeout(res, ms)); }

  // Nonce management: monotonic per-instance increasing number
  protected nextNonce(): string {
    const base = this.now() * 1000; // micro tick-ish granularity
    const n = base + (this.nonceSeq++ % 1000);
    return String(n);
  }

  // Signing / hashing helpers
  protected hmacSha256(data: string, key: string) { return crypto.createHmac('sha256', key).update(data).digest('hex'); }
  protected hmacSha512(data: string, key: string) { return crypto.createHmac('sha512', key).update(data).digest('hex'); }
  protected sha256(data: string) { return crypto.createHash('sha256').update(data).digest('hex'); }
  protected buildForm(params: Record<string, any>): string {
    return Object.entries(params).map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`).join('&');
  }

  protected maskBody(body: string) { return body.replace(/[0-9]/g, 'x'); }
  protected normalizePermMsg(msg: string) {
    const lower = msg?.toLowerCase?.() || '';
    if (/(permission|whitelist|ip address|ip whitelist|two-?factor)/i.test(lower)) {
      return msg + ' | Check API key rights and IP whitelist on the exchange.';
    }
    return msg;
  }

  protected getRetryConf() {
    return {
      maxRetries: Number(process.env.MAX_NONCE_RETRIES || 5),
      baseBackoff: Number(process.env.RETRY_BACKOFF_MS || 300),
      backoffFactor: Number(process.env.RETRY_BACKOFF_FACTOR || 1.5),
      maxBackoff: Number(process.env.RETRY_MAX_BACKOFF_MS || 3000),
      jitterMs: Number(process.env.RETRY_JITTER_MS || 100),
    };
  }

  protected computeBackoff(attempt: number) {
    const { baseBackoff, backoffFactor, maxBackoff, jitterMs } = this.getRetryConf();
    const exp = Math.min(maxBackoff, Math.floor(baseBackoff * Math.pow(backoffFactor, Math.max(0, attempt - 1))));
    return exp + Math.floor(Math.random() * Math.max(0, jitterMs));
  }

  // Generic retry wrapper for nonce-related transient errors
  protected async callWithRetry<T>(fn: () => Promise<T>, isNonceError: (e: any) => boolean): Promise<T> {
    const { maxRetries } = this.getRetryConf();
    let lastErr: any;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try { return await fn(); } catch (e: any) {
        lastErr = e;
        if (!isNonceError(e) || attempt === maxRetries) break;
        const wait = this.computeBackoff(attempt);
        await this.sleep(wait);
      }
    }
    throw lastErr;
  }
}
