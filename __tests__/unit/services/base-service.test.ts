import { describe, it, expect, vi, beforeEach } from 'vitest';
import BaseService from '../../../src/adapters/base-service';

describe('BaseService', () => {
  beforeEach(() => { vi.useRealTimers(); });

  it('withRetry retries and eventually succeeds', async () => {
    const svc = new BaseService();
    let attempts = 0;
    const fn = vi.fn(async () => {
      attempts++;
      if (attempts < 3) throw Object.assign(new Error('temp'), { code: 'EAGAIN' });
      return 'ok';
    });
    const t0 = Date.now();
    const res = await svc.withRetry(fn, 'test', 3, 1);
    const elapsed = Date.now() - t0;
    expect(res).toBe('ok');
    expect(fn).toHaveBeenCalledTimes(3);
    expect(elapsed).toBeGreaterThanOrEqual(1 + 2); // backoff 1 + 2 ms approx
  });

  it('withRetry bails early on ECONNRESET', async () => {
    const svc = new BaseService();
    const fn = vi.fn(async () => { throw Object.assign(new Error('boom ECONNRESET'), { code: 'ECONNRESET' }); });
    await expect(svc.withRetry(fn, 'label', 5, 1)).rejects.toThrow(/failed/);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('log helpers delegate to global logger', () => {
    const svc = new BaseService();
    const spyLog = vi.spyOn(console, 'log').mockImplementation(() => undefined);
    const spyWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const spyErr = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    try {
      process.env.TEST_MODE = '0';
      process.env.LOG_LEVEL = 'TRACE'; // allow all, and not suppressed in tests
      svc.debug('d');
      svc.info('i');
      svc.warn('w');
      svc.error('e');
      expect(spyLog).toHaveBeenCalled();
      expect(spyWarn).toHaveBeenCalled();
      expect(spyErr).toHaveBeenCalled();
    } finally {
      spyLog.mockRestore(); spyWarn.mockRestore(); spyErr.mockRestore();
    }
  });
});
