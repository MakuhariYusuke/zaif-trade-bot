import { describe, it, expect, vi, beforeEach } from 'vitest';
import { setupJsonLogs, captureLogs, expectJsonLog } from '../../helpers/logging';

describe('adapters/market-service retry boundaries', () => {
  beforeEach(() => { vi.resetModules(); setupJsonLogs(); });

  it('retries on 429 then succeeds', async () => {
    const logs = captureLogs();
    const trade = vi.fn()
      .mockRejectedValueOnce(Object.assign(new Error('429 Too Many Requests'), { error: '429 Too Many Requests', code: 429 }))
      .mockResolvedValueOnce({ return: { order_id: 'OK1' } });
    const mod = await import('../../../src/adapters/market-service');
    mod.init({ trade } as any);
    const res = await mod.placeLimitOrder('btc_jpy', 'BUY' as any, 100, 0.1);
    expect(res.order_id).toBe('OK1');
    expect(trade).toHaveBeenCalledTimes(2);
    expectJsonLog(logs, 'API-PRIVATE', 'WARN', 'retry', ['retries','cause']);
  });

  it('continuous 429 exhausts backoff and fails', async () => {
    const logs = captureLogs();
    const trade = vi.fn().mockRejectedValue(Object.assign(new Error('429 Too Many Requests'), { error: '429 Too Many Requests', code: 429 }));
    const mod = await import('../../../src/adapters/market-service');
    mod.init({ trade } as any);
    await expect(mod.placeLimitOrder('btc_jpy', 'SELL' as any, 101, 0.2)).rejects.toThrow();
    expect(trade).toHaveBeenCalled();
    expectJsonLog(logs, 'API-PRIVATE', 'ERROR', 'failed', ['retries','cause']);
  });

  it('ECONNRESET treated as failure without retries', async () => {
    const logs = captureLogs();
    const trade = vi.fn().mockRejectedValue(Object.assign(new Error('boom ECONNRESET'), { error: 'ECONNRESET', code: 'ECONNRESET' }));
    const mod = await import('../../../src/adapters/market-service');
    mod.init({ trade } as any);
    await expect(mod.placeLimitOrder('btc_jpy', 'BUY' as any, 100, 0.1)).rejects.toThrow();
    expect(trade).toHaveBeenCalledTimes(1);
    expectJsonLog(logs, 'API-PRIVATE', 'ERROR', 'failed', ['retries','cause']);
  });

  it('JSON parse error surfaces as failure', async () => {
    const logs = captureLogs();
    const trade = vi.fn().mockRejectedValue(Object.assign(new Error('Unexpected token < in JSON at position 0'), { error: 'parse error' }));
    const mod = await import('../../../src/adapters/market-service');
    mod.init({ trade } as any);
    await expect(mod.placeLimitOrder('btc_jpy', 'SELL' as any, 100, 0.1)).rejects.toThrow();
    expectJsonLog(logs, 'API-PRIVATE', 'ERROR', 'failed', ['retries','cause']);
  });
});
