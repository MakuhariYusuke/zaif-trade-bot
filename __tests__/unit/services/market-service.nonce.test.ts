import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('services/market-service private nonce retry', () => {
    beforeEach(() => { vi.resetModules(); });

    it('retries on invalid nonce then succeeds', async () => {
    const mod = await import('../../../src/adapters/market-service');
        const trade = vi.fn()
            .mockRejectedValueOnce(Object.assign(new Error('invalid nonce'), { error: 'invalid nonce' }))
            .mockResolvedValueOnce({ return: { order_id: 'OK1' } });
        const priv: any = { trade };
        mod.init(priv);
        const r = await mod.placeLimitOrder('btc_jpy', 'BUY' as any, 100, 0.1);
        expect(r.order_id).toBe('OK1');
        expect(trade).toHaveBeenCalledTimes(2);
    });

    it('throws on Unauthorized error', async () => {
    const mod = await import('../../../src/adapters/market-service');
        const trade = vi.fn().mockRejectedValue(Object.assign(new Error('Unauthorized'), { error: 'Unauthorized' }));
        const priv: any = { trade };
        mod.init(priv);
        await expect(mod.placeLimitOrder('btc_jpy', 'BUY' as any, 100, 0.1)).rejects.toThrow();
    });
});
