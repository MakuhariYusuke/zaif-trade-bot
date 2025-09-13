import { describe, it, expect, vi, beforeEach } from 'vitest';

// Simulate abnormal public API responses: timeout/429/Unauthorized

describe('services/market-service abnormal responses', () => {
  beforeEach(()=>{ vi.resetModules(); });

  it('fetchMarketOverview tolerates partial failures (e.g., getTrades 429)', async () => {
    vi.mock('../../../src/api/public', () => ({
      getTicker: vi.fn(async () => ({ last: 101 })),
      getOrderBook: vi.fn(async () => ({ bids: [[100,1]], asks: [[102,1]] })),
      getTrades: vi.fn(async () => { const e:any = new Error('429 Too Many Requests'); e.code = 429; throw e; }),
    }));
    const mod = await import('../../../src/services/market-service');
    const res = await mod.fetchMarketOverview('btc_jpy');
    expect(res.ticker?.last).toBe(101);
    expect(res.orderBook.asks[0][0]).toBe(102);
    expect(Array.isArray(res.trades)).toBe(true);
    expect(res.trades.length).toBe(0);
  });

  it('fetchBalance throws Unauthorized from private api', async () => {
    const mod = await import('../../../src/services/market-service');
    const priv: any = { get_info2: async () => ({ success: 0, error: 'Unauthorized' }) };
    mod.init(priv);
    await expect(mod.fetchBalance()).rejects.toThrow('Unauthorized');
  });
});
