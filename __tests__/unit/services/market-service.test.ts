import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../../../src/api/public', () => ({
  getTicker: vi.fn(async () => ({ last: 100 })),
  getOrderBook: vi.fn(async () => ({ bids: [[99,1]], asks: [[101,1]] })),
  getTrades: vi.fn(async () => ([{ price: 100, amount: 0.1 }])),
}));

describe('services/market-service', () => {
  beforeEach(()=>{ vi.resetModules(); });

  it('fetchMarketOverview returns aggregated data', async () => {
    const mod = await import('../../../src/services/market-service');
    const res = await mod.fetchMarketOverview('btc_jpy');
    expect(res.ticker.last).toBe(100);
    expect(res.orderBook.asks[0][0]).toBe(101);
    expect(res.trades.length).toBeGreaterThan(0);
  });

  it('fetchBalance throws on error', async () => {
    const mod = await import('../../../src/services/market-service');
    const priv: any = { get_info2: async () => ({ success: 0, error: 'bad' }) };
    mod.init(priv);
    await expect(mod.fetchBalance()).rejects.toThrow();
  });
});
