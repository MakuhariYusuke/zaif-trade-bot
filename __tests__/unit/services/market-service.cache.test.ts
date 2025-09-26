import { describe, it, expect, vi, beforeEach } from 'vitest';

const pub = {
  getTicker: vi.fn(async () => ({ last: Math.random() })),
  getOrderBook: vi.fn(async () => ({ bids: [[99, 1]], asks: [[101, 1]] })),
  getTrades: vi.fn(async () => ([{ price: 100, amount: 0.1 }]))
};

vi.mock('../../../ztb/api/public', () => pub);

describe('services/market-service cache', () => {
  beforeEach(()=>{ vi.resetModules(); vi.clearAllMocks(); process.env.MARKET_CACHE_TTL_MS = '1500'; delete process.env.TEST_MODE; delete process.env.VITEST_WORKER_ID; });

  it('uses cached ticker within TTL', async () => {
  const mod = await import('../../../ztb/adapters/market-service');
    const r1 = await mod.fetchTicker('btc_jpy');
    const r2 = await mod.fetchTicker('btc_jpy');
    expect(pub.getTicker).toHaveBeenCalledTimes(1);
    expect(r1.last).toBeDefined();
    expect(r2.last).toBeDefined();
  });
});
