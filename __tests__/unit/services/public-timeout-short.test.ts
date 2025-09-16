import { describe, it, expect, beforeEach, vi, beforeAll } from 'vitest';

vi.mock('../../../src/api/public', () => {
  return {
    getTicker: vi.fn().mockRejectedValue(Object.assign(new Error('ETIMEDOUT'), { code: 'ETIMEDOUT' })),
    getOrderBook: vi.fn().mockResolvedValue({ bids: [], asks: [] }),
    getTrades: vi.fn().mockResolvedValue([]),
  };
});

describe('public API timeout short (RETRY_ATTEMPTS=1)', () => {
  beforeEach(() => {
    process.env.RETRY_ATTEMPTS = '1';
  });
  let mod: typeof import('../../../src/adapters/market-service');

  beforeAll(async () => {
  mod = await import('../../../src/adapters/market-service');
  });

  it('getTicker ETIMEDOUT causes warn path and returns partial overview', async () => {
    const res = await mod.fetchMarketOverview('btc_jpy');
    expect(res.orderBook).toBeDefined();
    expect(Array.isArray(res.trades)).toBe(true);
    // ticker may be undefined due to failure; ensure no throw and structure exists
    // (exact warn log is covered in other abnormal tests)
  });
});