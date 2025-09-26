import { describe, expect, test, vi } from 'vitest';
import * as mod from '../../../ztb/api/public/zaif';

vi.mock('axios', () => ({
  default: {
    get: vi.fn((url: string) => {
      if (url.includes('/ticker/')) return Promise.resolve({ data: { last: 234 } });
      if (url.includes('/depth/')) return Promise.resolve({ data: { asks: [[100,1]], bids: [[99,2]] } });
      if (url.includes('/trades/')) return Promise.resolve({ data: [{ tid: 1, price: 100, amount: 0.1, date: 1577836800, trade_type: 'bid' }] });
      return Promise.resolve({ data: {} });
    })
  }
}));

describe('public/zaif', () => {
  test('getTicker', async () => {
    const d = await mod.getTicker('btc_jpy');
    expect(d.last).toBe(234);
  });
  test('getOrderBook', async () => {
    const d = await mod.getOrderBook('btc_jpy');
    expect(d.asks[0][0]).toBe(100);
    expect(d.bids[0][0]).toBe(99);
  });
  test('getTrades', async () => {
    const arr = await mod.getTrades('btc_jpy');
    expect(arr[0].tid).toBe(1);
  });
});
