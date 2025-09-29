import { describe, expect, test, vi, beforeEach } from 'vitest';
import * as mod from '../../../ztb/api/public/coincheck';

vi.mock('axios', () => {
  return {
    default: {
      get: vi.fn((url: string) => {
        if (url.endsWith('/ticker')) return Promise.resolve({ data: { last: 123 } });
        if (url.endsWith('/order_books')) return Promise.resolve({ data: { asks: [["100","1"]], bids: [["99","2"]] } });
        if (url.includes('/trades')) return Promise.resolve({ data: [{ id: 1, rate: '100', amount: '0.1', created_at: '2020-01-01T00:00:00Z', order_type: 'buy' }] });
        return Promise.resolve({ data: {} });
      })
    }
  };
});

describe('public/coincheck', () => {
  test('getTicker', async () => {
    const d = await mod.getTicker('btc_jpy');
    expect(d.last).toBe(123);
  });
  test('getOrderBook maps to numbers', async () => {
    const d = await mod.getOrderBook('btc_jpy');
    expect(d.asks[0][0]).toBe(100);
    expect(d.asks[0][1]).toBe(1);
    expect(d.bids[0][0]).toBe(99);
    expect(d.bids[0][1]).toBe(2);
  });
  test('getTrades maps fields', async () => {
    const arr = await mod.getTrades('btc_jpy');
    expect(arr.length).toBe(1);
    expect(arr[0].tid).toBe(1);
    expect(arr[0].trade_type).toBe('bid');
  });
});
