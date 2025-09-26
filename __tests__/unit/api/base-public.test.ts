import { describe, expect, test, vi } from 'vitest';
import { BaseExchangePublic } from '../../../ztb/api/base-public';

class TestPub extends BaseExchangePublic {
  async getTicker(pair: string){ return { pair, t: 1 }; }
  async getOrderBook(pair: string){ return { pair, asks: [], bids: [] }; }
  async getTrades(pair: string){ return [{ pair, tid: 1 }]; }
}

describe('BaseExchangePublic', () => {
  test('methods return shapes', async () => {
    const c = new TestPub();
    const tk = await c.getTicker('btc_jpy');
    expect(tk.pair).toBe('btc_jpy');
    const ob = await c.getOrderBook('btc_jpy');
    expect(ob.asks).toBeDefined();
    const tr = await c.getTrades('btc_jpy');
    expect(Array.isArray(tr)).toBe(true);
  });
});
