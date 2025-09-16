import { describe, it, vi, expect } from 'vitest';
import * as pub from '../../../src/api/public';
import { fetchMarketOverview } from '../../../src/adapters/market-service';

describe('public API abnormal cases', () => {
  it('timeout (ETIMEDOUT) is retried then warned/omitted', async () => {
    const obOk = { bids: [[100,1]], asks: [[101,1]] } as any;
    const trOk: any[] = [];
    const err = Object.assign(new Error('ETIMEDOUT'), { code: 'ETIMEDOUT' });
    const spyTicker = vi.spyOn(pub, 'getTicker').mockRejectedValueOnce(err).mockRejectedValueOnce(err).mockRejectedValueOnce(err);
    const spyOb = vi.spyOn(pub, 'getOrderBook').mockResolvedValue(obOk);
    const spyTr = vi.spyOn(pub, 'getTrades').mockResolvedValue(trOk);
    const r = await fetchMarketOverview('btc_jpy');
    expect(spyTicker).toHaveBeenCalled();
    expect(r.orderBook).toEqual(obOk);
    expect(r.trades).toEqual(trOk);
    expect(r.ticker).toBeUndefined();
  });
  it('network error (ECONNRESET) retried then omitted', async () => {
    const err = Object.assign(new Error('ECONNRESET'), { code: 'ECONNRESET' });
    vi.spyOn(pub, 'getTicker').mockResolvedValueOnce({ last: 1000 } as any);
    vi.spyOn(pub, 'getOrderBook').mockRejectedValueOnce(err).mockRejectedValueOnce(err);
    vi.spyOn(pub, 'getTrades').mockResolvedValue([]);
    const r = await fetchMarketOverview('btc_jpy');
    expect(r.ticker).toBeDefined();
    expect(r.orderBook).toEqual({ bids: [], asks: [] });
  });
});
