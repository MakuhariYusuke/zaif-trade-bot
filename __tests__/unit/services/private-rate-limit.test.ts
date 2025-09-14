import { describe, it, vi, expect } from 'vitest';
import * as adapters from '../../../src/api/adapters';
import { placeLimitOrder, init as initSvc } from '../../../src/services/market-service';

describe('private API rate-limit handling', () => {
  it('placeLimitOrder retries on 429 then throws', async () => {
    const mockApi: any = {
      trade: vi.fn()
        .mockRejectedValueOnce(new Error('429 Too Many Requests'))
        .mockRejectedValueOnce(new Error('429 Too Many Requests'))
        .mockRejectedValueOnce(new Error('429 Too Many Requests')),
    };
    vi.spyOn(adapters, 'createPrivateApi').mockReturnValue(mockApi);
    initSvc(mockApi);
    await expect(placeLimitOrder('btc_jpy', 'BUY' as any, 100, 0.1)).rejects.toThrow('Too Many Requests');
    expect(mockApi.trade).toHaveBeenCalledTimes(3);
  });
});
