import { describe, it, expect } from 'vitest';

describe('live-minimal DRY_RUN path', () => {
  it('decideOrderType infers market when rateInput empty', async () => {
    const m = await import('../../../ztb/tools/live-minimal');
    const r = m.decideOrderType('');
    expect(r.isMarket).toBe(true);
    expect(r.rateOverride).toBe(0);
  });
});
