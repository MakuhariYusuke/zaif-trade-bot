import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import path from 'path';
import fs from 'fs';

// Mock dependencies used inside execution-service
vi.mock('../../../src/services/market-service', () => {
  return {
    placeLimitOrder: vi.fn(async (_pair: string, _side: any, _price: number, _amount: number) => {
      return { order_id: 'ORDER1' };
    }),
    cancelOrder: vi.fn(async () => ({ result: true })),
  };
});

vi.mock('../../../src/api/private', () => ({
  getAndResetLastRequestNonceRetries: () => 0,
}));

describe('services/execution-service submitWithRetry', () => {
  const TMP = path.resolve(process.cwd(), '.tmp-exec-retry');

  beforeEach(() => {
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
    process.env.POSITION_STORE_DIR = path.join(TMP, 'positions');
    process.env.STATS_DIR = path.join(TMP, 'logs');
    process.env.POLL_INTERVAL_MS = '10';
    process.env.POLL_MIN_CYCLES = '1';
  });
  afterEach(() => { vi.clearAllMocks(); });

  it('returns immediate filled result in DRY_RUN mode', async () => {
    process.env.DRY_RUN = '1';
    const { submitWithRetry } = await import('../../../src/services/execution-service');
    const sum = await submitWithRetry({ currency_pair: 'btc_jpy', side: 'bid', limitPrice: 100, amount: 0.1, primaryTimeoutMs: 50, retryTimeoutMs: 50 });
    expect(sum.filledQty).toBeCloseTo(0.1, 10);
    expect(sum.filledCount).toBe(1);
    expect(sum.slippagePct).toBe(0);
  });

  it('first poll not filled then retry submission fills', async () => {
    delete process.env.DRY_RUN;
    const { init, submitWithRetry } = await import('../../../src/services/execution-service');
    const ms = await import('../../../src/services/market-service');

    // First submission returns ORDER1, second submission will return ORDER2
    (ms.placeLimitOrder as any).mockImplementationOnce(async () => ({ order_id: 'ORDER1' }));
    ;(ms.placeLimitOrder as any).mockImplementationOnce(async () => ({ order_id: 'ORDER2' }));

    // Mock PrivateApi hooks used internally through init()
    const activeOrdersSeq: any[] = [
      // first poll: ORDER1 open with remaining amount
      [{ order_id: 'ORDER1', amount: 0.1 }],
      // then assume it disappears but not filled fully -> triggers retry after timeout
      [],
      // for order2, nothing open
      [],
    ];
    let activeIdx = 0;
    const tradeHistoryByOrder: Record<string, any[]> = {
      ORDER1: [ /* small partial */ { order_id: 'ORDER1', amount: 0.02, price: 100, timestamp: Math.floor(Date.now()/1000) } ],
      ORDER2: [ { order_id: 'ORDER2', amount: 0.1, price: 99, timestamp: Math.floor(Date.now()/1000) } ],
    };
    const privApi: any = {
      active_orders: async () => {
        const v = activeOrdersSeq[Math.min(activeIdx, activeOrdersSeq.length - 1)];
        activeIdx++;
        return v;
      },
      trade_history: async () => {
        // Return history for both orders so that poll loops can detect fills by id
        return [ ...tradeHistoryByOrder.ORDER1, ...tradeHistoryByOrder.ORDER2 ];
      },
    };
    init(privApi);

    const res = await submitWithRetry({ currency_pair: 'btc_jpy', side: 'ask', limitPrice: 100, amount: 0.1, primaryTimeoutMs: 60, retryTimeoutMs: 60, improvePricePct: 0.001 });
    expect(res.filledQty).toBeGreaterThan(0);
    expect(res.filledCount).toBe(1);
    expect(res.submitRetryCount).toBeGreaterThanOrEqual(1);
  });
});
