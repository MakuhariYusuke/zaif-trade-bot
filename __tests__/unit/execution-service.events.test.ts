import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { InMemoryEventBus, setEventBus, getEventBus } from '../../ztb/application/events/bus';
import { submitWithRetry, init as initExec, pollFillState } from '../../ztb/adapters/execution-service';
import * as market from '../../ztb/adapters/market-service';

describe('execution-service events', () => {
  beforeEach(() => {
    setEventBus(new InMemoryEventBus());
    process.env.DRY_RUN = '0';
    process.env.POLL_INTERVAL_MS = '10';
    process.env.POLL_MIN_CYCLES = '1';
    vi.restoreAllMocks();
  });
  afterEach(() => {
    (getEventBus() as any).clear?.();
  });

  it('emits ORDER_SUBMITTED on submit', async () => {
    vi.spyOn(market, 'placeLimitOrder').mockResolvedValue({ order_id: '123', amount: 1 } as any);
    // Inject fake PrivateApi for execution-service poll loop
    initExec({
      active_orders: vi.fn().mockResolvedValue([{ order_id: '123', pair: 'btc_jpy', side: 'bid', amount: 0, price: 100, timestamp: Date.now() }]),
      trade_history: vi.fn().mockResolvedValue([]),
      get_info2: vi.fn(),
      trade: vi.fn(),
      cancel_order: vi.fn().mockResolvedValue({ return: { order_id: '123' } })
    } as any);

    const events: any[] = [];
    getEventBus().subscribe('ORDER_SUBMITTED' as any, (e:any)=>{ events.push(e); });
    getEventBus().subscribe('ORDER_FILLED' as any, (e:any)=>{ events.push(e); });

    const res = await submitWithRetry({ currency_pair: 'btc_jpy', side: 'bid', limitPrice: 100, amount: 1, primaryTimeoutMs: 200, retryTimeoutMs: 200 });
    expect(res.filledQty).toBeGreaterThanOrEqual(0);

    const sub = events.find(e=>e.type==='ORDER_SUBMITTED');
    expect(sub).toBeTruthy();
    expect(sub).toHaveProperty('orderId');
    const filled = events.find(e=>e.type==='ORDER_FILLED');
    expect(filled).toBeTruthy();
  });

  it('emits SLIPPAGE_REPRICED when slip beyond threshold', async () => {
    vi.spyOn(market, 'placeLimitOrder').mockResolvedValue({ order_id: '200', amount: 1 } as any);
    initExec({
      active_orders: vi.fn().mockResolvedValue([]),
      trade_history: vi.fn().mockResolvedValue([{ order_id: '200', amount: 0.1, price: 102, timestamp: Date.now()/1000 }]),
      get_info2: vi.fn(), trade: vi.fn(), cancel_order: vi.fn().mockResolvedValue({ return: { order_id: '200' } })
    } as any);
    process.env.RISK_MAX_SLIPPAGE_PCT = '0.00001';
    process.env.REPRICE_MAX_ATTEMPTS = '1';
    const events: any[] = [];
    getEventBus().subscribe('SLIPPAGE_REPRICED' as any, (e:any)=>{ events.push(e); });
    await submitWithRetry({ currency_pair: 'btc_jpy', side: 'bid', limitPrice: 100, amount: 1, primaryTimeoutMs: 50, retryTimeoutMs: 50 });
    await new Promise(r=>setTimeout(r,10));
    expect(events.find(e=>e.type==='SLIPPAGE_REPRICED')).toBeTruthy();
  });

  it('emits ORDER_EXPIRED from pollFillState on timeout', async () => {
    const got: any[] = [];
    getEventBus().subscribe('ORDER_EXPIRED' as any, (e:any)=>{ got.push(e); });
    vi.spyOn(market, 'listActiveOrders').mockResolvedValue([{ order_id: 999, amount: 1, currency_pair: 'btc_jpy', action: 'bid', price: 100, timestamp: Date.now() }] as any);
    vi.spyOn(market, 'fetchTradeHistory').mockResolvedValue([] as any);
    await pollFillState('btc_jpy', { side: 'bid', intendedPrice: 100, amount: 1, orderId: 999, submittedAt: Date.now(), originalAmount: 1, requestId: 'req-x' }, 10, 5);
    await new Promise(r=>setTimeout(r,10));
    expect(got.length).toBeGreaterThan(0);
  });
});
