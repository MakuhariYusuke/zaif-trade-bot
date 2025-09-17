import { describe, it, expect, beforeEach, vi } from 'vitest';
import { InMemoryEventBus, setEventBus, getEventBus } from '../../../src/application/events/bus';
import { registerPositionSubscriber } from '../../../src/application/events/subscribers/position-subscriber';
import * as store from '../../../src/adapters/position-store';

describe('position-subscriber', () => {
  beforeEach(() => { setEventBus(new InMemoryEventBus()); });

  it('updates position on ORDER_FILLED', async () => {
  const spy = vi.spyOn(store, 'updateOnFill').mockImplementation(() => undefined as any);
    registerPositionSubscriber();
    getEventBus().publish({ type: 'ORDER_FILLED', orderId: '1', requestId: 'r1', pair: 'btc_jpy', side: 'buy', amount: 0.1, price: 100, filled: 0.1, avgPrice: 100 } as any);
    await new Promise(r=>setTimeout(r,0));
    expect(spy).toHaveBeenCalled();
  });
});
