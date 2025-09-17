import { describe, it, expect, beforeEach } from 'vitest';
import { InMemoryEventBus, setEventBus, getEventBus } from '../../../src/application/events/bus';
import { registerLoggerSubscriber } from '../../../src/application/events/subscribers/logger-subscriber';
import { setupJsonLogs, captureLogs, expectJsonLog } from '../../helpers/logging';

describe('logger-subscriber', () => {
  beforeEach(() => { setEventBus(new InMemoryEventBus()); setupJsonLogs(); });

  it('logs all ORDER events with INFO', async () => {
    const logs = captureLogs();
    registerLoggerSubscriber();
    const ev = { type: 'ORDER_SUBMITTED', orderId: '1', requestId: 'r1', pair: 'btc_jpy', side: 'buy', amount: 0.1, price: 100 } as any;
    getEventBus().publish(ev);
    await new Promise(r=>setTimeout(r,0));
    expectJsonLog(logs, 'ORDER-EVENT', 'INFO', 'ORDER_SUBMITTED', ['requestId','pair','side','amount','price','orderId']);
  });
});
