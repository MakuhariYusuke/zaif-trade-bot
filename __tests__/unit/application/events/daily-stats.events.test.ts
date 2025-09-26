import { describe, it, expect, beforeEach, vi } from 'vitest';
import { InMemoryEventBus, setEventBus, getEventBus } from '../../../../ztb/application/events/bus';
import { registerStatsSubscriber } from '../../../../ztb/application/events/subscribers/stats-subscriber';
import * as stats from '../../../../ztb/utils/daily-stats';

describe('daily-stats subscriber', () => {
	beforeEach(() => { setEventBus(new InMemoryEventBus()); });

	it('increments stats on ORDER_FILLED', async () => {
		const spy = vi.spyOn(stats, 'appendFillPnl').mockImplementation(() => undefined as any);
		registerStatsSubscriber();
		getEventBus().publish({ type: 'ORDER_FILLED', orderId: '1', requestId: 'r1', pair: 'btc_jpy', side: 'sell', amount: 0.1, price: 110, filled: 0.1, avgPrice: 100 } as any);
		await new Promise(r=>setTimeout(r,0));
		expect(spy).toHaveBeenCalled();
	});

	it('count expired/canceled as attempts', async () => {
		const spy = vi.spyOn(stats, 'appendFillPnl').mockImplementation(() => undefined as any);
		registerStatsSubscriber();
		getEventBus().publish({ type: 'ORDER_EXPIRED', orderId: '2', requestId: 'r2', pair: 'btc_jpy', side: 'buy', amount: 1, price: 100 } as any);
		getEventBus().publish({ type: 'ORDER_CANCELED', orderId: '3', requestId: 'r3', pair: 'btc_jpy', side: 'buy', amount: 1, price: 100 } as any);
		await new Promise(r=>setTimeout(r,0));
		expect(spy).toHaveBeenCalledTimes(2);
	});
});
