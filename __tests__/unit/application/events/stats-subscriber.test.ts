import { describe, it, expect, beforeEach, vi } from 'vitest';
import { InMemoryEventBus, setEventBus, getEventBus } from '../../../../src/application/events/bus';
import { registerStatsSubscriber } from '../../../../src/application/events/subscribers/stats-subscriber';
import * as stats from '../../../../src/utils/daily-stats';

describe('stats-subscriber', () => {
	beforeEach(() => { setEventBus(new InMemoryEventBus()); });

	it('appends PnL on sell ORDER_FILLED', async () => {
		const spy = vi.spyOn(stats, 'appendFillPnl').mockImplementation(() => undefined as any);
		registerStatsSubscriber();
		getEventBus().publish({ type: 'ORDER_FILLED', orderId: '1', requestId: 'r1', pair: 'btc_jpy', side: 'sell', amount: 0.1, price: 110, filled: 0.1, avgPrice: 100 } as any);
		await new Promise(r=>setTimeout(r,0));
		expect(spy).toHaveBeenCalled();
	});
});