import { describe, it, expect, beforeEach } from 'vitest';
import { InMemoryEventBus, setEventBus, getEventBus } from '../../../../ztb/application/events/bus';

describe('event bus assigns eventId', () => {
	beforeEach(() => setEventBus(new InMemoryEventBus()));
	it('assigns unique eventId', async () => {
		const got: any[] = [];
		getEventBus().subscribe('ORDER_SUBMITTED' as any, (e:any)=>{ got.push(e); });
		getEventBus().publish({ type: 'ORDER_SUBMITTED', requestId: 'r', pair: 'btc_jpy', side: 'buy', amount: 1, price: 1, orderId: 'x' } as any);
		await new Promise(r=>setTimeout(r,0));
		expect(got[0].eventId).toBeTruthy();
	});
});
