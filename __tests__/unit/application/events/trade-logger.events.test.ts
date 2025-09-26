import { describe, it, expect, beforeEach } from 'vitest';
import { InMemoryEventBus, setEventBus, getEventBus } from '../../../../ztb/application/events/bus';
import { registerTradeLoggerSubscriber } from '../../../../ztb/application/events/subscribers/trade-logger-subscriber';
import fs from 'fs';
import path from 'path';

describe('trade-logger subscriber', () => {
	const LOG_DIR = path.resolve(process.cwd(), 'logs');
	const fileFor = () => path.join(LOG_DIR, `trades-${new Date().toISOString().slice(0,10)}.log`);
	beforeEach(() => {
		setEventBus(new InMemoryEventBus());
		try { fs.rmSync(LOG_DIR, { recursive: true, force: true }); } catch {}
	});

	it('appends JSON lines for events', async () => {
		registerTradeLoggerSubscriber();
		getEventBus().publish({ type: 'ORDER_SUBMITTED', orderId: '1', requestId: 'r1', pair: 'btc_jpy', side: 'buy', amount: 0.1, price: 100 } as any);
		await new Promise(r=>setTimeout(r,0));
		const file = fileFor();
		expect(fs.existsSync(file)).toBe(true);
		const content = fs.readFileSync(file, 'utf8');
		expect(content).toMatch(/\[EVENT\] ORDER_SUBMITTED/);
	});
});
