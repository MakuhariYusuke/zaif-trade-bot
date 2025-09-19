import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { getEventBus, setEventBus } from '../../../src/application/events/bus';

let runTradeLive: any;

describe('trade-live phase escalation', () => {
	const TMP = path.resolve(process.cwd(), 'tmp-test-live-flow-' + Date.now() + '-' + Math.random().toString(36).slice(2));
	const cfgFile = path.join(TMP, 'trade-config.json');
	const stateFile = path.join(TMP, 'trade-state.json');
	const logDir = path.join(TMP, 'logs');

	let events: any[] = [];
	beforeEach(async () => {
		fs.mkdirSync(TMP, { recursive: true });
		process.env.TEST_MODE = '1';
		process.env.PROMO_TO2_DAYS = '1';
		process.env.PROMO_TO3_SUCCESS = '20';
		process.env.PROMO_TO4_SUCCESS = '9999';
		process.env.TRADE_CONFIG_FILE = cfgFile;
		process.env.TRADE_STATE_FILE = stateFile;
		process.env.LOG_DIR = logDir;

		fs.writeFileSync(cfgFile, JSON.stringify({
			pair: 'btc_jpy',
			phase: 1,
			phaseSteps: [
				{ phase: 1, ordersPerDay: 1 },
				{ phase: 2, ordersPerDay: 3 },
				{ phase: 3, ordersPerDay: 5 },
				{ phase: 4, ordersPerDay: 10 }
			]
		}, null, 2));
		fs.writeFileSync(stateFile, JSON.stringify({ phase: 1, consecutiveDays: 0, totalSuccess: 0, lastDate: '' }, null, 2));
		setEventBus(new (getEventBus().constructor as any)());
		events = [];
		getEventBus().subscribe('EVENT/TRADE_PHASE' as any, (ev: any) => { events.push(ev); });
		const mod = await import('../../../src/tools/trade-live');
		runTradeLive = mod.runTradeLive;
	});

	it('promotes 1->2->3 with escalating plannedOrders', async () => {
		let out = await runTradeLive({ today: '2025-01-01', daySuccess: 1 });
		expect(out.phase).toBe(2);
		expect(out.plannedOrders).toBe(3);
		for (let i = 2; i <= 30; i++) {
			out = await runTradeLive({ today: '2025-01-' + (i < 10 ? '0'+i : i), daySuccess: 1 });
			if (out.phase === 3) break;
		}
		expect(out.phase).toBe(3);
		expect(out.plannedOrders).toBe(10);
		const stRaw = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
		expect(stRaw.phase).toBe(3);
		expect(stRaw.ordersPerDay).toBe(10);
		const transitions = events.filter(e => e.fromPhase !== e.toPhase).map(e => `${e.fromPhase}>${e.toPhase}`);
		expect(transitions).toContain('1>2');
		expect(transitions).toContain('2>3');
	});
});