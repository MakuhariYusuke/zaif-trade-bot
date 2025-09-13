// Application entrypoint that delegates to the app layer.
// - Re-exports strategy helpers from app/index.ts
// - Starts the periodic strategy loop when this file is executed directly (npm start)

import dotenv from 'dotenv';
dotenv.config();

// Importing app/index.ts performs one-time initialization (adapters, services, etc.)
import { strategyOnce } from './app/index';
export { strategyOnce };

// Simple runner so npm start (ts-node src/index.ts) executes the loop.
if (require.main === module) {
	(async () => {
		const pair = process.env.PAIR || 'btc_jpy';
		const intervalMs = Number(process.env.LOOP_INTERVAL_MS || 15000);
		const EXECUTE = process.env.DRY_RUN !== '1';
		for (;;) {
			await strategyOnce(pair, EXECUTE);
			await new Promise((r) => setTimeout(r, intervalMs));
		}
	})().catch((err) => {
		console.error('[FATAL] entry runner error', err?.message || err);
		process.exitCode = 1;
	});
}