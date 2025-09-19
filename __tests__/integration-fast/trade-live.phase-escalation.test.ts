import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { getEventBus, setEventBus } from '../../src/application/events/bus';

// We'll call runTradeLive directly; events bus in TEST_MODE should publishAndWait
// runTradeLive は環境変数 (PROMO_*) 反映後に動的 import する
let runTradeLive: any;

/**
 * Integration-fast: phase escalation & ordersPerDay mapping
 * - Mock short promotion thresholds via env (TEST_MODE=1 triggers to2 threshold=1)
 * - Simulate several days with at least 1 success to trigger promotions 1->2, then accumulate totalSuccess to reach 3
 * - Verify EVENT/TRADE_PHASE events fired with correct from/to progression
 * - Verify plannedOrders aligns to state.ordersPerDay after promotion (1,3,10 per mapping)
 */

describe('trade-live phase escalation', () => {
  const TMP = path.resolve(process.cwd(), 'tmp-test-live-flow-' + Date.now() + '-' + Math.random().toString(36).slice(2));
  const cfgFile = path.join(TMP, 'trade-config.json');
  const stateFile = path.join(TMP, 'trade-state.json');
  const logDir = path.join(TMP, 'logs');

  let events: any[] = [];
  beforeEach(async () => {
    fs.mkdirSync(TMP, { recursive: true });
    process.env.TEST_MODE = '1'; // ensures to2_consecutiveDays = 1
    process.env.PROMO_TO2_DAYS = '1';
    process.env.PROMO_TO3_SUCCESS = '20';
    process.env.PROMO_TO4_SUCCESS = '9999'; // avoid accidental phase 4
    process.env.TRADE_CONFIG_FILE = cfgFile;
    process.env.TRADE_STATE_FILE = stateFile;
    process.env.LOG_DIR = logDir; // if logger uses it

    // custom compact config (ordersPerDay for phase 1..4). Note phase 3 base config ordersPerDay=5 but state escalation overrides to 10.
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
    // fresh state
    fs.writeFileSync(stateFile, JSON.stringify({ phase: 1, consecutiveDays: 0, totalSuccess: 0, lastDate: '' }, null, 2));
  // reset bus and capture TRADE_PHASE events, then dynamic import after env setup
    setEventBus(new (getEventBus().constructor as any)());
    events = [];
    getEventBus().subscribe('EVENT/TRADE_PHASE' as any, (ev: any) => { events.push(ev); });
    const mod = await import('../../src/tools/trade-live');
    runTradeLive = mod.runTradeLive;
  });

  it('promotes 1->2->3 with escalating plannedOrders', async () => {
    // Day 1: success triggers promotion 1->2 (because TEST_MODE => to2 threshold=1)
  let out = await runTradeLive({ today: '2025-01-01', daySuccess: 1 });
    expect(out.phase).toBe(2);
    expect(out.plannedOrders).toBe(3); // mapping after promotion

    // Accumulate successes to exceed to3_totalSuccess (default 20). We'll loop days until promotion.
    // We already have totalSuccess = 1. Need >=20.
    for (let i = 2; i <= 30; i++) {
      out = await runTradeLive({ today: '2025-01-' + (i < 10 ? '0'+i : i), daySuccess: 1 });
      if (out.phase === 3) break;
    }
    expect(out.phase).toBe(3);
    // After promotion, plannedOrders should use escalation map (10)
    expect(out.plannedOrders).toBe(10);

    // Validate state file contents
    const stRaw = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
    expect(stRaw.phase).toBe(3);
    expect(stRaw.ordersPerDay).toBe(10);
    const transitions = events.filter(e => e.fromPhase !== e.toPhase).map(e => `${e.fromPhase}>${e.toPhase}`);
    expect(transitions).toContain('1>2');
    expect(transitions).toContain('2>3');
  });
});
