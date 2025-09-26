import { describe, it, expect, beforeEach } from 'vitest';
import fs from 'fs';
import path from 'path';
import { getEventBus, setEventBus } from '../../../ztb/application/events/bus';

describe('tools/trade-live flow', () => {
  const ROOT = process.cwd();
  const TMP = path.resolve(ROOT, `tmp-test-live-flow-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  const stateFile = path.join(TMP, 'trade-state.json');
  beforeEach(()=>{
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(TMP, { recursive: true });
    process.env = { ...process.env, TEST_MODE: '1', DRY_RUN: '1', TRADE_STATE_FILE: stateFile, PROMO_TO2_DAYS: '1' };
    const cfgFile = path.join(TMP, 'trade-config.json');
    fs.writeFileSync(cfgFile, JSON.stringify({ pair: 'btc_jpy', phase: 1, phaseSteps: [{ phase:1, ordersPerDay: 1 }], slippageGuardPct: 0.5 }, null, 2), 'utf8');
    process.env.TRADE_CONFIG_FILE = cfgFile;
    setEventBus(new (getEventBus().constructor as any)());
  });

  it('emits TRADE_PLAN on dry-run and TRADE_PHASE on live promotion', async () => {
    const evs: any[] = [];
    const bus = getEventBus();
    bus.subscribe('EVENT/TRADE_PLAN' as any, (ev: any) => { evs.push(ev); });
    bus.subscribe('EVENT/TRADE_PHASE' as any, (ev: any) => { evs.push(ev); });
    const mod = await import('../../../ztb/tools/trade-live');
    const sum1 = await mod.runTradeLive({ dryRun: true });
    expect(sum1?.plannedOrders ?? 0).toBeGreaterThanOrEqual(1);
    expect(evs.find(e => e.type === 'EVENT/TRADE_PLAN')).toBeTruthy();
    fs.writeFileSync(stateFile, JSON.stringify({ phase: 1, today: '2099-01-01', totalSuccess: 3 }), 'utf8');
    process.env.DRY_RUN = '0';
    const sum2 = await mod.runTradeLive({ dryRun: false, today: '2099-01-01' } as any);
    expect(sum2).toBeTruthy();
    expect(evs.find(e => e.type === 'EVENT/TRADE_PHASE')).toBeTruthy();
    const s = JSON.parse(fs.readFileSync(stateFile,'utf8'));
    expect(s.phase).toBeGreaterThanOrEqual(2);
  }, 15000);
});
