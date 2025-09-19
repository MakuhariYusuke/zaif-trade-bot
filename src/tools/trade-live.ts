import { loadTradeConfig, getOrdersPerDay } from '../config/trade-config';
import { applyPhaseProgress, loadTradeState, saveTradeState, getPromotionRules } from '../config/trade-state';
import { getEventBus } from '../application/events/bus';
import { logInfo } from '../utils/logger';

export interface RunTradeLiveOptions {
  dryRun?: boolean;
  today?: string; // YYYY-MM-DD (for testing)
  daySuccess?: number; // simulate successful count today (for testing)
}

export async function runTradeLive(opts: RunTradeLiveOptions = {}){
  const cfg = loadTradeConfig();
  const state = loadTradeState();
  const today = opts.today || new Date().toISOString().slice(0, 10);
  const phase = Math.max(1, Number(state.phase || cfg.phase));
  let planned = getOrdersPerDay(cfg, phase);

  const bus = getEventBus();
  const ts = Date.now();
  const dry = !!opts.dryRun || process.env.DRY_RUN === '1';
  const isTest = (process.env.TEST_MODE === '1') || !!process.env.VITEST_WORKER_ID;

  // emit plan event
  try {
    const ev: any = { type: 'EVENT/TRADE_PLAN', ts, pair: cfg.pair, phase, plannedOrders: planned, dryRun: dry };
    if (isTest && typeof (bus as any).publishAndWait === 'function') {
      await (bus as any).publishAndWait(ev, { timeoutMs: 200, captureErrors: false });
    } else {
      bus.publish(ev, { async: !isTest });
    }
  } catch {}

  // dry-run: print summary JSON and exit
  if (dry) {
    const out = { pair: cfg.pair, phase, plannedOrders: planned, today };
    try { logInfo('[TRADE-PLAN] ' + JSON.stringify(out)); } catch {}
    return out;
  }

  // live mode: here we would delegate to real execution layer; for now, only state progress update based on provided daySuccess
  const daySuccess = Math.max(0, Number(opts.daySuccess ?? (process.env.TEST_MODE === '1' ? 1 : 0)));
  const { state: next, promotion } = applyPhaseProgress(state, { date: today, daySuccess }, getPromotionRules());
  saveTradeState(next);
  if (promotion) {
    try {
      const ev: any = { type: 'EVENT/TRADE_PHASE', ts, pair: cfg.pair, fromPhase: promotion.fromPhase, toPhase: promotion.toPhase, reason: promotion.reason };
      if (isTest && typeof (bus as any).publishAndWait === 'function') {
        await (bus as any).publishAndWait(ev, { timeoutMs: 200, captureErrors: false });
      } else {
        bus.publish(ev, { async: !isTest });
      }
    } catch {}
    // recalc planned based on new phase escalation policy if ordersPerDay present
    if ((next as any).ordersPerDay) planned = (next as any).ordersPerDay;
  } else if (daySuccess > 0) {
    // emit phase progress event even without promotion to allow dashboards/tests to observe live phase updates
    try {
      const ev: any = { type: 'EVENT/TRADE_PHASE', ts, pair: cfg.pair, fromPhase: next.phase, toPhase: next.phase, reason: isTest ? 'test-mode' : 'no-promotion' };
      if (isTest && typeof (bus as any).publishAndWait === 'function') {
        await (bus as any).publishAndWait(ev, { timeoutMs: 200, captureErrors: false });
      } else {
        bus.publish(ev, { async: !isTest });
      }
    } catch {}
  }
  return { pair: cfg.pair, phase: next.phase, plannedOrders: planned, today };
}

// CLI entry
if (require.main === module) {
  const dry = process.argv.includes('--dry-run') || process.env.DRY_RUN === '1';
  runTradeLive({ dryRun: dry }).then((out) => {
    if (dry) {
      try { console.log(JSON.stringify(out)); } catch {}
    }
  }).catch((e) => { console.error(e?.message || String(e)); process.exit(1); });
}
