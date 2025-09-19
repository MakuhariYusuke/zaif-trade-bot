import { loadTradeConfig, getOrdersPerDay } from '../config/trade-config';
import { applyPhaseProgress, loadTradeState, saveTradeState, getPromotionRules } from '../config/trade-state';
import { getEventBus } from '../application/events/bus';
import { logInfo, logWarn } from '../utils/logger';

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
    const out = { pair: cfg.pair, phase, plannedOrders: planned, today, executed: 0, attempts: 0, cumulativePnl: 0 };
    try { logInfo('[TRADE-PLAN] ' + JSON.stringify(out)); } catch {}
    return out;
  }

  // live mode simplified execution loop with guards
  const maxOrdersPerDay = cfg.maxOrdersPerDay || Infinity;
  const maxLossPerDay = cfg.maxLossPerDay || Infinity; // currency (e.g. JPY)
  const slipGuard = cfg.slippageGuardPct || 0;

  let executed = 0; // successful executions counting towards phase progress
  let attempts = 0; // includes failed/slippage
  let cumulativePnl = 0; // realized PnL aggregate
  const successes: any[] = [];

  // For now we simulate orders rather than real placement.
  const simulateFill = (i: number) => {
    // Dummy simulated price behavior with minor variation
    const intendedPrice = 100 + i; // base
    const filledPrice = intendedPrice * (1 + (Math.random() - 0.5) * 0.002); // +/-0.1%
    const side: 'BUY' | 'SELL' = (i % 2 === 0) ? 'BUY' : 'SELL';
    const qty = 0.1;
    // PnL: SELL when price > earlier base approximated; rough dummy formula
    const pnl = side === 'SELL' ? (filledPrice - intendedPrice) * qty : 0;
    return { intendedPrice, filledPrice, qty, side, pnl };
  };

  for (let i = 0; i < planned; i++) {
    // Order-level guards BEFORE attempting
    if (executed >= maxOrdersPerDay) {
      try { logWarn('[GUARD] maxOrdersPerDay reached'); } catch {}
      try { bus.publish({ type: 'EVENT/TRADE_EXECUTED', ts: Date.now(), pair: cfg.pair, side: 'BUY', qty: 0, price: 0, success: false, requestId: 'guard-max-orders', reason: 'MAX_ORDERS' } as any, { async: !isTest }); } catch {}
      break;
    }
    if (cumulativePnl <= -Math.abs(maxLossPerDay)) {
      try { logWarn('[GUARD] maxLossPerDay exceeded'); } catch {}
      try { bus.publish({ type: 'EVENT/TRADE_EXECUTED', ts: Date.now(), pair: cfg.pair, side: 'BUY', qty: 0, price: 0, success: false, requestId: 'guard-max-loss', reason: 'MAX_LOSS' } as any, { async: !isTest }); } catch {}
      break;
    }

    attempts++;
    const sim = simulateFill(i);
    const slipRatio = Math.abs(sim.filledPrice - sim.intendedPrice) / sim.intendedPrice;
    if (slipGuard > 0 && slipRatio > slipGuard) {
      try { logWarn(`[SLIPPAGE] rejected intended=${sim.intendedPrice} filled=${sim.filledPrice} ratio=${(slipRatio*100).toFixed(3)}% > ${(slipGuard*100).toFixed(2)}%`); } catch {}
      try { bus.publish({ type: 'EVENT/TRADE_EXECUTED', ts: Date.now(), pair: cfg.pair, side: sim.side, qty: sim.qty, price: sim.filledPrice, success: false, requestId: `slip-${i}`, reason: 'SLIPPAGE' } as any, { async: !isTest }); } catch {}
      continue;
    }
    cumulativePnl += sim.pnl;
    executed++;
    successes.push(sim);
    try { bus.publish({ type: 'EVENT/TRADE_EXECUTED', ts: Date.now(), pair: cfg.pair, side: sim.side, qty: sim.qty, price: sim.filledPrice, pnl: sim.pnl, success: true, requestId: `exec-${i}` } as any, { async: !isTest }); } catch {}
  }

  const { state: next, promotion } = applyPhaseProgress(state, { date: today, daySuccess: executed }, getPromotionRules());
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
    if ((next as any).ordersPerDay) planned = (next as any).ordersPerDay;
  } else if (executed > 0) {
    try {
      const ev: any = { type: 'EVENT/TRADE_PHASE', ts, pair: cfg.pair, fromPhase: next.phase, toPhase: next.phase, reason: isTest ? 'test-mode' : 'no-promotion' };
      if (isTest && typeof (bus as any).publishAndWait === 'function') {
        await (bus as any).publishAndWait(ev, { timeoutMs: 200, captureErrors: false });
      } else {
        bus.publish(ev, { async: !isTest });
      }
    } catch {}
  }
  return { pair: cfg.pair, phase: next.phase, plannedOrders: planned, today, executed, attempts, cumulativePnl };
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
