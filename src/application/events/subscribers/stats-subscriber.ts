import { getEventBus } from '../bus';
import { appendFillPnl } from '../../../utils/daily-stats';
import { log } from '../../../utils/logger';
import { getCircuitBreaker } from '../../circuit-breaker';

function today(){ return new Date().toISOString().slice(0,10); }

type Sample = { ok: boolean; latencyMs: number };
const state = {
  win: 0,
  lose: 0,
  recent: [] as Sample[],
  submittedAtByReq: new Map<string, number>(),
  submittedAtByOrder: new Map<string, number>(),
  consecFail: 0,
  lastInfoAt: 0,
  window: Math.max(10, Number(process.env.STATS_WINDOW || '50')),
};

function pushSample(s: Sample){
  state.recent.push(s);
  if (state.recent.length > state.window) state.recent.shift();
}
function pctSuccess(){ const n = state.recent.length || 1; const ok = state.recent.filter(x=>x.ok).length; return ok / n; }
function medianLatency(){ const arr = state.recent.map(x=>x.latencyMs).filter(n=>Number.isFinite(n)); if (!arr.length) return 0; const a = [...arr].sort((a,b)=>a-b); const m = Math.floor(a.length/2); return a.length%2? a[m] : (a[m-1]+a[m])/2; }
function infoSnapshot(pair: string){
  const now = Date.now();
  if (now - state.lastInfoAt < 5000) return; // throttle
  state.lastInfoAt = now;
  try { log('INFO','STATS','snapshot',{ pair, window: state.window, successRate: pctSuccess(), consecFail: state.consecFail, medianLatencyMs: medianLatency(), wins: state.win, losses: state.lose }); } catch {}
}
function maybeWarn(pair: string){
  const rate = pctSuccess();
  const med = medianLatency();
  const lowSuccess = state.recent.length >= Math.min(20, state.window) && rate < 0.5;
  const tooManyConsec = state.consecFail > 5;
  const slow = med > 30000;
  if (lowSuccess || tooManyConsec || slow){
    try { log('WARN','STATS','anomaly',{ pair, window: state.window, successRate: rate, consecFail: state.consecFail, medianLatencyMs: med, reasons: { lowSuccess, tooManyConsec, slow } }); } catch {}
  }
}

export function registerStatsSubscriber(){
  const bus = getEventBus();
  bus.subscribe('ORDER_SUBMITTED' as any, (ev: any) => {
    try {
      const now = Date.now();
      if (ev.requestId) state.submittedAtByReq.set(ev.requestId, now);
      if (ev.orderId) state.submittedAtByOrder.set(String(ev.orderId), now);
    } catch {}
  });
  bus.subscribe('ORDER_FILLED' as any, (ev: any) => {
    try {
      // daily stat helper (0 placeholder to avoid double count)
      if (ev.side === 'sell') appendFillPnl(today(), 0, ev.pair);
    } catch {}
    try {
      const key = ev.requestId || ev.orderId;
      const start = (ev.requestId && state.submittedAtByReq.get(ev.requestId)) || (ev.orderId && state.submittedAtByOrder.get(String(ev.orderId))) || Date.now();
      const lat = Math.max(0, Date.now() - start);
      state.win++; state.consecFail = 0; pushSample({ ok: true, latencyMs: lat });
      try {
        // default global CB; category-specific recording is handled at call sites (BaseService)
        getCircuitBreaker().recordSuccess(lat);
      } catch {}
      infoSnapshot(ev.pair); maybeWarn(ev.pair);
    } catch {}
  });
  const onFail = (ev: any) => {
    try {
      appendFillPnl(today(), 0, ev.pair);
    } catch {}
    try {
      const start = (ev.requestId && state.submittedAtByReq.get(ev.requestId)) || (ev.orderId && state.submittedAtByOrder.get(String(ev.orderId))) || Date.now();
      const lat = Math.max(0, Date.now() - start);
      state.lose++; state.consecFail++; pushSample({ ok: false, latencyMs: lat });
      try {
        getCircuitBreaker().recordFailure(ev?.cause);
      } catch {}
      infoSnapshot(ev.pair); maybeWarn(ev.pair);
    } catch {}
  };
  bus.subscribe('ORDER_CANCELED' as any, onFail);
  bus.subscribe('ORDER_EXPIRED' as any, onFail);
}
