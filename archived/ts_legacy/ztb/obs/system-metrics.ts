import { log } from '../utils/logger';
import { monitorEventLoopDelay } from 'perf_hooks';

export interface SystemMetricsOptions { intervalMs?: number; }
let timer: NodeJS.Timeout | null = null;
let loopHist: ReturnType<typeof monitorEventLoopDelay> | null = null;

export function startSystemMetrics(opts: SystemMetricsOptions = {}) {
  const isTest = !!process.env.VITEST_WORKER_ID || process.env.NODE_ENV === 'test' || process.env.TEST_MODE === '1';
  const envInterval = Number(process.env.SYSTEM_METRICS_INTERVAL_MS || 60000);
  const intervalMs = opts.intervalMs || envInterval || 60000;
  if (intervalMs <= 0) return () => {};
  if (timer) return stopSystemMetrics; // already started
  try { loopHist = monitorEventLoopDelay({ resolution: 10 }); loopHist.enable(); } catch {}
  const emit = () => {
    try {
      const mem = process.memoryUsage();
      const rssMb = +(mem.rss / 1024 / 1024).toFixed(1);
      const heapUsedMb = +(mem.heapUsed / 1024 / 1024).toFixed(1);
      const heapTotalMb = +(mem.heapTotal / 1024 / 1024).toFixed(1);
      const extMb = +(mem.external / 1024 / 1024).toFixed(1);
      const activeHandles = (process as any)._getActiveHandles?.().length ?? undefined;
      const activeRequests = (process as any)._getActiveRequests?.().length ?? undefined;
      const loopP95 = loopHist ? Number((loopHist.percentile(95) / 1e6).toFixed(2)) : undefined; // ms
      const payload = { rssMb, heapUsedMb, heapTotalMb, extMb, activeHandles, activeRequests, loopP95 };
      log('INFO','SYS','metrics', payload);
      if (loopHist) loopHist.reset();
    } catch {}
  };
  emit();
  timer = setInterval(emit, intervalMs).unref?.();
  if (isTest && intervalMs > 200) { /* In tests we keep it short by manual override if needed */ }
  return stopSystemMetrics;
}

export function stopSystemMetrics(){ if (timer) { try { clearInterval(timer); } catch {} timer = null; } if (loopHist) { try { loopHist.disable(); } catch {} loopHist=null; }
}
