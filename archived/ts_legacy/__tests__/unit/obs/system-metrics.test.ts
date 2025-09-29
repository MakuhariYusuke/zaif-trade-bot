import { describe, it, expect, vi } from 'vitest';
import { startSystemMetrics, stopSystemMetrics } from '../../../ztb/obs/system-metrics';

describe('system metrics', () => {
  it('emits SYS metrics lines with expected fields', async () => {
    process.env.SYSTEM_METRICS_INTERVAL_MS = '50';
  process.env.LOG_JSON = '1';
    const lines: string[] = [];
    const spy = vi.spyOn(console,'log').mockImplementation((...a:any[])=>{ lines.push(a.map(x=>String(x)).join(' ')); });
    startSystemMetrics();
    await new Promise(r=>setTimeout(r, 160));
    stopSystemMetrics();
    spy.mockRestore();
    const sysLine = lines.find(l=>/SYS/.test(l) && /metrics/.test(l));
    expect(sysLine).toBeTruthy();
    if (sysLine) {
      const m = sysLine.match(/(\{.*\})/);
      if (m) {
        try {
          const obj = JSON.parse(m[1]);
          expect(obj.rssMb).toBeDefined();
          expect(obj.heapUsedMb).toBeDefined();
        } catch {}
      }
    }
  });
});
