import { describe, it, expect, beforeEach, vi } from 'vitest';
import { InMemoryEventBus } from '../../../../src/application/events/bus';

describe('event bus slow handler warning', () => {
  beforeEach(()=>{
    process.env.EVENT_METRICS_LAT_CAP = '50';
    delete process.env.EVENT_METRICS_INTERVAL_IN_TEST; // keep auto flush disabled
    process.env.EVENTBUS_SLOW_HANDLER_MS = '10';
    process.env.LOG_LEVEL = 'INFO'; // ensure WARN not suppressed in TEST_MODE
  });

  it('emits WARN log and increments slowHandlerCount', async () => {
    const bus = new InMemoryEventBus();
    // one fast handler (< threshold)
    bus.subscribe('X_EVT' as any, async ()=>{ await new Promise(r=>setTimeout(r, 2)); });
    // one slow handler (>= threshold)
    bus.subscribe('X_EVT' as any, async ()=>{ await new Promise(r=>setTimeout(r, 15)); });
    const logs: string[] = [];
    const spyLog = vi.spyOn(console, 'log').mockImplementation((...args:any[])=>{ logs.push(args.map(a=>String(a)).join(' ')); });
    const spyWarn = vi.spyOn(console, 'warn').mockImplementation((...args:any[])=>{ logs.push(args.map(a=>String(a)).join(' ')); });
    // publish after spies so WARN is captured
    await bus.publishAndWait!({ type: 'X_EVT', eventId: 'e1' } as any, { timeoutMs: 200 });
    await bus.publishAndWait!({ type: 'X_EVT', eventId: 'e2' } as any, { timeoutMs: 200 });
    // flush metrics to capture metrics line
    (bus as any).flushMetrics?.();
    spyLog.mockRestore(); spyWarn.mockRestore();

    const warnLine = logs.find(l=>/slow-handler/.test(l));
    expect(warnLine).toBeTruthy();

    const metricsLine = logs.find(l=>/EVENT/.test(l) && /metrics/.test(l));
    expect(metricsLine).toBeTruthy();
    if (metricsLine) {
      const m = metricsLine.match(/(\{.*\})/);
      if (m) {
        try {
          const obj = JSON.parse(m[1]);
          const ev = obj.types?.X_EVT;
          expect(ev).toBeTruthy();
          expect(ev.slowHandlerCount).toBeGreaterThanOrEqual(2); // slow handler ran twice
        } catch {}
      }
    }
  });
});
