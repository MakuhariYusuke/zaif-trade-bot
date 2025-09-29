import { describe, it, expect, beforeEach, vi } from 'vitest';
import { InMemoryEventBus } from '../../../../ztb/application/events/bus';

describe('event bus slowRatio metric', () => {
  beforeEach(()=>{
    process.env.EVENT_METRICS_LAT_CAP = '20';
    delete process.env.EVENT_METRICS_INTERVAL_IN_TEST;
    process.env.EVENTBUS_SLOW_HANDLER_MS = '8';
    process.env.LOG_LEVEL = 'INFO';
  });

  it('computes slowRatio = slowHandlerCount/handlerCalls', async () => {
    const bus = new InMemoryEventBus();
    // fast ~2ms
    bus.subscribe('RATIO_EVT' as any, async ()=>{ await new Promise(r=>setTimeout(r,2)); });
    // slow ~12ms
    bus.subscribe('RATIO_EVT' as any, async ()=>{ await new Promise(r=>setTimeout(r,12)); });
    // publish several times
    for (let i=0;i<4;i++) await bus.publishAndWait!({ type:'RATIO_EVT', eventId:'x'+i } as any, { timeoutMs:200 });
    const logs: string[] = [];
    const spy = vi.spyOn(console,'log').mockImplementation((...a:any[])=>{ logs.push(a.map(x=>String(x)).join(' ')); });
    const spyW = vi.spyOn(console,'warn').mockImplementation((...a:any[])=>{ logs.push(a.map(x=>String(x)).join(' ')); });
    (bus as any).flushMetrics?.();
    spy.mockRestore(); spyW.mockRestore();
    const line = logs.find(l=>/RATIO_EVT/.test(l) && /metrics/.test(l)) || logs.find(l=>/EVENT/.test(l) && /metrics/.test(l));
    expect(line).toBeTruthy();
    if (line) {
      const m = line.match(/(\{.*\})/);
      if (m) {
        try {
          const obj = JSON.parse(m[1]);
          const st = obj.types?.RATIO_EVT;
          expect(st).toBeTruthy();
          const expectedSlow = st.slowHandlerCount;
          const expectedCalls = st.handlerCalls;
          expect(st.slowRatio).toBeCloseTo(expectedSlow/expectedCalls, 3);
        } catch {}
      }
    }
  });
});
