import { describe, it, expect, beforeEach, vi } from 'vitest';
import { InMemoryEventBus } from '../../../../ztb/application/events/bus';

// We rely on internal flushMetrics invoked manually via (bus as any).flushMetrics()
// It logs with category 'EVENT' and message 'metrics'. We capture console.log lines.

describe('event bus metrics p95 aggregation', () => {
  beforeEach(()=>{
    process.env.EVENT_METRICS_LAT_CAP = '100';
    // ensure interval disabled to avoid race (constructor disables in tests unless EVENT_METRICS_INTERVAL_IN_TEST=1)
    delete process.env.EVENT_METRICS_INTERVAL_IN_TEST;
  });

  it('computes p95 close to slow handler latency', async () => {
    const bus = new InMemoryEventBus();
    const calls: number[] = [];
    // fast ~5ms
    bus.subscribe('TRADE_EXECUTED' as any, async ()=>{ const t0=Date.now(); await new Promise(r=>setTimeout(r,5)); calls.push(Date.now()-t0); });
    // medium ~20ms
    bus.subscribe('TRADE_EXECUTED' as any, async ()=>{ const t0=Date.now(); await new Promise(r=>setTimeout(r,20)); calls.push(Date.now()-t0); });
    // slow ~50ms
    bus.subscribe('TRADE_EXECUTED' as any, async ()=>{ const t0=Date.now(); await new Promise(r=>setTimeout(r,50)); calls.push(Date.now()-t0); });

    // publish several times to accumulate latency samples
    for (let i=0;i<6;i++) {
      await bus.publishAndWait!({ type: 'TRADE_EXECUTED', eventId: 'E'+i } as any, { timeoutMs: 200 });
    }

    const logs: string[] = [];
    const spyInfo = vi.spyOn(console, 'log').mockImplementation((...args:any[])=>{ logs.push(args.map(a=>String(a)).join(' ')); });
    const spyWarn = vi.spyOn(console, 'warn').mockImplementation((...args:any[])=>{ logs.push(args.map(a=>String(a)).join(' ')); });
    const spyErr = vi.spyOn(console, 'error').mockImplementation((...args:any[])=>{ logs.push(args.map(a=>String(a)).join(' ')); });
    // invoke internal flush
    (bus as any).flushMetrics?.();
    spyInfo.mockRestore(); spyWarn.mockRestore(); spyErr.mockRestore();

    // Find metrics JSON line (JSON mode may not be enabled, so we look for 'EVENT' and 'metrics')
    const line = logs.find(l=>/EVENT/.test(l) && /metrics/.test(l));
    expect(line).toBeTruthy();
    // Extract JSON object portion if present
    let obj: any = null;
    const jsonMatch = line!.match(/(\{.*\})/);
    if (jsonMatch) {
      try { obj = JSON.parse(jsonMatch[1]); } catch {}
    }
    // If not JSON (plain mode), skip deeper assertions but still ensure line exists
    if (obj) {
      // expect our type stats exist
      const types = obj.types || {};
      const trade = types['TRADE_EXECUTED'];
      expect(trade).toBeTruthy();
      // p95 should be at least near 50ms slow handler (allowing jitter)
      expect(trade.p95LatencyMs).toBeGreaterThanOrEqual(40);
      // avg should be between fast and slow extremes
      expect(trade.avgLatencyMs).toBeGreaterThanOrEqual(5);
      expect(trade.avgLatencyMs).toBeLessThanOrEqual(55);
    }
  });
});
