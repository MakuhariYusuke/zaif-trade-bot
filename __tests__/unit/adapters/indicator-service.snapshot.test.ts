import { beforeEach, describe, expect, it, vi } from 'vitest';
import { logFeatureSample, stopFeaturesLoggerTimers } from '../../../ztb/utils/features-logger';
import { InMemoryEventBus, setEventBus, getEventBus } from '../../../ztb/application/events/bus';
import { logger } from '../../../ztb/utils/logger';

// Minimal stub for price series provider
process.env.TEST_MODE = '1';

describe('IndicatorService snapshot and event emission', () => {
  let bus: InMemoryEventBus;

  beforeEach(() => {
    bus = new InMemoryEventBus();
    setEventBus(bus);
    stopFeaturesLoggerTimers();
  });

  it('populates snapshot latest fields and emits EVENT/INDICATOR', async () => {
  const events: any[] = [];
  getEventBus().subscribe('EVENT/INDICATOR' as any, (e: any) => { events.push(e); });

  const ts = Date.now();
  // First call will also prime the IndicatorService instance internally
  logFeatureSample({ ts, pair: 'BTC_JPY', side: 'bid', price: 100, qty: 0.01, bestBid: 99.5, bestAsk: 100.5 } as any);
  // one more update to ensure latest snapshot fields are populated
  logFeatureSample({ ts: ts + 1000, pair: 'BTC_JPY', side: 'ask', price: 100.2, qty: 0.01, bestBid: 99.7, bestAsk: 100.7 } as any);
  const snap = events.at(-1)?.snapshot;
    // sanity: snapshot contains some core & new fields (existence/null tolerated for early periods)
    expect(snap).toBeTruthy();
    // core
    expect(Object.prototype.hasOwnProperty.call(snap, 'rsi14')).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(snap, 'macd')).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(snap, 'atr14')).toBe(true);
    // new ones
    expect(Object.prototype.hasOwnProperty.call(snap, 'ha_close')).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(snap, 'viPlus')).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(snap, 'aroonUp')).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(snap, 'tsi')).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(snap, 'kc_upper')).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(snap, 'don_upper')).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(snap, 'choppiness')).toBe(true);

  // volume-based should be null due to missing volume in service context
  expect(snap.mfi).toBeNull();
  expect(snap.obv).toBeNull();

    // event was emitted with compact shape
    expect(events.length).toBeGreaterThanOrEqual(1);
    const evt = events.at(-1);
    expect(evt && evt.type).toBe('EVENT/INDICATOR');
    expect(evt && typeof evt.ts).toBe('number');
    expect(evt && evt.pair).toBe('BTC_JPY');
    expect(evt && typeof evt.snapshot).toBe('object');
  });

  it('logs WARN for MFI/OBV at most once per cycle', () => {
  const logSpy = vi.spyOn(logger as any, 'log').mockImplementation((_level: any, _category: any, _message: any, _meta: any) => {});
    const ts = Date.now();
    logFeatureSample({ ts, pair: 'BTC_JPY', side: 'bid', price: 100, qty: 0.01 } as any);
    logFeatureSample({ ts: ts + 1000, pair: 'BTC_JPY', side: 'ask', price: 100.2, qty: 0.01 } as any);
    const warnCalls = (logSpy.mock.calls as any[])
      .map((args: any[]) => ({ level: args?.[0], category: args?.[1], message: args?.[2], meta: args?.[3] }))
      .filter((c: any) => c.level === 'WARN' && c.category === 'IND' && c.message === 'missing');
    const mfiWarns = warnCalls.filter((c: any) => c.meta?.indicator === 'mfi');
    const obvWarns = warnCalls.filter((c: any) => c.meta?.indicator === 'obv');
    expect(mfiWarns.length).toBeGreaterThanOrEqual(1);
    expect(obvWarns.length).toBeGreaterThanOrEqual(1);
    expect(mfiWarns.length).toBeLessThanOrEqual(2);
    expect(obvWarns.length).toBeLessThanOrEqual(2);
  });
});
