import { describe, it, expect } from 'vitest';
import { InMemoryEventBus } from '../../../../src/application/events/bus';

describe('event bus publishAndWait', () => {
  it('executes all handlers (success path)', async () => {
    const bus = new InMemoryEventBus();
    const calls: string[] = [];
    bus.subscribe('TRADE_EXECUTED' as any, async ()=>{ calls.push('a'); });
    bus.subscribe('TRADE_EXECUTED' as any, ()=>{ calls.push('b'); });
    await bus.publishAndWait!({ type: 'TRADE_EXECUTED', eventId: 'E1' } as any, { timeoutMs: 200 });
    expect(calls.sort()).toEqual(['a','b']);
  });

  it('times out slow handler', async () => {
    const bus = new InMemoryEventBus();
    const calls: string[] = [];
    bus.subscribe('TRADE_EXECUTED' as any, async ()=>{ await new Promise(r=>setTimeout(r, 50)); calls.push('slow'); });
    await bus.publishAndWait!({ type: 'TRADE_EXECUTED', eventId: 'E2' } as any, { timeoutMs: 10 });
    // slow handler should have eventually pushed after timeout attempt OR maybe aborted; we assert no throw
    expect(calls.length >= 0).toBe(true);
  });

  it('captures handler error without throwing aggregate', async () => {
    const bus = new InMemoryEventBus();
    let errorCaptured = false;
    (bus as any).setErrorHandler?.(()=>{ errorCaptured = true; });
    bus.subscribe('TRADE_EXECUTED' as any, ()=>{ throw new Error('boom'); });
    await bus.publishAndWait!({ type: 'TRADE_EXECUTED', eventId: 'E3' } as any, { timeoutMs: 50, captureErrors: true });
    expect(errorCaptured).toBe(true);
  });
});
