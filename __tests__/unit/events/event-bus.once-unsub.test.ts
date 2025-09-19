import { describe, it, expect } from 'vitest';
import { InMemoryEventBus } from '../../../src/application/events/bus';

interface DummyEvent { type: 'DUMMY'; value: number; }

describe('EventBus once & unsubscribe', () => {
  it('subscribeOnce only fires once', async () => {
    const bus = new InMemoryEventBus();
    let calls = 0;
    bus.subscribeOnce?.('DUMMY', (ev: any) => { calls++; expect(ev.value).toBe(1); });
    bus.publish({ type: 'DUMMY', value: 1 } as any, { async: false });
    bus.publish({ type: 'DUMMY', value: 2 } as any, { async: false });
    expect(calls).toBe(1);
  });

  it('unsubscribe removes handler', () => {
    const bus = new InMemoryEventBus();
    let calls = 0;
    const off = bus.subscribe('DUMMY' as any, () => { calls++; });
    bus.publish({ type: 'DUMMY', value: 1 } as any, { async: false });
    off();
    bus.publish({ type: 'DUMMY', value: 2 } as any, { async: false });
    expect(calls).toBe(1);
  });
});
