import type { AppEvent } from './types';
import { log as logCat } from '../../utils/logger';

export type EventHandler<T extends AppEvent = AppEvent> = (event: T) => void | Promise<void>;

export interface EventBus {
  publish<K extends AppEvent['type']>(event: Extract<AppEvent, { type: K }>, opts?: { async?: boolean }): void;
  publishAndWait?<K extends AppEvent['type']>(event: Extract<AppEvent, { type: K }>, opts?: { timeoutMs?: number; captureErrors?: boolean }): Promise<void>;
  subscribe<K extends AppEvent['type']>(type: K, handler: EventHandler<Extract<AppEvent, { type: K }>>): () => void;
  subscribeOnce?<T extends AppEvent>(type: T['type'], handler: EventHandler<T>): () => void;
  clear(): void;
  has(type: string): boolean;
  flush?(): Promise<void>;
  setErrorHandler?(h: (error: any, event: AppEvent, handler: EventHandler) => void): void;
  stop?(): void;
}

export class InMemoryEventBus implements EventBus {
  private handlers: Map<string, Set<EventHandler<any>>> = new Map();
  private onError?: (error: any, event: AppEvent, handler: EventHandler) => void;
  // Metrics state
  private handlerIds = new WeakMap<EventHandler, string>();
  private nextId = 1;
  private metrics = {
    start: Date.now(),
    byType: new Map<string, {
      publishes: number;
      handlerCalls: number;
      errors: number;
      sumLatency: number;
      lats: number[];
      slowHandlerCount: number;
      byHandler: Map<string, { name: string; calls: number; errors: number; sumLatency: number; lats: number[] }>
    }>(),
  };
  private latCap = (() => {
    const raw = process.env.EVENT_METRICS_LAT_CAP;
    const parsed = Number(raw);
    const value = !raw || isNaN(parsed) ? 300 : parsed;
    return Math.max(50, value);
  })();
  private intervalMs = Math.max(0, Number(process.env.EVENT_METRICS_INTERVAL_MS || 60000));
  private timer: NodeJS.Timeout | null = null;
  constructor() {
    const isTest = (process.env.TEST_MODE === '1') || !!process.env.VITEST_WORKER_ID;
    if (isTest && process.env.EVENT_METRICS_INTERVAL_IN_TEST !== '1') {
      this.timer = setInterval(() => {
        try { this.flushMetrics(); } catch { /* ignore */ }
      }, this.intervalMs);
      // Only call .unref() if available (Node.js)
      if (typeof (this.timer as any).unref === 'function') {
        (this.timer as any).unref();
      }
    }
  }
  private stopTimer() { if (this.timer) { clearInterval(this.timer); this.timer = null; } }
  private getHandlerId(h: EventHandler): string {
    let id = this.handlerIds.get(h);
    if (!id) { id = `h${this.nextId++}:${(h as any).name || 'anon'}`; this.handlerIds.set(h, id); }
    return id;
  }
  private recPublish(type: string) {
    const s = this.metrics.byType.get(type) || { publishes: 0, handlerCalls: 0, errors: 0, sumLatency: 0, lats: [] as number[], slowHandlerCount: 0, byHandler: new Map<string, { name: string; calls: number; errors: number; sumLatency: number; lats: number[] }>() };
    s.publishes++;
    this.metrics.byType.set(type, s);
  }
  private recCall(type: string, h: EventHandler, ok: boolean, latencyMs: number) {
    const s = this.metrics.byType.get(type) || { publishes: 0, handlerCalls: 0, errors: 0, sumLatency: 0, lats: [] as number[], slowHandlerCount: 0, byHandler: new Map<string, { name: string; calls: number; errors: number; sumLatency: number; lats: number[] }>() };
    s.handlerCalls++;
    if (!ok) s.errors++;
    s.sumLatency += latencyMs;
    if (s.lats.length < this.latCap) s.lats.push(latencyMs);
    const slowThreshold = Number(process.env.EVENTBUS_SLOW_HANDLER_MS || 100);
    if (latencyMs >= slowThreshold) {
      s.slowHandlerCount++;
      try { logCat('WARN', 'EVENT', 'slow-handler', { type, latencyMs, handler: (h as any).name || 'anon', threshold: slowThreshold }); } catch { }
    }
    const id = this.getHandlerId(h);
    const name = (h as any).name || 'anon';
    const hs = s.byHandler.get(id) || { name, calls: 0, errors: 0, sumLatency: 0, lats: [] as number[] };
    hs.calls++;
    if (!ok) hs.errors++;
    hs.sumLatency += latencyMs;
    if (hs.lats.length < this.latCap) hs.lats.push(latencyMs);
    s.byHandler.set(id, hs);
    this.metrics.byType.set(type, s);
  }
  private p95(arr: number[]): number {
    if (!arr.length) return 0;
    const s = arr.slice().sort((a, b) => a - b);
    const idx = Math.floor(0.95 * (s.length - 1));
    return s[Math.max(0, idx)];
  }
  private flushMetrics() {
    const windowMs = Date.now() - this.metrics.start;
    const out: any = { windowMs, types: {} as any };
    for (const [type, s] of this.metrics.byType.entries()) {
      const avg = s.handlerCalls ? s.sumLatency / s.handlerCalls : 0;
      const p95 = this.p95(s.lats);
      const byHandler: any = {};
      for (const [id, hs] of s.byHandler.entries()) {
        byHandler[id] = {
          name: hs.name,
          calls: hs.calls,
          errors: hs.errors,
          avgLatencyMs: hs.calls ? hs.sumLatency / hs.calls : 0,
          p95LatencyMs: this.p95(hs.lats),
        };
      }
      out.types[type] = {
        publishes: s.publishes,
        handlerCalls: s.handlerCalls,
        errors: s.errors,
        avgLatencyMs: Math.round(avg),
        p95LatencyMs: Math.round(p95),
        slowHandlerCount: s.slowHandlerCount,
        slowRatio: s.handlerCalls ? Number((s.slowHandlerCount / s.handlerCalls).toFixed(3)) : 0,
        handlers: byHandler,
      };
    }
    logCat('INFO', 'EVENT', 'metrics', out);
    // reset window
    this.metrics = { start: Date.now(), byType: new Map() } as any;
  }
  private invoke(handler: EventHandler, event: AppEvent) {
    const t0 = Date.now();
    try {
      const r = handler(event);
      if (r && typeof (r as any).then === 'function') {
        (r as Promise<void>)
          .then(() => { this.recCall(event.type, handler, true, Date.now() - t0); })
          .catch((err) => { this.recCall(event.type, handler, false, Date.now() - t0); this.onError?.(err, event, handler); });
      } else {
        this.recCall(event.type, handler, true, Date.now() - t0);
      }
    } catch (err) {
      this.recCall(event.type, handler, false, Date.now() - t0);
      this.onError?.(err, event, handler);
    }
  }
  has(type: string): boolean { const set = this.handlers.get(type); return !!(set && set.size > 0); }
  publish<K extends AppEvent['type']>(evt: Extract<AppEvent, { type: K }>, opts?: { async?: boolean }): void {
    if (!(evt as any).eventId) {
      (evt as any).eventId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    }
    const set = this.handlers.get(evt.type);
    if (!set || set.size === 0) return;
    const handlers = Array.from(set);
    this.recPublish(evt.type);
    const call = () => {
      if (!handlers) return;
      for (const h of handlers) this.invoke(h, evt);
    };
    const async_ = opts?.async !== false; // default async
    if (!async_) { call(); return; }
    // Prefer queueMicrotask for microtask scheduling; fallback to setTimeout if unavailable (e.g., older environments)
    try { queueMicrotask(call); } catch { setTimeout(call, 0); }
  }
  subscribe<K extends AppEvent['type']>(type: K, handler: EventHandler<Extract<AppEvent, { type: K }>>): () => void {
    if (!this.handlers.has(type)) this.handlers.set(type, new Set());
    const set = this.handlers.get(type)!; set.add(handler);
    return () => { set.delete(handler); };
  }
  subscribeOnce<T extends AppEvent>(type: T['type'], handler: EventHandler<T>): () => void {
    if (!this.handlers.has(type)) this.handlers.set(type, new Set());
    const set = this.handlers.get(type)!;
    const wrapper: EventHandler = (ev: any) => {
      try { handler(ev); } finally { set.delete(wrapper); }
    };
    set.add(wrapper);
    return () => { set.delete(wrapper); };
  }
  async publishAndWait<K extends AppEvent['type']>(event: Extract<AppEvent, { type: K }>, opts?: { timeoutMs?: number; captureErrors?: boolean }): Promise<void> {
    if (!(event as any).eventId) {
      (event as any).eventId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    }
    const set = this.handlers.get(event.type);
    if (!set || set.size === 0) return;
    const handlers = Array.from(set);
    this.recPublish(event.type);
    const timeoutMs = Math.max(0, Number(opts?.timeoutMs ?? (process.env.EVENTBUS_HANDLER_TIMEOUT_MS || 2000)));
    const tasks = handlers.map((h) => {
      const t0 = Date.now();
      const run = Promise.resolve().then(() => h(event)).then(() => {
        this.recCall(event.type, h, true, Date.now() - t0);
      }).catch((err) => {
        this.recCall(event.type, h, false, Date.now() - t0);
        if (opts?.captureErrors !== false) this.onError?.(err, event, h);
        if (opts?.captureErrors === false) throw err;
      });
      if (!timeoutMs) {
        return run.catch((err) => { if (opts?.captureErrors !== false) this.onError?.(err, event, h); });
      }
      return Promise.race([
        run.catch((err) => { if (opts?.captureErrors !== false) this.onError?.(err, event, h); }),
        new Promise<void>((_res, rej) => { const t = setTimeout(() => rej(new Error('Event handler timeout')), timeoutMs); run.finally(() => clearTimeout(t)); })
      ]).catch((err) => { if (opts?.captureErrors !== false) this.onError?.(err, event, h); /* swallow for aggregate */ });
    });
    await Promise.all(tasks);
  }
  async flush(): Promise<void> { await new Promise((res) => setTimeout(res, 0)); }
  /**
   * Removes all event handlers from the bus.
   * Does NOT stop the metrics timer; subscribing again will reuse the existing timer.
   */
  clear() { this.handlers.clear(); }

  /**
   * Stops the metrics timer.
   * After calling stop(), metrics will no longer be collected until a new EventBus is created.
   */
  stop() { this.stopTimer(); }
}

let _bus: EventBus | undefined;
export function getEventBus(): EventBus {
  if (!_bus) _bus = new InMemoryEventBus();
  return _bus;
}
export function setEventBus(bus: EventBus) {
  const prev = _bus;
  if (prev && prev instanceof InMemoryEventBus) {
    try { prev.stop?.(); } catch { }
  }
  _bus = bus;
}
/**
 * Sets a global error handler for the EventBus.
 * This handler will be called whenever an event handler throws an error or returns a rejected promise.
 * Only works with the default InMemoryEventBus implementation.
 */
export function setEventBusErrorHandler(handler: EventBusErrorHandler) {
  const bus = getEventBus();
  if (bus instanceof InMemoryEventBus) bus.setErrorHandler(handler);
}
  }
  _bus = bus;
}
/**
 * Sets a global error handler for the EventBus.
 * This handler will be called whenever an event handler throws an error or returns a rejected promise.
 * Only works with the default InMemoryEventBus implementation.
 */
export function setEventBusErrorHandler(handler: EventBusErrorHandler) {
  const bus = getEventBus();
  if (bus instanceof InMemoryEventBus) bus.setErrorHandler(handler);
}
