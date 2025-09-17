export { getEventBus, setEventBus, InMemoryEventBus } from './bus';
export * from './types';
export function registerAllSubscribers(){
  try { require('./subscribers/position-subscriber').registerPositionSubscriber(); } catch {}
  try { require('./subscribers/stats-subscriber').registerStatsSubscriber(); } catch {}
  try { require('./subscribers/logger-subscriber').registerLoggerSubscriber(); } catch {}
  try { require('./subscribers/trade-logger-subscriber').registerTradeLoggerSubscriber(); } catch {}
}
