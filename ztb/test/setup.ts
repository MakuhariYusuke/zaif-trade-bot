import { beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest';
import resetTestState, { cleanupRegisteredTempDirs } from '../utils/test-reset';
import { installLeakDetector, assertNoLeaksAndCleanup, uninstallLeakDetector } from './leak-detector';
import { stopFeaturesLoggerTimers } from '../utils/features-logger';
import { getEventBus } from '../application/events/bus';

let envSnapshot: NodeJS.ProcessEnv;

beforeAll(() => { try { installLeakDetector(); } catch {} });

beforeEach(() => {
  envSnapshot = { ...process.env };
  // Disable background event-metrics interval during tests unless explicitly enabled
  if (process.env.EVENT_METRICS_INTERVAL_IN_TEST !== '1') {
    process.env.EVENT_METRICS_INTERVAL_MS = '0';
  }
});

afterEach(async () => {
  try { vi.restoreAllMocks(); } catch {}
  try { vi.useRealTimers(); } catch {}
  try { vi.clearAllTimers(); } catch {}
  try { stopFeaturesLoggerTimers(); } catch {}
  try { (getEventBus() as any).stop?.(); } catch {}
  try { (getEventBus() as any).clear?.(); } catch {}
  try { await vi.resetModules(); } catch {}
  try { resetTestState({ envSnapshot, restoreEnv: true }); } catch {}
  try { cleanupRegisteredTempDirs(); } catch {}
  try { assertNoLeaksAndCleanup(); } catch (e) { throw e; }
});

afterAll(() => { try { uninstallLeakDetector(); } catch {} });
