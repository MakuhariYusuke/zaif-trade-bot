import { beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest';
import resetTestState, { cleanupRegisteredTempDirs } from '../utils/test-reset';
import { installLeakDetector, assertNoLeaksAndCleanup, uninstallLeakDetector } from './leak-detector';

let envSnapshot: NodeJS.ProcessEnv;

beforeAll(() => { try { installLeakDetector(); } catch {} });

beforeEach(() => {
  envSnapshot = { ...process.env };
});

afterEach(async () => {
  try { vi.restoreAllMocks(); } catch {}
  try { vi.useRealTimers(); } catch {}
  try { vi.clearAllTimers(); } catch {}
  try { await vi.resetModules(); } catch {}
  try { resetTestState({ envSnapshot, restoreEnv: true }); } catch {}
  try { cleanupRegisteredTempDirs(); } catch {}
  try { assertNoLeaksAndCleanup(); } catch (e) { throw e; }
});

afterAll(() => { try { uninstallLeakDetector(); } catch {} });
