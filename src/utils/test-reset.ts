import { resetConfigCache } from './config';
import { resetPriceCache } from './price-cache';
import { resetDailyStatsCache } from './daily-stats';
import fs from 'fs';
import path from 'path';

/**
 * Reset global side-effects between tests.
 * - Clears module cache (optional via caller)
 * - Restores env (callers pass a snapshot)
 * - Clears timers
 * - Resets known in-memory caches (config/price-cache/daily-stats)
 */
export function resetTestState(opts?: {
  envSnapshot?: NodeJS.ProcessEnv;
  restoreEnv?: boolean;
  clearTimers?: boolean;
  resetModules?: boolean;
}){
  const o = opts || {};
  try { resetConfigCache(); } catch {}
  try { resetPriceCache(); } catch {}
  try { resetDailyStatsCache(); } catch {}
  if (o.restoreEnv && o.envSnapshot){
    // Replace process.env wholesale to avoid residuals
    const snap = o.envSnapshot;
    // Remove keys not in snapshot
    for (const k of Object.keys(process.env)){
      if (!(k in snap)) delete (process.env as any)[k];
    }
    // Assign snapshot keys
    for (const [k, v] of Object.entries(snap)){
      (process.env as any)[k] = v as any;
    }
  }
  if (o.clearTimers){
    try { const g = globalThis as any; g.clearImmediate?.(); } catch {}
    // Best-effort: there's no official way to enumerate timers; rely on libs exposing reset hooks.
  }
  if (o.resetModules){
    try {
      // vitest provides vi.resetModules(), but we don't import it here.
      // Caller should invoke vi.resetModules() alongside this helper.
    } catch {}
  }
}

export default resetTestState;

// --- temp dirs cleanup ---
const registeredTempDirs = new Set<string>();

/** Register a temp directory to be auto-removed in global afterEach. */
export function registerTempDir(dir: string){
  try { registeredTempDirs.add(path.resolve(dir)); } catch {}
}

/** Remove all registered temp directories. Called automatically from setup afterEach. */
export function cleanupRegisteredTempDirs(){
  for (const d of Array.from(registeredTempDirs)){
    try { fs.rmSync(d, { recursive: true, force: true }); } catch {}
    registeredTempDirs.delete(d);
  }
}
