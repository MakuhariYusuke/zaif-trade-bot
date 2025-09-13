import dotenv from 'dotenv';
dotenv.config();
import { createPrivateApi } from '../api/adapters';
import { logInfo, logError } from '../utils/logger';

(async ()=>{
  try {
    const ex = (process.env.EXCHANGE || 'zaif').toLowerCase();
  logInfo(`[EXCHANGE] ${ex}`);
    const api = createPrivateApi();
    const before: any = await (api as any).get_info2();
    if (!before || before.success !== 1) {
      logError('Balance fetch failed', before?.error || 'unknown');
      if (typeof (api as any).healthCheck === 'function') {
        try {
          const h = await (api as any).healthCheck();
          logInfo('[HEALTH]', h);
        } catch(e:any){ logError('[HEALTH] call failed', e?.message||e); }
      }
      process.exitCode = 1;
      return;
    }
    const fundsBefore = before.return?.funds || {};
    logInfo('Funds (before)', fundsBefore);
    logInfo('Open Orders (before)', before.return?.open_orders);

    // Optionally run flow/minimal trade if RUN_FLOW=1
    if (process.env.RUN_FLOW === '1') {
      logInfo('[BALANCE] Running test:min-trade or test:flow is expected to be invoked separately. Skipping inline run.');
    }

    // Fetch after
    const after:any = await (api as any).get_info2();
    const fundsAfter = after.return?.funds || {};
    logInfo('Funds (after)', fundsAfter);
    const dry = process.env.DRY_RUN === '1';
    if (dry) {
      logInfo('Diff', { note: 'DRY_RUN=1 (no change expected)' });
    } else {
      const diff: Record<string, number> = {};
      const keys = new Set([...Object.keys(fundsBefore), ...Object.keys(fundsAfter)]);
      for (const k of keys) diff[k] = Number(fundsAfter[k]||0) - Number(fundsBefore[k]||0);
      logInfo('Diff', diff);
    }
  logInfo('Rights', after.return?.rights);
  } catch(e:any){
    logError('Balance tool error', e?.message || e);
    process.exitCode = 1;
  }
})();
