import dotenv from 'dotenv';
dotenv.config();
import { createPrivateApi } from '../api/adapters';
import { logInfo, logError, logWarn } from '../utils/logger';
import { baseFromPair, fetchBalances } from '../utils/toolkit';

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
    // exposure quick check for selected PAIR if provided
    const pair = process.env.PAIR || 'btc_jpy';
    const base = baseFromPair(pair).toLowerCase();
    const jpy = Number((fundsAfter as any).jpy || 0);
    const balBase = Number((fundsAfter as any)[base] || 0);
    if (jpy >= 0 && balBase >= 0) {
      const pxHint = Number(process.env.PRICE_HINT || 0);
      const notional = pxHint > 0 ? balBase * pxHint : undefined;
      logInfo('[EXPOSURE]', { base: base.toUpperCase(), balBase, jpy, notional: notional ?? '(set PRICE_HINT to compute)' });
      if (pxHint > 0 && jpy > 0) {
        const ratio = (balBase * pxHint) / jpy;
        if (ratio > 0.1) logWarn(`[EXPOSURE] base notional ~${(ratio*100).toFixed(1)}% of JPY (threshold 10%)`);
      }
    }
  } catch(e:any){
    logError('Balance tool error', e?.message || e);
    process.exitCode = 1;
  }
})();
