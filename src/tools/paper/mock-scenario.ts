import { createPrivateApi } from "../../api/adapters";
import { fetchTradeHistory, getActiveOrders, initMarket } from "../../core/market";
import { submitOrderWithRetry, initExecution } from "../../core/execution";
import { appendSummary } from "../../utils/daily-stats";
import { logInfo, logWarn } from "../../utils/logger";
import { loadPairs } from "../../utils/config";
import { todayStr } from "../../utils/toolkit";
import { clampAmountForSafety } from "../../utils/toolkit";

// initialize market/execution modules with the selected private API

async function run(){
  // Always run against mock in paper mode
  process.env.USE_PRIVATE_MOCK = '1';
  process.env.DRY_RUN = '1';
  // Scenario variable mapping
  // - Latency/timeout -> mock exit delay to slow fills
  if (process.env.SCENARIO_PAPER_LATENCY_MS) process.env.MOCK_EXIT_DELAY_MS = process.env.SCENARIO_PAPER_LATENCY_MS;
  process.env.MOCK_EXIT_DELAY_MS = String(Number(process.env.SCENARIO_PAPER_TIMEOUT_MS || 0) || Number(process.env.MOCK_EXIT_DELAY_MS || 0) || 0);
  // - Error rate: probabilistic failure injection wrapper in this script
  const ERROR_RATE = Number(process.env.SCENARIO_PAPER_ERROR_RATE || '0');
  // - Loop count for high-frequency/stress
  const LOOP = Math.max(1, Number(process.env.LOOP || '1'));
  const api = createPrivateApi();
  initMarket(api); initExecution(api as any);
  logInfo('[SCENARIO] Starting MOCK scenario');
  const pairs = loadPairs();
  const pair = pairs[0] || 'btc_jpy';
  const bal = { funds: {} as any };
  logInfo('[SCENARIO] Balance', bal.funds);
  let injectedErrors = 0;
  let otherErrors = 0;
  let entryAmt = 0.001;
  if (process.env.SAFETY_MODE === '1') {
    try {
      const funds = await (createPrivateApi()).get_info2();
      const f = (funds as any)?.return?.funds || {};
      entryAmt = clampAmountForSafety('bid', entryAmt, 1000000, f, pair);
    } catch {}
  }
  async function maybeFail<T>(fn: ()=>Promise<T>): Promise<T> {
    if (ERROR_RATE > 0 && Math.random() < ERROR_RATE) {
      throw new Error('Injected scenario error');
    }
    return await fn();
  }
  // optional pacing only between entry and exit, no other sleeps
  const sleepMs = Math.max(0, Number(process.env.SCENARIO_SLEEP_MS || 0));
  for (let i=0; i<LOOP; i++){
    try {
      const entry = await maybeFail(()=>submitOrderWithRetry({ 
        currency_pair: pair, 
        side:'bid', 
        limitPrice: 1000000, 
        amount: entryAmt }));
      logInfo('[SCENARIO] Entry summary', entry);
  appendSummary(todayStr(), entry as any);
  if (sleepMs > 0) { await new Promise(r=>setTimeout(r, sleepMs)); }
      let exitAmt = (entry as any).filledQty || 0.001;
      if (process.env.SAFETY_MODE === '1') {
        try {
          const funds = await (createPrivateApi()).get_info2();
          const f = (funds as any)?.return?.funds || {};
          exitAmt = clampAmountForSafety('ask', exitAmt, 1000000, f, pair);
        } catch {}
      }
      try {
        const exit = await maybeFail(()=>submitOrderWithRetry({ 
          currency_pair: pair, 
          side:'ask', 
          limitPrice: 1000000, 
          amount: exitAmt 
        }));
        logInfo('[SCENARIO] Exit summary', exit);
        appendSummary(todayStr(), exit as any);
      } catch(e:any){
        const msg = e?.message || String(e);
        if (msg.includes('Injected scenario error')) {
          injectedErrors++;
          logWarn('[SCENARIO] injected error on exit (non-fatal)');
        } else {
          otherErrors++;
          logWarn('[SCENARIO] exit error (non-fatal)', msg);
        }
      }
    } catch(e:any){
      const msg = e?.message || String(e);
      if (msg.includes('Injected scenario error')) {
        injectedErrors++;
        logWarn('[SCENARIO] injected error on entry (non-fatal)');
      } else {
        otherErrors++;
        logWarn('[SCENARIO] entry error (non-fatal)', msg);
      }
      // skip to next loop iteration
      continue;
    }
    if (i % 10 === 0) {
      try {
        const hist = await fetchTradeHistory(pair, { count: 50 });
        logInfo('[SCENARIO] Recent fills', hist.slice(-3));
      } catch(e:any){ logWarn('[SCENARIO] fetch hist warn', e?.message||e); }
      try {
        logInfo('[SCENARIO] Active orders', await getActiveOrders(pair));
      } catch(e:any){ logWarn('[SCENARIO] active orders warn', e?.message||e); }
    }
  }
  logInfo('[SCENARIO] Done (non-fatal summary)', { injectedErrors, otherErrors });
}
run().catch(e=>{ logWarn('[SCENARIO] error (treated as non-fatal)', e?.message||e); process.exit(0); });
