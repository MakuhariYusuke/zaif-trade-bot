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
  if (process.env.SCENARIO_PAPER_TIMEOUT_MS) process.env.MOCK_EXIT_DELAY_MS = process.env.SCENARIO_PAPER_TIMEOUT_MS;
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
  const sleepMs = Number(process.env.SCENARIO_SLEEP_MS || 50);
  for (let i=0; i<LOOP; i++){
    const entry = await maybeFail(()=>submitOrderWithRetry({ 
      currency_pair: pair, 
      side:'bid', 
      limitPrice: 1000000, 
      amount: entryAmt }));
    logInfo('[SCENARIO] Entry summary', entry);
    appendSummary(todayStr(), entry as any);
    await new Promise(r=>setTimeout(r,sleepMs));
    let exitAmt = (entry as any).filledQty || 0.001;
    if (process.env.SAFETY_MODE === '1') {
      try {
        const funds = await (createPrivateApi()).get_info2();
        const f = (funds as any)?.return?.funds || {};
        exitAmt = clampAmountForSafety('ask', exitAmt, 1000000, f, pair);
      } catch {}
    }
    const exit = await maybeFail(()=>submitOrderWithRetry({ 
      currency_pair: pair, 
      side:'ask', 
      limitPrice: 1000000, 
      amount: exitAmt 
    }));
    logInfo('[SCENARIO] Exit summary', exit);
    appendSummary(todayStr(), exit as any);
    if (i % 10 === 0) {
      const hist = await fetchTradeHistory(pair, { count: 50 });
      logInfo('[SCENARIO] Recent fills', hist.slice(-3));
      logInfo('[SCENARIO] Active orders', await getActiveOrders(pair));
    }
  }
  logInfo('[SCENARIO] Done');
}
run().catch(e=>{ logWarn('[SCENARIO] error', e?.message||e); process.exit(1); });
