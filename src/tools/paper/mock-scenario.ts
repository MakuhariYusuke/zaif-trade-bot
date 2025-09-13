import { createPrivateApi } from "../../api/adapters";
import { fetchTradeHistory, getActiveOrders, initMarket } from "../../core/market";
import { submitOrderWithRetry, initExecution } from "../../core/execution";
import { appendSummary, loadDaily } from "../../utils/daily-stats";
import { strategyOnce } from "../../index";
import { logInfo, logWarn, logAssert } from "../../utils/logger";
import { getOrderBook } from "../../api/public";
import { loadPairs } from "../../utils/config";
import { todayStr } from "../../utils/toolkit";
import { fetchBalances, clampAmountForSafety } from "../../utils/toolkit";
import fs from 'fs';
import path from 'path';
import os from 'os';

// initialize market/execution modules with the selected private API

async function run(){
  process.env.USE_PRIVATE_MOCK = '1';
  process.env.DRY_RUN = '1';
  if (process.env.SCENARIO_PAPER_ERROR_RATE) process.env.PAPER_ERROR_RATE = process.env.SCENARIO_PAPER_ERROR_RATE;
  if (process.env.SCENARIO_PAPER_LATENCY_MS) process.env.PAPER_LATENCY_MS = process.env.SCENARIO_PAPER_LATENCY_MS;
  if (process.env.SCENARIO_PAPER_FILL_MODE) process.env.PAPER_FILL_MODE = process.env.SCENARIO_PAPER_FILL_MODE;
  const api = createPrivateApi();
  initMarket(api); initExecution(api as any);
  logInfo('[SCENARIO] Starting MOCK scenario');
  const pairs = loadPairs();
  const pair = pairs[0] || 'btc_jpy';
  const forceRsi = process.env.SCENARIO_FORCE_RSI === '1';
  const forceTrail = process.env.SCENARIO_FORCE_TRAIL === '1';
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
  const entry = await submitOrderWithRetry({ currency_pair: pair, side:'bid', limitPrice: 1000000, amount: entryAmt });
  logInfo('[SCENARIO] Entry summary', entry);
  appendSummary(todayStr(), entry as any);
  const sleepMs = Number(process.env.SCENARIO_SLEEP_MS || 300);
  await new Promise(r=>setTimeout(r,sleepMs));
  let exitAmt = entry.filledQty || 0.001;
  if (process.env.SAFETY_MODE === '1') {
    try {
      const funds = await (createPrivateApi()).get_info2();
      const f = (funds as any)?.return?.funds || {};
      exitAmt = clampAmountForSafety('ask', exitAmt, 1000000, f, pair);
    } catch {}
  }
  const exit = await submitOrderWithRetry({ currency_pair: pair, side:'ask', limitPrice: 1000000, amount: exitAmt });
  logInfo('[SCENARIO] Exit summary', exit);
  appendSummary(todayStr(), exit as any);
  const hist = await fetchTradeHistory(pair, { count: 50 });
  logInfo('[SCENARIO] Recent fills', hist.slice(-10));
  logInfo('[SCENARIO] Active orders', await getActiveOrders(pair));
  logInfo('[SCENARIO] Done');
}
run().catch(e=>{ logWarn('[SCENARIO] error', e?.message||e); process.exit(1); });
