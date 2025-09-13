import { createPrivateApi } from "../../api/adapters";
import { fetchTradeHistory, getActiveOrders } from "../../core/market";
import { submitOrderWithRetry } from "../../core/execution";
import { appendSummary, loadDaily } from "../../utils/daily-stats";
import { strategyOnce } from "../../index";
import { logInfo, logWarn, logAssert } from "../../utils/logger";
import { getOrderBook } from "../../api/public";
import { loadPairs } from "../../utils/config";
import fs from 'fs';
import path from 'path';
import os from 'os';

function initMarket(api: any) { return; }
function initExec(api: any) { return; }

function todayStr(d: Date){ return d.toISOString().slice(0,10); }

async function run(){
  process.env.USE_PRIVATE_MOCK = '1';
  process.env.DRY_RUN = '1';
  if (process.env.SCENARIO_PAPER_ERROR_RATE) process.env.PAPER_ERROR_RATE = process.env.SCENARIO_PAPER_ERROR_RATE;
  if (process.env.SCENARIO_PAPER_LATENCY_MS) process.env.PAPER_LATENCY_MS = process.env.SCENARIO_PAPER_LATENCY_MS;
  if (process.env.SCENARIO_PAPER_FILL_MODE) process.env.PAPER_FILL_MODE = process.env.SCENARIO_PAPER_FILL_MODE;
  const api = createPrivateApi();
  initMarket(api); initExec(api);
  logInfo('[SCENARIO] Starting MOCK scenario');
  const pairs = loadPairs();
  const pair = pairs[0] || 'btc_jpy';
  const forceRsi = process.env.SCENARIO_FORCE_RSI === '1';
  const forceTrail = process.env.SCENARIO_FORCE_TRAIL === '1';
  const bal = { funds: {} as any };
  logInfo('[SCENARIO] Balance', bal.funds);
  const entry = await submitOrderWithRetry({ currency_pair: pair, side:'bid', limitPrice: 1000000, amount: 0.001 });
  logInfo('[SCENARIO] Entry summary', entry);
  appendSummary(todayStr(new Date()), entry as any);
  const sleepMs = Number(process.env.SCENARIO_SLEEP_MS || 300);
  await new Promise(r=>setTimeout(r,sleepMs));
  const exit = await submitOrderWithRetry({ currency_pair: pair, side:'ask', limitPrice: 1000000, amount: entry.filledQty || 0.001 });
  logInfo('[SCENARIO] Exit summary', exit);
  appendSummary(todayStr(new Date()), exit as any);
  const hist = await fetchTradeHistory(pair, { count: 50 });
  logInfo('[SCENARIO] Recent fills', hist.slice(-10));
  logInfo('[SCENARIO] Active orders', await getActiveOrders(pair));
  logInfo('[SCENARIO] Done');
}
run().catch(e=>{ logWarn('[SCENARIO] error', e?.message||e); process.exit(1); });
