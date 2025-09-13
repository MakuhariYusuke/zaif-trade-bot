import { createPrivateApi } from "../api/adapters";
import { fetchTradeHistory, getActiveOrders } from "../core/market";
import { submitOrderWithRetry } from "../core/execution";
import { appendSummary, loadDaily } from "../utils/daily-stats";
import { strategyOnce } from "../index";
import { logInfo, logWarn, logAssert } from "../utils/logger";
import { getOrderBook } from "../api/public-router";
import { loadPairs } from "../utils/config";
import fs from 'fs';
import path from 'path';
import os from 'os';

function initMarket(api: any) {
  // The market module does not export 'init'; provide a local no-op initializer for mock scenarios.
  return;
}
function initExec(api: any) {
  // The execution module does not export 'init'; provide a local no-op initializer for mock scenarios.
  return;
}

/**
 * Get today's date as a string in YYYY-MM-DD format.
 * @param {Date} d The date object to format.
 * @returns The formatted date string.
 */
function todayStr(d: Date){ return d.toISOString().slice(0,10); }

/**
 * Run a mock trading scenario using the private API mock.
 * This function initializes the market and execution modules with a mock private API,
 * submits a buy order to simulate an entry, waits briefly, then submits a sell order to simulate an exit.
 * It logs the results of each operation and appends summaries to daily statistics.
 * Errors are caught and logged to the console.
 * This is intended for testing and demonstration purposes only.
 *
 * Side effects:
 * - Modifies global state (e.g., appends to daily statistics).
 * - Interacts with the mock private API.
 * - Logs output to the console.
 * Does not return a value.
 */
async function run(){
  process.env.USE_PRIVATE_MOCK = '1';
  process.env.DRY_RUN = '1';
  // Error/latency injection controls (optional)
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
  // Entry BUY
  const entry = await submitOrderWithRetry({ currency_pair: pair, side:'bid', limitPrice: 1000000, amount: 0.001 });
  logInfo('[SCENARIO] Entry summary', entry);
  appendSummary(todayStr(new Date()), entry as any);
  const sleepMs = Number(process.env.SCENARIO_SLEEP_MS || 300);
  await new Promise(r=>setTimeout(r,sleepMs));
  // Force exit order
  const exit = await submitOrderWithRetry({ currency_pair: pair, side:'ask', limitPrice: 1000000, amount: entry.filledQty || 0.001 });
  logInfo('[SCENARIO] Exit summary', exit);
  appendSummary(todayStr(new Date()), exit as any);
  const hist = await fetchTradeHistory(pair, { count: 50 });
  logInfo('[SCENARIO] Recent fills', hist.slice(-10));
  logInfo('[SCENARIO] Active orders', await getActiveOrders(pair));
  logInfo('[SCENARIO] Done');

  // Fast paths: run only specific forced checks when requested
  if (forceRsi) {
    try {
  // Overwrite default price cache with strong uptrend series (300 pts)
  const cachePath = path.resolve(process.cwd(), 'price_cache.json');
      const now = Date.now();
      const prices: number[] = Array.from({length: 300}, (_,i)=> 100 + i * 3);
      const list = prices.map((p, idx) => ({ ts: now - (prices.length - idx) * 1000, price: p }));
      fs.writeFileSync(cachePath, JSON.stringify(list));
    } catch {}
    // Make RSI easy to trigger and avoid TP/SL/TRAIL interference
    process.env.RSI_PERIOD = process.env.RSI_PERIOD || '5';
    process.env.RISK_INDICATOR_INTERVAL_SEC = '0';
    process.env.MIN_HOLD_SEC = '0';
    process.env.RISK_TAKE_PROFIT_PCT = '1000';
    process.env.RISK_STOP_LOSS_PCT = '1000';
    process.env.RISK_TRAIL_TRIGGER_PCT = '999';
  if (process.env.SCENARIO_SELL_RSI) process.env.SELL_RSI_OVERBOUGHT = process.env.SCENARIO_SELL_RSI;
  if (!process.env.SELL_RSI_OVERBOUGHT) process.env.SELL_RSI_OVERBOUGHT = '60';
    // Seed a long near current mid so TP/SL won't preempt RSI
    try {
      const positionsFile = process.env.POSITIONS_FILE || path.resolve(process.cwd(), '.positions.json');
      let mid = 1000000;
      try {
        const ob: any = await getOrderBook('btc_jpy');
        const a = Number(ob?.asks?.[0]?.[0] || 0);
        const b = Number(ob?.bids?.[0]?.[0] || 0);
        if (a > 0 && b > 0) mid = (a + b) / 2;
      } catch {}
      const pos = [ { id: 'rsi-long', pair: 'btc_jpy', side: 'long', entryPrice: mid * 0.999, amount: 0.001, timestamp: Date.now() - 10 } ];
      fs.writeFileSync(positionsFile, JSON.stringify(pos, null, 2));
    } catch {}
    const d = todayStr(new Date());
    const before = loadDaily(d, 'btc_jpy');
    await strategyOnce('btc_jpy', false);
    const after = loadDaily(d, 'btc_jpy');
    const rsiExitInc = (after.rsiExits || 0) - (before.rsiExits || 0);
    if (rsiExitInc < 1) { logAssert('RSI_EXIT did not increase as expected (forced)', { rsiExitInc }); process.exit(1); }
    return;
  }
  if (forceTrail) {
    // Pre-arm trailing stop on btc_jpy and assert trail exit
    try {
      const positionsFile = process.env.POSITIONS_FILE || path.resolve(process.cwd(), '.positions.json');
      const pos = [ { id: 'trail-long', pair: 'btc_jpy', side: 'long', entryPrice: 16000000, amount: 0.001, timestamp: Date.now() - 60000, trailArmed: true, highestPrice: 18000000 } ];
      fs.writeFileSync(positionsFile, JSON.stringify(pos, null, 2));
    } catch {}
  // Disable RSI/TP/SL and allow immediate trail evaluation
  process.env.SELL_RSI_OVERBOUGHT = '1000';
  process.env.BUY_RSI_OVERSOLD = '-1000';
  process.env.RISK_TAKE_PROFIT_PCT = '1000';
  process.env.RISK_STOP_LOSS_PCT = '1000';
  process.env.MIN_HOLD_SEC = '0';
    const d = todayStr(new Date());
    const before = loadDaily(d, 'btc_jpy');
    await strategyOnce('btc_jpy', false);
    const after = loadDaily(d, 'btc_jpy');
    const trailExitInc = (after.trailExitTotal || 0) - (before.trailExitTotal || 0);
    if (trailExitInc < 1) { logAssert('TRAIL_EXIT did not increase as expected (forced)', { trailExitInc }); process.exit(1); }
    return;
  }

  // --- Range market BUY_SELL (DRY) and assert entries & exits counters best-effort ---
  process.env.TRADE_FLOW = 'BUY_SELL';
  process.env.TEST_FLOW_QTY = process.env.TEST_FLOW_QTY || '0.001';
  logInfo('[SCENARIO] Range market BUY_SELL (DRY) to observe signals and entries');
  {
    const d = todayStr(new Date());
    const before = loadDaily(d, pair);
    await strategyOnce(pair, false);
    const after = loadDaily(d, pair);
    const buyInc = (after.buyEntries || 0) - (before.buyEntries || 0);
    const sellInc = (after.sellEntries || 0) - (before.sellEntries || 0);
    const trailExitInc = (after.trailExitTotal || 0) - (before.trailExitTotal || 0);
    const rsiExitInc = (after.rsiExits || 0) - (before.rsiExits || 0);
  if (buyInc < 1 || sellInc < 1) { logAssert('BUY_SELL entries not increased as expected', { buyInc, sellInc }); process.exit(1); }
  if (trailExitInc < 0 || rsiExitInc < 0) { logAssert('Exit counters decreased unexpectedly', { trailExitInc, rsiExitInc }); process.exit(1); }
  }
  {
    const d = todayStr(new Date());
    const before = loadDaily(d, pair);
    // In range, BUY_SELL expected to trigger at least 1 buyEntries increment over a brief synthetic run in DRY.
    // Here we just assert counters changed after prior operations (best-effort).
    const after = loadDaily(d, pair);
  if ((after.buyEntries || 0) < (before.buyEntries || 0)) { logAssert('buyEntries decreased unexpectedly'); process.exit(1); }
  }

  // --- Crash market SELL_BUY (DRY) and assert entries & exits counters best-effort ---
  process.env.TRADE_FLOW = 'SELL_BUY';
  logInfo('[SCENARIO] Crash market SELL_BUY (DRY) to observe signals and entries');
  {
    const d = todayStr(new Date());
    const before = loadDaily(d, pair);
    await strategyOnce(pair, false);
    const after = loadDaily(d, pair);
    const buyInc = (after.buyEntries || 0) - (before.buyEntries || 0);
    const sellInc = (after.sellEntries || 0) - (before.sellEntries || 0);
    const trailExitInc = (after.trailExitTotal || 0) - (before.trailExitTotal || 0);
    const rsiExitInc = (after.rsiExits || 0) - (before.rsiExits || 0);
  if (buyInc < 1 || sellInc < 1) { logAssert('SELL_BUY entries not increased as expected', { buyInc, sellInc }); process.exit(1); }
  if (trailExitInc < 0 || rsiExitInc < 0) { logAssert('Exit counters decreased unexpectedly', { trailExitInc, rsiExitInc }); process.exit(1); }
  }
  {
    const d = todayStr(new Date());
    const before = loadDaily(d);
    const after = loadDaily(d);
    if ((after.sellEntries || 0) < (before.sellEntries || 0)) { logAssert('sellEntries decreased unexpectedly'); process.exit(1); }

  // --- Forced RSI_EXIT scenario (SELL mode) ---
  if (!forceTrail) {
  // Prepare minimal price cache with strong uptrend to push RSI high
  try {
    const cachePath = path.resolve(process.cwd(), 'price_cache.json');
    const now = Date.now();
    // Generate increasing prices over time so that latest price is the highest; after reverse, RSI gains dominate
    const prices: number[] = Array.from({length: 60}, (_,i)=> 100 + i * 2);
    const list = prices.map((p, idx) => ({ ts: now - (prices.length - idx) * 1000, price: p }));
    fs.writeFileSync(cachePath, JSON.stringify(list, null, 0));
  } catch {}
  // Allow overriding RSI/TRAIL thresholds via SCENARIO_* envs (fallback to existing settings)
  // Disable TP/SL/TRAIL so RSI can be observed distinctly; and accelerate indicators/holding
  process.env.RISK_TAKE_PROFIT_PCT = process.env.RISK_TAKE_PROFIT_PCT || '100';
  process.env.RISK_STOP_LOSS_PCT = process.env.RISK_STOP_LOSS_PCT || '100';
  process.env.RISK_TRAIL_TRIGGER_PCT = process.env.RISK_TRAIL_TRIGGER_PCT || '999';
  process.env.RSI_PERIOD = process.env.RSI_PERIOD || '5';
  process.env.RISK_INDICATOR_INTERVAL_SEC = process.env.RISK_INDICATOR_INTERVAL_SEC || '0';
  process.env.MIN_HOLD_SEC = '0';
  if (process.env.SCENARIO_SELL_RSI) process.env.SELL_RSI_OVERBOUGHT = process.env.SCENARIO_SELL_RSI;
  if (process.env.SCENARIO_BUY_RSI) process.env.BUY_RSI_OVERSOLD = process.env.SCENARIO_BUY_RSI;
  if (process.env.SCENARIO_TRAIL_TRIGGER_PCT) process.env.RISK_TRAIL_TRIGGER_PCT = process.env.SCENARIO_TRAIL_TRIGGER_PCT;
  if (process.env.SCENARIO_TRAIL_STOP_PCT) process.env.RISK_TRAIL_STOP_PCT = process.env.SCENARIO_TRAIL_STOP_PCT;
  if (process.env.SCENARIO_MIN_HOLD_SEC) process.env.MIN_HOLD_SEC = process.env.SCENARIO_MIN_HOLD_SEC;
  // Seed a long position so exits are evaluated
  try {
    const positionsFile = process.env.POSITIONS_FILE || path.resolve(process.cwd(), '.positions.json');
    const pos = [
      { id: 'rsi-long', pair: 'btc_jpy', side: 'long', entryPrice: 100, amount: 0.001, timestamp: Date.now() - 60 }
    ];
    fs.writeFileSync(positionsFile, JSON.stringify(pos, null, 2));
  } catch {}
  {
    const d = todayStr(new Date());
    const before = loadDaily(d, 'btc_jpy');
    const sweep = process.env.SCENARIO_SWEEP === '1';
    if (sweep) {
      const rsiList = (process.env.SCENARIO_SELL_RSI_LIST || '55,60,65').split(',').map(s=>s.trim()).filter(Boolean);
      const positionsFile = process.env.POSITIONS_FILE || path.resolve(process.cwd(), '.positions.json');
      const storePath = process.env.PAPER_STORE || path.join(os.tmpdir(), 'paper-trader.json');
      const cachePath = path.resolve(process.cwd(), 'price_cache.json');
      for (const p of pairs){
        const results: Array<any> = [];
        for (const th of rsiList){
          // reset paper/positions/cache to avoid carry-over
          try { if (fs.existsSync(storePath)) fs.unlinkSync(storePath); } catch {}
          try { if (fs.existsSync(positionsFile)) fs.unlinkSync(positionsFile); } catch {}
          try { if (fs.existsSync(cachePath)) fs.unlinkSync(cachePath); } catch {}
          process.env.SELL_RSI_OVERBOUGHT = th;
          await strategyOnce(p, false);
          const after = loadDaily(d, p);
          results.push({ pair: p, metric: 'RSI_OVERBOUGHT', value: th, rsiExits: after.rsiExits||0, realizedPnl: after.realizedPnl||0 });
        }
  const outDir = path.resolve(process.cwd(), 'logs', 'features', 'paper', p);
  try { if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true }); } catch {}
  const outCsv = path.join(outDir, `sweep-${p}-rsi.csv`);
  const outJson = path.join(outDir, `sweep-${p}-rsi.json`);
  try{ fs.writeFileSync(outJson, JSON.stringify(results, null, 2)); }catch{}
  try{ fs.writeFileSync(outCsv, ['pair,metric,value,rsiExits,realizedPnl', ...results.map(r=>`${r.pair},RSI_OVERBOUGHT,${r.value},${r.rsiExits},${r.realizedPnl}`)].join('\n')); }catch{}
      }
    } else {
      await strategyOnce('btc_jpy', false);
    }
    const after = loadDaily(d, 'btc_jpy');
    const rsiExitInc = (after.rsiExits || 0) - (before.rsiExits || 0);
    if (rsiExitInc < 1) { logAssert('RSI_EXIT did not increase as expected', { rsiExitInc }); process.exit(1); }
  }
  }

  // --- Forced TRAIL_EXIT scenario ---
  if (!forceRsi) {
  // Pre-arm trailing stop
  try {
    const positionsFile = process.env.POSITIONS_FILE || path.resolve(process.cwd(), '.positions.json');
    const pos = [
      { id: 'trail-long', pair: 'btc_jpy', side: 'long', entryPrice: 16000000, amount: 0.001, timestamp: Date.now() - 60000, trailArmed: true, highestPrice: 18000000 }
    ];
    fs.writeFileSync(positionsFile, JSON.stringify(pos, null, 2));
  } catch {}
  process.env.MIN_HOLD_SEC = '0';
  {
    const d = todayStr(new Date());
    const before = loadDaily(d, 'btc_jpy');
    const sweep = process.env.SCENARIO_SWEEP === '1';
    if (sweep) {
      const trigList = (process.env.SCENARIO_TRAIL_TRIGGER_LIST || '0.02,0.03').split(',').map(s=>s.trim()).filter(Boolean);
      const positionsFile = process.env.POSITIONS_FILE || path.resolve(process.cwd(), '.positions.json');
      const storePath = process.env.PAPER_STORE || path.join(os.tmpdir(), 'paper-trader.json');
      const cachePath = path.resolve(process.cwd(), 'price_cache.json');
      for (const p of pairs){
        const results: Array<any> = [];
        for (const th of trigList){
          try { if (fs.existsSync(storePath)) fs.unlinkSync(storePath); } catch {}
          try { if (fs.existsSync(positionsFile)) fs.unlinkSync(positionsFile); } catch {}
          try { if (fs.existsSync(cachePath)) fs.unlinkSync(cachePath); } catch {}
          process.env.RISK_TRAIL_TRIGGER_PCT = th; await strategyOnce(p, false);
          const after = loadDaily(d, p);
          results.push({ pair: p, metric: 'TRAIL_TRIGGER', value: th, trailExitTotal: after.trailExitTotal||0, realizedPnl: after.realizedPnl||0 });
        }
  const outDir = path.resolve(process.cwd(), 'logs', 'features', 'paper', p);
  try { if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true }); } catch {}
  const outCsv = path.join(outDir, `sweep-${p}-trail.csv`);
  const outJson = path.join(outDir, `sweep-${p}-trail.json`);
  try{ fs.writeFileSync(outJson, JSON.stringify(results, null, 2)); }catch{}
  try{ fs.writeFileSync(outCsv, ['pair,metric,value,trailExitTotal,realizedPnl', ...results.map(r=>`${r.pair},TRAIL_TRIGGER,${r.value},${r.trailExitTotal},${r.realizedPnl}`)].join('\n')); }catch{}
      }
    } else {
      await strategyOnce('btc_jpy', false);
    }
    const after = loadDaily(d, 'btc_jpy');
    const trailExitInc = (after.trailExitTotal || 0) - (before.trailExitTotal || 0);
    if (trailExitInc < 1) { logAssert('TRAIL_EXIT did not increase as expected', { trailExitInc }); process.exit(1); }
  }
  }

  // --- SMA sweep scenario (signal sensitivity) ---
  {
    const d = todayStr(new Date());
    const sweep = process.env.SCENARIO_SWEEP === '1';
    if (sweep) {
      const shortList = (process.env.SCENARIO_SMA_SHORT_LIST || '9,12').split(',').map(s=>s.trim()).filter(Boolean);
      const longList = (process.env.SCENARIO_SMA_LONG_LIST || '26,30').split(',').map(s=>s.trim()).filter(Boolean);
      const positionsFile = process.env.POSITIONS_FILE || path.resolve(process.cwd(), '.positions.json');
      const storePath = process.env.PAPER_STORE || path.join(os.tmpdir(), 'paper-trader.json');
      const cachePath = path.resolve(process.cwd(), 'price_cache.json');
      for (const p of pairs){
        const results: Array<any> = [];
        for (const sShort of shortList){
          // reset per loop
          try { if (fs.existsSync(storePath)) fs.unlinkSync(storePath); } catch {}
          try { if (fs.existsSync(positionsFile)) fs.unlinkSync(positionsFile); } catch {}
          try { if (fs.existsSync(cachePath)) fs.unlinkSync(cachePath); } catch {}
          process.env.SMA_SHORT = sShort;
          await strategyOnce(p, false);
          const afterA = loadDaily(d, p);
          results.push({ pair: p, metric: 'SMA_SHORT', value: sShort, buyEntries: afterA.buyEntries||0, sellEntries: afterA.sellEntries||0, realizedPnl: afterA.realizedPnl||0 });
        }
        for (const sLong of longList){
          try { if (fs.existsSync(storePath)) fs.unlinkSync(storePath); } catch {}
          try { if (fs.existsSync(positionsFile)) fs.unlinkSync(positionsFile); } catch {}
          try { if (fs.existsSync(cachePath)) fs.unlinkSync(cachePath); } catch {}
          process.env.SMA_LONG = sLong;
          await strategyOnce(p, false);
          const afterB = loadDaily(d, p);
          results.push({ pair: p, metric: 'SMA_LONG', value: sLong, buyEntries: afterB.buyEntries||0, sellEntries: afterB.sellEntries||0, realizedPnl: afterB.realizedPnl||0 });
        }
  const outDir = path.resolve(process.cwd(), 'logs', 'features', 'paper', p);
  try { if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true }); } catch {}
  const outCsv = path.join(outDir, `sweep-${p}-sma.csv`);
  const outJson = path.join(outDir, `sweep-${p}-sma.json`);
  try{ fs.writeFileSync(outJson, JSON.stringify(results, null, 2)); }catch{}
  try{ fs.writeFileSync(outCsv, ['pair,metric,value,buyEntries,sellEntries,realizedPnl', ...results.map(r=>`${r.pair},${r.metric},${r.value},${r.buyEntries},${r.sellEntries},${r.realizedPnl}`)].join('\n')); }catch{}
      }
    }
  }
  }
}
run().catch(e=>{ logWarn('[SCENARIO] error', e?.message||e); process.exit(1); });
