import dotenv from 'dotenv';
dotenv.config();
import fs from 'fs';
import path from 'path';
import { createPrivateApi } from '../api/adapters';
import { getOrderBook, getTrades } from '../api/public';
import { logInfo, logError, logWarn } from '../utils/logger';
import { incBuyEntry, incSellEntry } from '../utils/daily-stats';
import type { PrivateApi } from '../types/private';
import { logFeatureSample } from '../utils/features-logger';
import { appendPriceSamples, getPriceSeries } from '../utils/price-cache';
import { calculateSma, calculateRsi } from '../core/risk';
import { todayStr, fetchBalances, clampAmountForSafety, baseFromPair, getExposureWarnPct, computeExposureRatio, sleep } from '../utils/toolkit';

type Flow = 'BUY_ONLY' | 'SELL_ONLY' | 'BUY_SELL' | 'SELL_BUY';

export function isDryRun(){ return process.env.DRY_RUN === '1'; }

export function decideOrderType(rateInput: string | undefined){
  const orderType = (process.env.ORDER_TYPE || (process.env as any).orderType || '').toLowerCase();
  const isMarket = orderType ? (orderType === 'market') : (rateInput === '' || rateInput == null);
  const rateOverride = isMarket ? 0 : Number(rateInput || '0');
  return { isMarket, rateOverride };
}

async function placeDry(action: 'bid'|'ask', price: number, qty: number){
  const id = `DRY-${Date.now()}`;
  logInfo(`[MIN-TRADE][DRY] place ${action} id=${id} price=${price} qty=${qty}`);
  logInfo(`[MIN-TRADE][SIMULATED CANCEL] ${id}`);
  await sleep(50);
  return { id, filledQty: 0, avgPrice: price, status: 'cancelled' as const };
}

async function placeAndCancel(api: PrivateApi, pair: string, action: 'bid' | 'ask', price: number, qty: number, opts?: { market?: boolean; refPx?: number }) {
  if (isDryRun()) return placeDry(action, price, qty);
  const market = !!opts?.market;
  const refPx = Number(opts?.refPx || price || 0);
  const params: any = { currency_pair: pair, action };
  if (market) {
      if (action === 'bid') {
          const notional = Math.max(1, Math.round(qty * Math.max(1, refPx)));
          params.market_notional = notional;
      } else {
          params.amount = qty;
      }
  } else {
      params.price = price; params.amount = qty;
  }
  const r: any = await api.trade(params);
  const id = String(r?.return?.order_id || '');
  if (market) {
      let filledQty = 0; let value = 0;
      try {
          const hist: any[] = await api.trade_history({ currency_pair: pair, count: 100 });
          const fills = hist.filter(h => String(h.order_id) === id);
          for (const f of fills) { const a = Number(f.amount || 0); const p = Number(f.price || 0); filledQty += a; value += a * p; }
      } catch {}
      const avgPrice = filledQty > 0 ? value / filledQty : 0;
      logInfo(`[LIVE][MARKET_FILLED] order_id=${id} filledQty=${filledQty} avgPrice=${avgPrice}`);
      return { id, filledQty, avgPrice, status: 'filled' as const };
  }
  try { await api.cancel_order({ order_id: id }); }
  catch (e: any) { logError('[LIVE][CANCEL_FAIL]', e?.message || e); return { id, filledQty: 0, avgPrice: 0, status: 'failed' as const }; }
  let filledQty = 0; let value = 0;
  try {
      const hist: any[] = await api.trade_history({ currency_pair: pair, count: 100 });
      const fills = hist.filter(h => String(h.order_id) === id);
      for (const f of fills) { const a = Number(f.amount || 0); const p = Number(f.price || 0); filledQty += a; value += a * p; }
  } catch {}
  const avgPrice = filledQty > 0 ? value / filledQty : 0;
  logInfo(`[LIVE][CANCELLED] order_id=${id} filledQty=${filledQty} avgPrice=${avgPrice}`);
  return { id, filledQty, avgPrice, status: 'cancelled' as const };
}

async function recordFeatures(featuresPair: string, side: 'bid'|'ask', fillPrice: number, qty: number, status: 'cancelled'|'failed'|'filled', pxBid: number, pxAsk: number, bestBid: number, bestAsk: number, bestBidSize: number, bestAskSize: number, balancesRef: Record<string, number> | null){
  const ts = Date.now();
  try {
      appendPriceSamples([{ ts, price: fillPrice || (side==='bid'? pxBid: pxAsk) }]);
      const rsiPeriod = Number(process.env.RSI_PERIOD || 14);
      const smaShortP = Number(process.env.SMA_SHORT || 9);
      const smaLongP = Number(process.env.SMA_LONG || 26);
      const series = getPriceSeries(Math.max(200, smaLongP+2, rsiPeriod+2));
      const rsi = calculateRsi(series, rsiPeriod);
      const smaS = calculateSma(series, smaShortP) as any;
      const smaL = calculateSma(series, smaLongP) as any;
      let volumeRecent = 0;
      try {
          const trades = await getTrades(featuresPair.split('_').slice(1).join('_'));
          const nowSec = Math.floor(Date.now()/1000);
          for (const t of (trades||[])){
              const dt = Number((t as any).date || (t as any).created_at || 0);
              if (dt && nowSec - dt <= 60){ volumeRecent += Number((t as any).amount || 0); }
          }
      } catch {}
      const funds = balancesRef || {};
      logFeatureSample({ ts, pair: featuresPair, side, rsi: rsi ?? undefined, sma_short: smaS ?? undefined, sma_long: smaL ?? undefined, price: fillPrice || (side==='bid'? pxBid: pxAsk), qty, pnl: 0, win: null as any, balance: { jpy: (funds as any).jpy, btc: (funds as any).btc, eth: (funds as any).eth, xrp: (funds as any).xrp }, bestBid, bestAsk, status, depthBid: bestBidSize, depthAsk: bestAskSize, volumeRecent });
  } catch (e) {
      logError('[FEATURES_LOG_ERROR]', e instanceof Error ? e.message : String(e));
  }
}

function warnIfOverPct(pair: string, side:'bid'|'ask', qty:number, price:number, balancesBefore: Record<string, number>){
  try{
      const pct = getExposureWarnPct();
      const ratio = computeExposureRatio(side, qty, price, balancesBefore as any, pair);
      if (ratio > pct) {
          if (side==='bid') logWarn(`[WARN][BALANCE] bid notional exceeds ${(pct*100).toFixed(1)}% of JPY (ratio ${(ratio*100).toFixed(1)}%)`);
          else {
              const base = baseFromPair(pair).toUpperCase();
              logWarn(`[WARN][BALANCE] ask qty exceeds ${(pct*100).toFixed(1)}% of ${base} (ratio ${(ratio*100).toFixed(1)}%)`);
          }
      }
  } catch {}
}

export async function runLiveMinimal(){
  const ex = (process.env.EXCHANGE || 'zaif').toLowerCase();
  const pair = process.env.PAIR || 'btc_jpy';
  const flow = (process.env.TRADE_FLOW || 'BUY_ONLY') as Flow;
  const qtyRaw = Number(process.env.TEST_FLOW_QTY || (isDryRun() ? '0.002' : '0'));
  const rateInput = process.env.TEST_FLOW_RATE;
  const { isMarket, rateOverride } = decideOrderType(rateInput);
  if (!isDryRun() && !(qtyRaw > 0)) { console.error('TEST_FLOW_QTY must be > 0'); process.exit(1); }

  const api = createPrivateApi();
  if (!process.env.FEATURES_SOURCE) process.env.FEATURES_SOURCE = 'live';

  // Order book
  let ob: any;
  try {
    ob = await getOrderBook(pair);
    if (!ob || !Array.isArray(ob.bids) || !Array.isArray(ob.asks)) throw new Error('Order book is invalid or undefined');
  } catch (e:any){ logWarn(`[MIN-TRADE] Failed to fetch order book: ${e?.message||e}`); process.exit(1); }
  const bestBid = Number((ob?.bids?.[0]?.[0]) || 0);
  const bestBidSize = Number((ob?.bids?.[0]?.[1]) || 0);
  const bestAsk = Number((ob?.asks?.[0]?.[0]) || 0);
  const bestAskSize = Number((ob?.asks?.[0]?.[1]) || 0);
  if (!isMarket && !(rateOverride > 0)) { console.error('When ORDER_TYPE=limit, TEST_FLOW_RATE must be > 0'); process.exit(1); }
  const pxBid = (!isMarket && rateOverride > 0) ? rateOverride : (bestBid > 0 ? bestBid : Math.max(1, bestAsk * 0.999));
  const pxAsk = (!isMarket && rateOverride > 0) ? rateOverride : (bestAsk > 0 ? bestAsk : Math.max(1, bestBid * 1.001));

  const funds = await fetchBalances(api).catch(()=>({} as any));
  const balancesBefore = { jpy: Number((funds as any).jpy||0), btc: Number((funds as any).btc||0), eth: Number((funds as any).eth||0), xrp: Number((funds as any).xrp||0) };
  const featuresPair = `${ex}_${pair}`;

  const executed: Array<{ side:'bid'|'ask'; qty:number; price:number; status:string }> = [];

  async function runBid() {
      let qty = (process.env.SAFETY_MODE==='1') ? clampAmountForSafety('bid', qtyRaw, pxBid, funds, pair) : qtyRaw;
      if (!isDryRun() && !(qty > 0)) throw new Error('clamped qty <= 0');
      warnIfOverPct(pair, 'bid', qty, pxBid, balancesBefore);
      const r = await placeAndCancel(api, pair, 'bid', pxBid, qty, { market: isMarket, refPx: pxAsk });
      if (!isDryRun()) incBuyEntry(todayStr(), pair);
      executed.push({ side:'bid', qty, price:pxBid, status: (r as any).status || 'cancelled' });
      try { await recordFeatures(featuresPair, 'bid', (r as any).avgPrice || pxBid, qty, (r as any).status || 'cancelled', pxBid, pxAsk, bestBid, bestAsk, bestBidSize, bestAskSize, funds); } catch {}
      return r;
  }
  async function runAsk() {
      let qty = (process.env.SAFETY_MODE==='1') ? clampAmountForSafety('ask', qtyRaw, pxAsk, funds, pair) : qtyRaw;
      if (!isDryRun() && !(qty > 0)) throw new Error('clamped qty <= 0');
      warnIfOverPct(pair, 'ask', qty, pxAsk, balancesBefore);
      const r = await placeAndCancel(api, pair, 'ask', pxAsk, qty, { market: isMarket, refPx: pxBid });
      if (!isDryRun()) incSellEntry(todayStr(), pair);
      executed.push({ side:'ask', qty, price:pxAsk, status: (r as any).status || 'cancelled' });
      try { await recordFeatures(featuresPair, 'ask', (r as any).avgPrice || pxAsk, qty, (r as any).status || 'cancelled', pxBid, pxAsk, bestBid, bestAsk, bestBidSize, bestAskSize, funds); } catch {}
      return r;
  }

  if (flow === 'BUY_ONLY') await runBid();
  else if (flow === 'SELL_ONLY') await runAsk();
  else if (flow === 'BUY_SELL') { await runBid(); await runAsk(); }
  else if (flow === 'SELL_BUY') { await runAsk(); await runBid(); }

  // stats diff archive (best-effort)
  try {
      const { spawnSync } = await import('child_process');
      const npx = process.platform === 'win32' ? 'npx.cmd' : 'npx';
      const r = spawnSync(npx, ['ts-node', 'src/tools/stats/stats-today.ts', '--diff'], { encoding: 'utf8' });
      const out = (r.stdout || '').trim();
      const outDir = path.resolve(process.cwd(), 'logs', 'live');
      if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(outDir, 'stats-diff-live.json'), out || '{}');
  } catch (e){ logError('[ARCHIVE_STATS_DIFF_ERROR]', e instanceof Error ? e.message : String(e)); }

  // write live summary
  try {
      const date = todayStr();
      const outDir = path.resolve(process.cwd(), 'logs', 'live');
      if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
      const diffPath = path.join(outDir, 'stats-diff-live.json');
      const after = await fetchBalances(api).catch(()=>({} as any));
      const balancesAfter = { jpy: Number((after as any).jpy||0), btc: Number((after as any).btc||0), eth: Number((after as any).eth||0), xrp: Number((after as any).xrp||0) };
      const deltas = { jpy: (balancesAfter.jpy - balancesBefore.jpy), btc: (balancesAfter.btc - balancesBefore.btc), eth: (balancesAfter.eth - balancesBefore.eth), xrp: (balancesAfter.xrp - balancesBefore.xrp) };
      const warnings: string[] = [];
      const pct = getExposureWarnPct();
      for (const exed of executed){
          const r = computeExposureRatio(exed.side, exed.qty, exed.price, balancesBefore as any, pair);
          if (r > pct) warnings.push(exed.side==='bid' ? 'over5pct_jpy' : 'over5pct_base');
      }
      let summary: any = { env: { EXCHANGE: process.env.EXCHANGE, TRADE_FLOW: process.env.TRADE_FLOW, TEST_FLOW_QTY: process.env.TEST_FLOW_QTY, TEST_FLOW_RATE: process.env.TEST_FLOW_RATE, DRY_RUN: process.env.DRY_RUN }, balancesBefore, balancesAfter, deltas, executed, warnings };
      if (fs.existsSync(diffPath)){
          try {
              const d = JSON.parse(fs.readFileSync(diffPath,'utf8'));
              const vals = d?.values || {};
              const diff = d?.diff || {};
              summary.stats = { incBuy: (diff.buyEntries||0), incSell: (diff.sellEntries||0), incPnl: (diff.realizedPnl||0), winRate: (vals.trades? (vals.wins||0)/(vals.trades||1): 0), streakWin: vals.streakWin||0, streakLoss: vals.streakLoss||0 };
          } catch {}
      }
      const sumPath = path.join(outDir, `summary-${date}.json`);
      fs.writeFileSync(sumPath, JSON.stringify(summary, null, 2));
  } catch (e){ logError('[SUMMARY_WRITE_ERROR]', e instanceof Error ? e.message : String(e)); }
}

if (require.main === module){
  runLiveMinimal().then(()=>{ logInfo('[MIN-TRADE] done'); }).catch(e=>{ logError('[MIN-TRADE][ERROR]', e?.message||e); process.exit(1); });
}
