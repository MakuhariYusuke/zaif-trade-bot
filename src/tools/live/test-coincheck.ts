import dotenv from 'dotenv';
dotenv.config();
import { createPrivateApi } from '../../api/adapters';
import { logInfo, logWarn } from '../../utils/logger';
import { sleep, fetchBalances, clampAmountForSafety, baseFromPair } from '../../utils/toolkit';

(async ()=>{
  process.env.EXCHANGE = 'coincheck';
  const api: any = createPrivateApi();
  logInfo('[TEST] Coincheck start');
  const bal = await api.get_info2();
  logInfo('[TEST] Balance', bal.return?.funds);
  if (process.env.DRY_RUN==='1') { logInfo('[TEST] DRY_RUN: skip order'); return; }
  const pair = process.env.PAIR || 'btc_jpy';
  const rate = Number(process.env.CC_TEST_RATE||'1000000');
  let amount = Number(process.env.CC_TEST_AMOUNT||'0.005');
  if (process.env.SAFETY_MODE==='1'){
    try{ const funds = await fetchBalances(api); amount = clampAmountForSafety('bid', amount, rate, funds, pair); }catch{}
  }
  // exposure check
  try {
    const funds = await fetchBalances(api);
    const base = baseFromPair(pair).toLowerCase();
    const jpy = Number((funds as any).jpy||0); const balBase = Number((funds as any)[base]||0);
    logInfo('[EXPOSURE]', { jpy, [base]: balBase });
    const notional = amount * rate;
    if (jpy>0 && notional > jpy*0.05) logWarn(`[WARN][BALANCE] bid notional ${notional} exceeds 5% of JPY ${jpy}`);
  } catch {}
  logInfo('[TEST] Place BUY', { pair, rate, amount });
  const tr = await api.trade({ currency_pair: pair, action: 'bid', price: rate, amount });
  logInfo('[TEST] Placed', tr);
  await sleep(2000);
  const opens = await api.active_orders({ currency_pair: pair });
  logInfo('[TEST] Opens', opens.slice(0,5));
  if (opens.length){
    logInfo('[TEST] Cancel first', opens[0].order_id);
    try { await api.cancel_order({ order_id: String(opens[0].order_id) }); } catch(e:any){ logWarn('[TEST] cancel failed', e?.message||e); }
  }
  const hist = await api.trade_history({ currency_pair: pair, count: 10 });
  logInfo('[TEST] Trades', hist.slice(0,5));
  logInfo('OK');
})();
