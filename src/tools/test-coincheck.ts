import dotenv from 'dotenv';
dotenv.config();
import { createPrivateApi } from '../api/adapters';
import { logInfo, logWarn } from '../utils/logger';

function sleep(ms:number){ return new Promise(r=>setTimeout(r,ms)); }

(async ()=>{
  process.env.EXCHANGE = 'coincheck';
  const api: any = createPrivateApi();
  logInfo('[TEST] Coincheck start');
  const bal = await api.get_info2();
  logInfo('[TEST] Balance', bal.return?.funds);
  if (process.env.DRY_RUN==='1') { logInfo('[TEST] DRY_RUN: skip order'); return; }
  const pair = 'btc_jpy';
  const rate = Number(process.env.CC_TEST_RATE||'1000000');
  const amount = Number(process.env.CC_TEST_AMOUNT||'0.005');
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
