import { createPrivateApi } from "../../api/adapters";
import { logInfo } from "../../utils/logger";

(async () => {
  process.env.USE_PRIVATE_MOCK = '1';
  process.env.MOCK_FORCE_EXIT = process.env.MOCK_FORCE_EXIT || '1';
  const api = createPrivateApi();
  const r:any = await api.trade({ currency_pair: 'btc_jpy', action: 'bid', price: 1000000, amount: 0.001 });
  const id = String(r.return.order_id);
  const open1:any = await api.active_orders({ currency_pair: 'btc_jpy' });
  logInfo('OPEN_RAW', open1);
  const found1 = Array.isArray(open1) ? open1.find((o:any) => o.order_id === id) : undefined;
  const hist1:any = await api.trade_history({ currency_pair: 'btc_jpy', count: 50 });
  logInfo('MOCK SUM', { id, remain: found1?.amount, filled: found1?.filled, fills: hist1.filter((h:any)=> String(h.order_id)===id) });
})();
