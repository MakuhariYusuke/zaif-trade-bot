import { createPrivateApi } from "../../api/adapters";
import { logInfo, logError } from "../../utils/logger";

(async () => {
  process.env.USE_PRIVATE_MOCK = '1';
  process.env.MOCK_FORCE_EXIT = process.env.MOCK_FORCE_EXIT || '1';
  const api: any = createPrivateApi();
  const amt = 0.001;
  const r = await api.trade({ currency_pair: 'btc_jpy', action: 'bid', price: 1_000_000, amount: amt });
  const id = String(r.return.order_id);
  const open = await api.active_orders({ currency_pair: 'btc_jpy' });
  const hist = await api.trade_history({ currency_pair: 'btc_jpy', count: 50 });
  const found = open.find((o: any) => String(o.order_id) === id);
  if (!found) { logError('Order not found in open list (expected partial)'); process.exit(1); }
  if (!(found.amount < amt)) { logError(`Remaining not reduced: remain=${found.amount}`); process.exit(1); }
  const fills = hist.filter((h: any) => String(h.order_id) === id);
  if (!fills.length) { logError('No fills recorded for order'); process.exit(1); }
  await api.cancel_order({ order_id: id });
  const open2 = await api.active_orders({ currency_pair: 'btc_jpy' });
  if (open2.find((o: any) => String(o.order_id) === id)) { logError('Order still present after cancel'); process.exit(1); }
  logInfo('OK');
})();
