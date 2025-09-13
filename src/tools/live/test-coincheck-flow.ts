import dotenv from 'dotenv';
dotenv.config();
import { createPrivateApi } from '../../api/adapters';
import { getOrderBook } from '../../api/public';
import { logInfo, logWarn } from '../../utils/logger';
import { sleep } from '../../utils/toolkit';

(async () => {
    process.env.EXCHANGE = 'coincheck';
    const api: any = createPrivateApi();
    const pair = 'btc_jpy';
    logInfo('[FLOW] Coincheck SELL -> opens -> cancel -> history -> BUY');
    const ob = await getOrderBook(pair);
    const bestBid = Number((ob.bids?.[0]?.[0]) || 0);
    const bestAsk = Number((ob.asks?.[0]?.[0]) || 0);
    const qty = Number(process.env.CC_TEST_AMOUNT || '0.005');

    // SELL
    const sellRate = Math.max(1, bestBid || Number(process.env.CC_TEST_RATE || '1000000'));
    logInfo('[FLOW] SELL place', { rate: sellRate, qty });
    const s = await api.trade({ currency_pair: pair, action: 'ask', price: sellRate, amount: qty });
    logInfo('[FLOW] SELL order id', s.return.order_id);
    await sleep(1500);
    const open1 = await api.active_orders({ currency_pair: pair });
    logInfo('[FLOW] opens len', open1.length);
    if (open1.length) {
        logInfo('[FLOW] cancel', open1[0].order_id);
        try { await api.cancel_order({ order_id: String(open1[0].order_id) }); } catch(e:any){ logWarn('[FLOW] cancel failed', e?.message||e); }
    }
    await sleep(1500);
    const hist1 = await api.trade_history({ currency_pair: pair, count: 10 });
    logInfo('[FLOW] history(SELL first 3)', hist1.slice(0, 3));

    // BUY
    const buyRate = Math.max(1, bestAsk || (sellRate * 0.99));
    logInfo('[FLOW] BUY place', { rate: buyRate, qty });
    const b = await api.trade({ currency_pair: pair, action: 'bid', price: buyRate, amount: qty });
    logInfo('[FLOW] BUY order id', b.return.order_id);
    await sleep(1500);
    const open2 = await api.active_orders({ currency_pair: pair });
    logInfo('[FLOW] opens len 2', open2.length);
    const hist2 = await api.trade_history({ currency_pair: pair, count: 10 });
    logInfo('[FLOW] history(BUY first 3)', hist2.slice(0, 3));
    logInfo('OK');
})();
