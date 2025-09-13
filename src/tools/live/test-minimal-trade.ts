import dotenv from 'dotenv';
dotenv.config();
import { createPrivateApi } from '../../api/adapters';
import { getOrderBook } from '../../api/public';
import { logInfo, logWarn, logError } from '../../utils/logger';

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }

type Flow = 'BUY_ONLY' | 'SELL_ONLY' | 'BUY_SELL' | 'SELL_BUY';

(async () => {
    const ex = (process.env.EXCHANGE || 'zaif').toLowerCase();
    const dry = process.env.DRY_RUN === '1';
    const pair = process.env.PAIR || 'btc_jpy';
    const flow = (process.env.TRADE_FLOW || 'BUY_ONLY') as Flow;
    const qty = Number(process.env.TEST_FLOW_QTY || '0.002');
    const rateOverride = Number(process.env.TEST_FLOW_RATE || '0');
    if (!qty || qty <= 0) { console.error('TEST_FLOW_QTY must be > 0'); process.exit(1); }
    const api = createPrivateApi();
    logInfo(`[MIN-TRADE] ex=${ex} dry=${dry} flow=${flow} qty=${qty} rate=${rateOverride || '(from OB)'}`);
    let ob: any;
    try {
        ob = await getOrderBook(pair);
        if (!ob || !Array.isArray(ob.bids) || !Array.isArray(ob.asks)) {
            throw new Error('Order book is invalid or undefined');
        }
    } catch (e: any) {
        logWarn(`[MIN-TRADE] Failed to fetch order book: ${e?.message || e}`);
        process.exit(1);
    }
    const bestBid = Number((ob?.bids?.[0]?.[0]) || 0);
    const bestAsk = Number((ob?.asks?.[0]?.[0]) || 0);
    const pxBid = rateOverride > 0
        ? rateOverride
        : (bestBid > 0
            ? bestBid
            : Math.max(1, bestAsk * 0.999));
    const pxAsk = rateOverride > 0
        ? rateOverride
        : (bestAsk > 0
            ? bestAsk
            : Math.max(1, bestBid * 1.001));

    async function place(action: 'bid' | 'ask', price: number) {
        if (dry) {
            const id = `DRY-${Date.now()}`;
            logInfo(`[MIN-TRADE][DRY] place ${action} id=${id} price=${price} qty=${qty}`);
            logInfo(`[MIN-TRADE][SIMULATED CANCEL] ${id}`);
            return id;
        }
        const r: any = await api.trade({ currency_pair: pair, action, price, amount: qty });
        const id = String(r?.return?.order_id || '');
        logInfo(`[MIN-TRADE] placed ${action} id=${id} price=${price} qty=${qty}`);
        try {
            await sleep(800);
            const res:any = await api.cancel_order({ order_id: id });
            let filledQty = 0;
            try {
                const hist:any[] = await api.trade_history({ currency_pair: pair, count: 50 });
                const fills = hist.filter(h=> String(h.order_id)===id);
                filledQty = fills.reduce((s,f)=> s + Number(f.amount||0), 0);
            } catch {}
            logInfo(`[CANCELLED] order_id=${id} filledQty=${filledQty}`);
        } catch (e: any) {
            logError('[CANCEL_FAIL]', e?.message || e);
        }
        return id;
    }

    if (flow === 'BUY_ONLY') {
        await place('bid', pxBid);
    } else if (flow === 'SELL_ONLY') {
        await place('ask', pxAsk);
    } else if (flow === 'BUY_SELL') {
        const b = await place('bid', pxBid);
        await place('ask', Math.max(pxAsk, (rateOverride > 0 ? rateOverride : (pxBid * 1.001))));
    } else if (flow === 'SELL_BUY') {
        const s = await place('ask', pxAsk);
        await place('bid', Math.min(pxBid, (rateOverride > 0 ? rateOverride : (pxAsk * 0.999))));
    }
    logInfo('[MIN-TRADE] done');
})();
