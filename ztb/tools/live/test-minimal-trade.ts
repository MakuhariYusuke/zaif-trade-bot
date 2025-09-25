import dotenv from 'dotenv';
dotenv.config();
import { createPrivateApi } from '../../api/adapters';
import { getOrderBook } from '../../api/public';
import { logInfo, logWarn, logError } from '../../utils/logger';
import { sleep, fetchBalances, clampAmountForSafety, baseFromPair, getExposureWarnPct, computeExposureRatio } from '../../utils/toolkit';

type Flow = 'BUY_ONLY' | 'SELL_ONLY' | 'BUY_SELL' | 'SELL_BUY';

(async () => {
    const ex = (process.env.EXCHANGE || 'zaif').toLowerCase();
    const dry = process.env.DRY_RUN === '1';
    const pair = process.env.PAIR || 'btc_jpy';
    const flow = (process.env.TRADE_FLOW || 'BUY_ONLY') as Flow;
    const qtyRaw = Number(process.env.TEST_FLOW_QTY || '0.002');
    const rateOverride = Number(process.env.TEST_FLOW_RATE || '0');
    if (!qtyRaw || qtyRaw <= 0) { console.error('TEST_FLOW_QTY must be > 0'); process.exit(1); }
    const api = createPrivateApi();
    logInfo(`[MIN-TRADE] ex=${ex} dry=${dry} flow=${flow} qty=${qtyRaw} rate=${rateOverride || '(from OB)'}`);
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

    async function place(action: 'bid' | 'ask', price: number, qty: number) {
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

    // SAFETY clamp if enabled
    let qtyBid = qtyRaw;
    let qtyAsk = qtyRaw;
    if (process.env.SAFETY_MODE === '1') {
        try {
            const funds = await fetchBalances(api);
            qtyBid = clampAmountForSafety('bid', qtyRaw, pxBid, funds, pair);
            qtyAsk = clampAmountForSafety('ask', qtyRaw, pxAsk, funds, pair);
        } catch (e:any) { logWarn('[MIN-TRADE] fetchBalances failed for clamp', e?.message||e); }
    }

    function warnIfOver5Pct(side:'bid'|'ask', qty:number, price:number, balancesBefore: Record<string, number>){
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

    const balancesRef = (process.env.SAFETY_MODE==='1') ? await (async()=>{ try { return await fetchBalances(api); } catch { return null; } })() : null;

    if (flow === 'BUY_ONLY') {
        if (balancesRef) warnIfOver5Pct('bid', qtyBid, pxBid, balancesRef);
        await place('bid', pxBid, qtyBid);
    } else if (flow === 'SELL_ONLY') {
        if (balancesRef) warnIfOver5Pct('ask', qtyAsk, pxAsk, balancesRef);
        await place('ask', pxAsk, qtyAsk);
    } else if (flow === 'BUY_SELL') {
        if (balancesRef) warnIfOver5Pct('bid', qtyBid, pxBid, balancesRef);
        await place('bid', pxBid, qtyBid);
        if (balancesRef) warnIfOver5Pct('ask', qtyAsk, pxAsk, balancesRef);
        await place('ask', Math.max(pxAsk, (rateOverride > 0 ? rateOverride : (pxBid * 1.001))), qtyAsk);
    } else if (flow === 'SELL_BUY') {
        if (balancesRef) warnIfOver5Pct('ask', qtyAsk, pxAsk, balancesRef);
        await place('ask', pxAsk, qtyAsk);
        if (balancesRef) warnIfOver5Pct('bid', qtyBid, pxBid, balancesRef);
        await place('bid', Math.min(pxBid, (rateOverride > 0 ? rateOverride : (pxAsk * 0.999))), qtyBid);
    }
    logInfo('[MIN-TRADE] done');
})();
