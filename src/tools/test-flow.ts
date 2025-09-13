import dotenv from 'dotenv';
dotenv.config();
import { strategyOnce } from '../index';
import { logInfo, logWarn, logError } from '../utils/logger';
import { createPrivateApi } from '../api/adapters';
import { getOrderBook } from '../api/public';

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }

(async () => {
    const pair = process.env.PAIR || 'btc_jpy';
    const qtyRequired = process.env.TEST_FLOW_QTY;
    const rateRequired = process.env.TEST_FLOW_RATE;
    if (!qtyRequired || !rateRequired) {
        logError('TEST_FLOW_QTY and TEST_FLOW_RATE are required for test-flow');
        process.exit(1);
    }
    const qtyDefault = qtyRequired;
    const flows = (process.env.FLOWS || 'BUY_ONLY,SELL_ONLY,BUY_SELL,SELL_BUY').split(',').map(s => s.trim()).filter(Boolean);
    const exchanges = (process.env.EXCHANGE || 'zaif').split(',').map(s => s.trim().toLowerCase());
    const runLive = process.env.ALLOW_LIVE === '1';
    const dryModes = runLive ? ['1', '0'] : ['1'];

    process.env.TEST_FLOW_QTY = qtyDefault;
    for (const ex of exchanges) {
        process.env.EXCHANGE = ex;
        for (const dry of dryModes) {
            process.env.DRY_RUN = dry;
            for (const flow of flows) {
                process.env.TRADE_FLOW = flow;
                const liveTag = dry === '0' ? '[LIVE]' : '[DRY]';
                logInfo(`[TEST-FLOW] ${liveTag} ex=${ex} flow=${flow} qty=${qtyDefault} rate=${rateRequired}`);
                const EXECUTE = process.env.DRY_RUN !== '1';
                try {
                    // Run the strategy loop once to produce logs/stats
                    await strategyOnce(pair, EXECUTE);

                    // If live, also place cancellable orders explicitly for BUY_SELL/SELL_BUY to ensure both legs cancel
                    const api:any = createPrivateApi();
                    const dryMode = process.env.DRY_RUN === '1';
                    if (flow === 'BUY_SELL' || flow === 'SELL_BUY') {
                        const qty = Number(qtyDefault);
                        const rate = Number(rateRequired);
                        const ob:any = await getOrderBook(pair);
                        const bestBid = Number((ob?.bids?.[0]?.[0]) || 0);
                        const bestAsk = Number((ob?.asks?.[0]?.[0]) || 0);
                        const buyRate = Math.max(1, rate || bestBid || (bestAsk*0.999));
                        const sellRate = Math.max(1, rate || bestAsk || (bestBid*1.001));
                        const place = async (action:'bid'|'ask', price:number)=>{
                            if (dryMode){
                                const id = `DRY-${Date.now()}`; logInfo(`[TEST-FLOW] [SIM] place ${action} id=${id} price=${price} qty=${qty}`);
                                logInfo(`[TEST-FLOW] [SIM] cancel ${id}`); return id;
                            }
                            const r:any = await api.trade({ currency_pair: pair, action, price, amount: qty });
                            const id = String(r?.return?.order_id || ''); logInfo(`[TEST-FLOW] [LIVE] placed ${action} id=${id} price=${price} qty=${qty}`);
                            try { await api.cancel_order({ order_id: id }); logInfo(`[TEST-FLOW] [LIVE] canceled ${id}`); } catch(e:any){ logWarn('[TEST-FLOW] cancel failed', e?.message||e); }
                            return id;
                        };
                        if (flow === 'BUY_SELL') { await place('bid', buyRate); await place('ask', sellRate); }
                        else { await place('ask', sellRate); await place('bid', buyRate); }
                    }
                } catch (e: any) { logWarn('[TEST-FLOW] iteration error', e?.message || e); }
                // Wait 500ms between test iterations to avoid rate limits or API bans; adjust as needed for faster tests.
                await sleep(500);
            }
        }
    }
    logInfo('[TEST-FLOW] done');
})();
