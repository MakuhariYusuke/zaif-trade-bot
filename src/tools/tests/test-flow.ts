import { strategyOnce } from '../../index';
import { logInfo, logWarn, logError } from '../../utils/logger';
import { createPrivateApi } from '../../api/adapters';
import { getOrderBook } from '../../api/public';

async function main(){
  try {
    const pair = process.env.PAIR || 'btc_jpy';
    const dry = process.env.DRY_RUN === '1';
    const api = createPrivateApi();
    const ob = await getOrderBook(pair);
    logInfo('[FLOW] best bid/ask', { bid: ob?.bids?.[0]?.[0], ask: ob?.asks?.[0]?.[0] });
    await strategyOnce(pair, !dry);
    logInfo('[FLOW] done');
  } catch (e: any) {
    logError('[FLOW] error', e?.message || e);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}
