import { strategyOnce } from '../../src/index';
import { logInfo, logError } from '../../src/utils/logger';
import { createPrivateApi } from '../../src/api/adapters';
import { getOrderBook } from '../../src/api/public';
import { baseFromPair, fetchBalances } from '../../src/utils/toolkit';

async function main(){
  try {
    const pair = process.env.PAIR || 'btc_jpy';
    const dry = process.env.DRY_RUN === '1';
    const api = createPrivateApi();
    const ob = await getOrderBook(pair);
    logInfo('[FLOW] best bid/ask', { bid: ob?.bids?.[0]?.[0], ask: ob?.asks?.[0]?.[0] });
    try {
      const funds = await fetchBalances(api as any);
      const base = baseFromPair(pair).toLowerCase();
      const jpy = Number((funds as any).jpy || 0);
      const balBase = Number((funds as any)[base] || 0);
      logInfo('[FLOW][BALANCE]', { jpy, [base]: balBase });
    } catch {}
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
