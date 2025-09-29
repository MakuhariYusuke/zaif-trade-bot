import { BuyStrategy } from '@core/strategies/buy-strategy';
import { CoreRiskManager } from '@core/risk';
import { logger } from '@utils/logger';

export async function runBuyStrategy(ctx: any) {
  const strat = new BuyStrategy(logger, new CoreRiskManager());
  return strat.runCycle(ctx);
}
