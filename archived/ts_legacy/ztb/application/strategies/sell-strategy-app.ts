import { SellStrategy } from '@core/strategies/sell-strategy';
import { CoreRiskManager } from '@core/risk';
import { logger } from '@utils/logger';

export async function runSellStrategy(ctx: any) {
  const strat = new SellStrategy(logger, new CoreRiskManager());
  return strat.runCycle(ctx);
}
