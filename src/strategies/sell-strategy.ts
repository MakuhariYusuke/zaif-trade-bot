import { warnOnce } from '@utils/logger';
warnOnce('deprecate-sell-strategy', 'Use @application/runSellStrategy instead (will be removed in next major).', { category: 'CONFIG' });

export { runSellStrategy } from '@application/strategies/sell-strategy-app';
