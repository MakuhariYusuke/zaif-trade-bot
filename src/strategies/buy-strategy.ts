import { warnOnce } from '@utils/logger';
warnOnce('deprecate-buy-strategy', 'Use @application/runBuyStrategy instead (will be removed in next major).', { category: 'CONFIG' });

export { runBuyStrategy } from '@application/strategies/buy-strategy-app';
