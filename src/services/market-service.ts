try {
  const g: any = global as any;
  if (!g.__deprecate_market_svc_once) {
    g.__deprecate_market_svc_once = 1;
    import('@utils/logger').then(m => m.warnOnce?.(
      'deprecate-services-market-service',
      'Import path deprecated: use @adapters/market-service (will be removed in next major).',
      { category: 'CONFIG' }
    )).catch(()=>{});
  }
} catch {}

export * from '@adapters/market-service';
import * as impl from '@adapters/market-service';
export default (impl as any).default ?? impl;