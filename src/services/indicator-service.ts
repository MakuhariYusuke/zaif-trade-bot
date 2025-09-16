// Emit deprecation once without hard import-time requirement for tests
try {
  const g: any = global as any;
  if (!g.__deprecate_indicator_svc_once) {
    g.__deprecate_indicator_svc_once = 1;
    // lazy import to play nice with vitest mocks
    import('@utils/logger').then(m => m.warnOnce?.(
      'deprecate-services-indicator-service',
      'Import path deprecated: use @adapters/indicator-service (will be removed in next major).',
      { category: 'CONFIG' }
    )).catch(()=>{});
  }
} catch {}

export * from '@adapters/indicator-service';
import * as impl from '@adapters/indicator-service';
export default (impl as any).default ?? impl;
