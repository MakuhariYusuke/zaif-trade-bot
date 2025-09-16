try {
    const g: any = global as any;
    if (!g.__deprecate_position_store_once) {
        g.__deprecate_position_store_once = 1;
        import('@utils/logger').then(m => m.warnOnce?.(
            'deprecate-services-position-store',
            'Import path deprecated: use @adapters/position-store (will be removed in next major).',
            { category: 'CONFIG' }
        )).catch(()=>{});
    }
} catch {}

export * from '@adapters/position-store';
import * as impl from '@adapters/position-store';
export default (impl as any).default ?? impl;
