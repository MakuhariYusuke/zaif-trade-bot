import { warnOnce } from '@utils/logger';
warnOnce(
    'deprecate-services-execution-service',
    'Import path deprecated: use @adapters/execution-service (will be removed in next major).',
    { category: 'CONFIG' }
);

export * from '@adapters/execution-service';
import * as impl from '@adapters/execution-service';
export default (impl as any).default ?? impl;