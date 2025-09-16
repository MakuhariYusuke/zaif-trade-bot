try {
	const g: any = global as any;
	if (!g.__deprecate_risk_svc_once) {
		g.__deprecate_risk_svc_once = 1;
		import('@utils/logger').then(m => m.warnOnce?.(
			'deprecate-services-risk-service',
			'Import path deprecated: use @adapters/risk-service (will be removed in next major).',
			{ category: 'CONFIG' }
		)).catch(()=>{});
	}
} catch {}

export * from '@adapters/risk-service';
import * as impl from '@adapters/risk-service';
export default (impl as any).default ?? impl;
