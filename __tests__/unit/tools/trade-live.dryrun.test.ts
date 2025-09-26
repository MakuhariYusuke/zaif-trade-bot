import { describe, it, expect } from 'vitest';
import { runTradeLive } from '../../../ztb/tools/trade-live';

describe('trade-live dry-run', () => {
  it('emits plan and returns summary', async () => {
    process.env.DRY_RUN = '1';
    const out: any = await runTradeLive({ today: '2025-01-10', dryRun: true });
    expect(out).toBeTruthy();
    expect(out.pair).toBeDefined();
    expect(out.phase).toBeGreaterThan(0);
    expect(out.plannedOrders).toBeGreaterThan(0);
    expect(out.today).toBe('2025-01-10');
  });
});
