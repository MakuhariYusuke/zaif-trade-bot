import { describe, it, expect, vi } from 'vitest';
import { BaseStrategy } from '../../../src/core/strategies/base-strategy';

class FakeRisk {
  validateOrder = vi.fn((x:any)=>({ ok:true, value: undefined }));
  manageTrailingStop = vi.fn(()=>null);
  clampExposure = vi.fn((b:any,i:any)=>i);
}

class TestStrategy extends BaseStrategy {
  order: string[] = [];
  protected preCheck(ctx: any) { this.order.push('pre'); this.logger.debug('pre', ctx); }
  protected async decide(ctx: any): Promise<any> { this.order.push('decide'); return { foo: 1 }; }
  protected applyRisk(decision: any, ctx: any){ this.order.push('risk'); return decision; }
  protected async execute(ctx: any, decision: any){ this.order.push('exec'); return { ok: true }; }
  protected summarize(ctx: any, result: any){ this.order.push('sum'); this.logger.info('done', { result }); }
}

const fakeLogger = { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() };

describe('BaseStrategy', () => {
  it('calls hooks in order', async () => {
    const risk = new FakeRisk() as any;
    const s = new TestStrategy(fakeLogger as any, risk);
    const res = await s.runCycle({});
    expect(res).toEqual({ ok: true });
    expect((s as any).order).toEqual(['pre', 'decide', 'risk', 'exec', 'sum']);
    expect(fakeLogger.debug).toHaveBeenCalled();
    expect(fakeLogger.info).toHaveBeenCalled();
  });
});
