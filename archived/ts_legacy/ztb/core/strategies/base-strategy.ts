import type { RiskManager } from '@contracts';
import type { Logger } from '../../utils/logger';

export abstract class BaseStrategy {
  protected logger: Logger;
  protected risk: RiskManager;

  constructor(logger: Logger, risk: RiskManager){
    this.logger = logger;
    this.risk = risk;
  }

  async runCycle(ctx: any): Promise<any> {
    this.preCheck(ctx);
    const decision = await this.decide(ctx);
    const safeDecision = this.applyRisk(decision, ctx);
    const result = await this.execute(ctx, safeDecision);
    this.summarize(ctx, result);
    return result;
  }

  protected preCheck(ctx: any) { this.logger.debug('preCheck', ctx); }
  protected abstract decide(ctx: any): Promise<any>;
  protected applyRisk(decision: any, ctx: any){ return this.risk.validateOrder(decision); }
  protected async execute(ctx: any, decision: any): Promise<any>{ return decision; }
  protected summarize(ctx: any, result: any){ this.logger.info('summarize', { ctx, result }); }
}
