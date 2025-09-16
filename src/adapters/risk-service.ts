/** @deprecated 次メジャーで core に統合予定 */
export { loadRiskConfig, loadPositions, savePositions } from './risk-config';
export { loadRiskConfig as getRiskConfig, loadPositions as getPositions, savePositions as savePositionsToFile } from './risk-config';
export { openPosition, incrementDca, removePosition } from './risk-config';
export { positionSizeFromBalance, describeExit } from "../core/risk";
export { calculateSma as calcSMA, calculateRsi as calcRSI, evaluateExitConditions as evaluateExitSignals, manageTrailingStop as trailManager } from "../core/risk";

// Contract adapter (delegates to core)
import type { RiskManager } from "@contracts";
import { CoreRiskManager } from "../core/risk";
import BaseService from "./base-service";

export class ServiceRiskManager extends BaseService implements RiskManager {
	private core = new CoreRiskManager();
	validateOrder(intent: any) { return this.core.validateOrder(intent); }
	manageTrailingStop(state: any, price: number) { return this.core.manageTrailingStop(state, price); }
	clampExposure(balance: any, intent: any) { return this.core.clampExposure(balance, intent); }
}
export function createServiceRiskManager(): RiskManager { return new ServiceRiskManager(); }
