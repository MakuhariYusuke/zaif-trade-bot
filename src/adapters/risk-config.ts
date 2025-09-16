import fs from 'fs';
import path from 'path';
import { logInfo, logError } from '../utils/logger';

export interface RiskConfig { stopLossPct: number; takeProfitPct: number; positionPct: number; smaPeriod: number; positionsFile: string; trailTriggerPct: number; trailStopPct: number; dcaStepPct: number; maxPositions: number; maxDcaPerPair: number; minTradeSize: number; maxSlippagePct: number; indicatorIntervalSec: number; }
export type PositionSide = "long" | "short";
export interface Position { id: string; pair: string; side: PositionSide; entryPrice: number; amount: number; timestamp: number; highestPrice?: number; dcaCount?: number; openOrderIds?: number[]; }

export function loadRiskConfig(): RiskConfig {
	const positionsFile = process.env.POSITIONS_FILE || path.resolve(process.cwd(), ".positions.json");
	return {
		stopLossPct: Number(process.env.RISK_STOP_LOSS_PCT || 0.02),
		takeProfitPct: Number(process.env.RISK_TAKE_PROFIT_PCT || 0.05),
		positionPct: Number(process.env.RISK_POSITION_PCT || 0.05),
		smaPeriod: Number(process.env.RISK_SMA_PERIOD || 20),
		positionsFile,
		trailTriggerPct: Number(process.env.RISK_TRAIL_TRIGGER_PCT || 0.05),
		trailStopPct: Number(process.env.RISK_TRAIL_STOP_PCT || 0.03),
		dcaStepPct: Number(process.env.RISK_DCA_STEP_PCT || 0.01),
		maxPositions: Number(process.env.RISK_MAX_POSITIONS || 5),
		maxDcaPerPair: Number(process.env.RISK_MAX_DCA_PER_PAIR || 3),
		minTradeSize: Number(process.env.RISK_MIN_TRADE_SIZE || 0.0001),
		maxSlippagePct: Number(process.env.RISK_MAX_SLIPPAGE_PCT || 0.005),
		indicatorIntervalSec: Number(process.env.RISK_INDICATOR_INTERVAL_SEC || 60)
	};
}

function readFileSafe(file: string): string | null {
	try { return fs.readFileSync(file, 'utf8'); } catch { return null; }
}

export function loadPositions(file: string): Position[] {
	const txt = readFileSafe(file);
	if (!txt) return [];
	try {
		const data = JSON.parse(txt);
		if (Array.isArray(data)) return data as Position[];
	} catch (err) { logError('JSON parse fail', err); }
	return [];
}

export function savePositions(file: string, positions: Position[]) {
	try { fs.writeFileSync(file, JSON.stringify(positions, null, 2)); } catch (err) { logError('save positions fail', err); }
}

export function openPosition(file: string, p: Omit<Position, 'id'|'timestamp'|'highestPrice'|'dcaCount'|'openOrderIds'> & { entryPrice: number; amount: number; }): Position {
	const positions = loadPositions(file);
	const pos: Position = {
		id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
		timestamp: Date.now(),
		...p,
		highestPrice: p.entryPrice,
		dcaCount: 0,
		openOrderIds: []
	};
	positions.push(pos);
	savePositions(file, positions);
	logInfo(`ポジション追加 id=${pos.id} pair=${pos.pair} entry=${pos.entryPrice}`);
	return pos;
}

export function incrementDca(file: string, posId: string) {
	const positions = loadPositions(file);
	const target = positions.find(p => p.id === posId);
	if (target) {
		target.dcaCount = (target.dcaCount || 0) + 1;
		savePositions(file, positions);
	}
}

export function removePosition(file: string, id: string) {
	const positions = loadPositions(file).filter(p => p.id !== id);
	savePositions(file, positions);
}
