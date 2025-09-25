import fs from 'fs';
import path from 'path';

export interface TradeState {
  phase: number;
  consecutiveDays: number; // days with at least 1 completed trade
  totalSuccess: number;    // cumulative successful trades
  lastDate: string;        // YYYY-MM-DD
  ordersPerDay?: number;   // effective orders/day for current phase
}

export const DEFAULT_STATE: TradeState = {
  phase: 1,
  consecutiveDays: 0,
  totalSuccess: 0,
  lastDate: '',
  ordersPerDay: 1,
};

export interface PromotionRules {
  // thresholds: phase 1->2 needs 5 consecutive days; 2->3 needs 20 totalSuccess; 3->4 needs 50 totalSuccess
  to2_consecutiveDays: number;
  to3_totalSuccess: number;
  to4_totalSuccess: number;
}

export const DEFAULT_RULES: PromotionRules = {
  to2_consecutiveDays: Number(process.env.PROMO_TO2_DAYS || (process.env.TEST_MODE === '1' ? 1 : 5)),
  to3_totalSuccess: Number(process.env.PROMO_TO3_SUCCESS || 20),
  to4_totalSuccess: Number(process.env.PROMO_TO4_SUCCESS || 50),
};

// Always compute from current env (tests can tweak PROMO_* between runs)
export function getPromotionRules(): PromotionRules {
  return {
    to2_consecutiveDays: Number(process.env.PROMO_TO2_DAYS || (process.env.TEST_MODE === '1' ? 1 : 5)),
    to3_totalSuccess: Number(process.env.PROMO_TO3_SUCCESS || 20),
    to4_totalSuccess: Number(process.env.PROMO_TO4_SUCCESS || 50),
  };
}

function stateFilePath(cwd = process.cwd()): string {
  const override = process.env.TRADE_STATE_FILE;
  return path.resolve(cwd, override || 'trade-state.json');
}

export function loadTradeState(cwd = process.cwd()): TradeState {
  try {
    const p = stateFilePath(cwd);
    const raw = fs.readFileSync(p, 'utf8');
    const s = JSON.parse(raw);
    return normalizeState(s);
  } catch {
    return { ...DEFAULT_STATE };
  }
}

export function saveTradeState(state: TradeState, cwd = process.cwd()): void {
  const p = stateFilePath(cwd);
  try { fs.writeFileSync(p, JSON.stringify(state, null, 2), 'utf8'); } catch {}
}

export function normalizeState(s: any): TradeState {
  const phase = Number(s?.phase || 1);
  const consecutiveDays = Math.max(0, Number(s?.consecutiveDays || 0));
  const totalSuccess = Math.max(0, Number(s?.totalSuccess || 0));
  const lastDate = String(s?.lastDate || '');
  const ordersPerDay = Number(s?.ordersPerDay || (phase === 1 ? 1 : phase === 2 ? 3 : phase === 3 ? 10 : 25));
  return { phase, consecutiveDays, totalSuccess, lastDate, ordersPerDay };
}

export interface DayProgress {
  date: string;        // YYYY-MM-DD for this run
  daySuccess: number;  // successful trades counted for this day
}

export interface PromotionResult {
  promoted: boolean;
  fromPhase: number;
  toPhase: number;
  reason?: string;
}

export function applyPhaseProgress(state: TradeState, progress: DayProgress, rules: PromotionRules = DEFAULT_RULES): { state: TradeState; promotion?: PromotionResult } {
  const s = { ...state };
  const isNewDay = s.lastDate !== progress.date;
  if (isNewDay) {
    s.lastDate = progress.date;
    // Count as consecutive day if at least one success
    if (progress.daySuccess > 0) s.consecutiveDays += 1; else s.consecutiveDays = 0;
  } else {
    // Same day; only consecutive flag changes if previously zero and now success arrives
    if (progress.daySuccess > 0 && s.consecutiveDays === 0) s.consecutiveDays = 1;
  }
  if (progress.daySuccess > 0) s.totalSuccess += progress.daySuccess;

  const prevPhase = s.phase;
  let reason: string | undefined;
  if (s.phase === 1 && s.consecutiveDays >= rules.to2_consecutiveDays) { s.phase = 2; reason = `consecutiveDays>=${rules.to2_consecutiveDays}`; }
  if (s.phase === 2 && s.totalSuccess >= rules.to3_totalSuccess) { s.phase = 3; reason = `totalSuccess>=${rules.to3_totalSuccess}`; }
  if (s.phase === 3 && s.totalSuccess >= rules.to4_totalSuccess) { s.phase = 4; reason = `totalSuccess>=${rules.to4_totalSuccess}`; }

  if (s.phase !== prevPhase) {
    // escalate ordersPerDay mapping new policy: 1->1,2->3,3->10,4->25
    const map: Record<number, number> = { 1: 1, 2: 3, 3: 10, 4: 25 };
    s.ordersPerDay = map[s.phase] || s.ordersPerDay || 1;
    return { state: s, promotion: { promoted: true, fromPhase: prevPhase, toPhase: s.phase, reason } };
  } else {
    // keep in sync if undefined
    if (!s.ordersPerDay) {
      const map: Record<number, number> = { 1: 1, 2: 3, 3: 10, 4: 25 };
      s.ordersPerDay = map[s.phase] || 1;
    }
  }
  return { state: s };
}
