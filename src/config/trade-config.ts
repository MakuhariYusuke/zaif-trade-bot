import fs from 'fs';
import path from 'path';

export interface PhaseStep { phase: number; ordersPerDay: number; }
export interface TradeConfig {
  pair: string;
  phase: number;
  phaseSteps: PhaseStep[];
}

const DEFAULT_CONFIG: TradeConfig = {
  pair: 'btc_jpy',
  phase: 1,
  phaseSteps: [
    { phase: 1, ordersPerDay: 1 },
    { phase: 2, ordersPerDay: 3 },
    { phase: 3, ordersPerDay: 5 },
    { phase: 4, ordersPerDay: 10 },
  ],
};

export function loadTradeConfig(cwd = process.cwd()): TradeConfig {
  const override = process.env.TRADE_CONFIG_FILE;
  const file = override ? path.resolve(cwd, override) : path.resolve(cwd, 'trade-config.json');
  try {
    const raw = fs.readFileSync(file, 'utf8');
    const cfg = JSON.parse(raw);
    return normalizeTradeConfig(cfg);
  } catch {
    return { ...DEFAULT_CONFIG };
  }
}

export function normalizeTradeConfig(cfg: any): TradeConfig {
  const pair = String(cfg?.pair || 'btc_jpy').toLowerCase();
  const phase = Number(cfg?.phase || 1);
  const stepsIn = Array.isArray(cfg?.phaseSteps) ? cfg.phaseSteps : [];
  const steps: PhaseStep[] = stepsIn
    .map((s: any) => ({ phase: Number(s?.phase), ordersPerDay: Number(s?.ordersPerDay) }))
    .filter((s: any) => Number.isFinite(s.phase) && s.phase > 0 && Number.isFinite(s.ordersPerDay) && s.ordersPerDay > 0)
    .sort((a: PhaseStep, b: PhaseStep) => a.phase - b.phase);
  return { pair, phase, phaseSteps: steps.length ? steps : DEFAULT_CONFIG.phaseSteps };
}

export function getOrdersPerDay(cfg: TradeConfig, phase?: number): number {
  const p = Number.isFinite(phase) ? (phase as number) : cfg.phase;
  const found = cfg.phaseSteps.find(s => s.phase === p);
  if (found) return found.ordersPerDay;
  // if not found, best effort: pick nearest lower, else first
  const lower = [...cfg.phaseSteps].filter(s => s.phase <= p).sort((a,b)=>b.phase-a.phase)[0];
  return lower ? lower.ordersPerDay : cfg.phaseSteps[0]?.ordersPerDay ?? 1;
}
