import fs from 'fs';
import path from 'path';

export interface PhaseStep { phase: number; ordersPerDay: number; }
export interface TradeConfig {
  pair: string;
  phase: number;
  phaseSteps: PhaseStep[];
  maxOrdersPerDay?: number;
  maxLossPerDay?: number; // absolute currency loss cap (e.g. JPY)
  slippageGuardPct?: number; // e.g. 0.01 = 1%
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
  maxOrdersPerDay: 50,
  maxLossPerDay: 10000,
  slippageGuardPct: 0.01,
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
  const maxOrdersPerDay = Number(cfg?.maxOrdersPerDay ?? DEFAULT_CONFIG.maxOrdersPerDay);
  const maxLossPerDay = Number(cfg?.maxLossPerDay ?? DEFAULT_CONFIG.maxLossPerDay);
  const slippageGuardPct = Number(cfg?.slippageGuardPct ?? DEFAULT_CONFIG.slippageGuardPct);
  const out: TradeConfig = {
    pair,
    phase,
    phaseSteps: steps.length ? steps : DEFAULT_CONFIG.phaseSteps,
    maxOrdersPerDay: Number.isFinite(maxOrdersPerDay) && maxOrdersPerDay > 0 ? maxOrdersPerDay : DEFAULT_CONFIG.maxOrdersPerDay,
    maxLossPerDay: Number.isFinite(maxLossPerDay) && maxLossPerDay > 0 ? maxLossPerDay : DEFAULT_CONFIG.maxLossPerDay,
    slippageGuardPct: Number.isFinite(slippageGuardPct) && slippageGuardPct > 0 ? slippageGuardPct : DEFAULT_CONFIG.slippageGuardPct,
  };
  // basic warn to stdout (no logger dependency here) if values look extreme
  try {
    if ((out.maxOrdersPerDay || 0) > 1000) console.warn('[trade-config] maxOrdersPerDay unusually high:', out.maxOrdersPerDay);
    if ((out.slippageGuardPct || 0) > 0.20) console.warn('[trade-config] slippageGuardPct > 20%:', out.slippageGuardPct);
  } catch {}
  return out;
}

export function getOrdersPerDay(cfg: TradeConfig, phase?: number): number {
  const p = Number.isFinite(phase) ? (phase as number) : cfg.phase;
  const found = cfg.phaseSteps.find(s => s.phase === p);
  if (found) return found.ordersPerDay;
  // if not found, best effort: pick nearest lower, else first
  const lower = [...cfg.phaseSteps].filter(s => s.phase <= p).sort((a,b)=>b.phase-a.phase)[0];
  return lower ? lower.ordersPerDay : cfg.phaseSteps[0]?.ordersPerDay ?? 1;
}
