import { TradeConfig } from './trade-config';
import { warnOnce, logWarn } from '../utils/logger';

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  normalized?: TradeConfig;
}

function isAscUnique(phases: number[]): boolean {
  for (let i=1;i<phases.length;i++) if (phases[i] <= phases[i-1]) return false;
  return true;
}

const PAIR_RE = /^[a-z0-9_]+$/;

export function validateConfig(cfg: TradeConfig, env: NodeJS.ProcessEnv = process.env): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!(cfg.phase > 0)) errors.push('phase must be > 0');
  if (!Array.isArray(cfg.phaseSteps) || cfg.phaseSteps.length === 0) errors.push('phaseSteps required');
  else {
    const phases = cfg.phaseSteps.map(s=>s.phase);
    if (!isAscUnique(phases)) errors.push('phaseSteps must be strictly ascending');
    if (cfg.phaseSteps.some(s=> !(s.ordersPerDay>0))) errors.push('ordersPerDay must be > 0 for all phaseSteps');
  }
  const maxStep = Math.max(...(cfg.phaseSteps||[]).map(s=>s.ordersPerDay||0), 0);
  if (cfg.maxOrdersPerDay != null && cfg.maxOrdersPerDay < maxStep) warnings.push('maxOrdersPerDay < largest ordersPerDay in steps');
  if (cfg.slippageGuardPct != null && !(cfg.slippageGuardPct >= 0 && cfg.slippageGuardPct <= 0.5)) warnings.push('slippageGuardPct outside [0,0.5]');
  if (cfg.maxLossPerDay != null && !(cfg.maxLossPerDay > 0)) warnings.push('maxLossPerDay should be > 0');
  if (!PAIR_RE.test(cfg.pair)) errors.push('pair format invalid');

  // Env validations
  function num(name: string, fn: (v:number)=>boolean, warnMsg: string){
    const raw = env[name]; if (raw == null) return; const v = Number(raw); if (!Number.isFinite(v)) { warnings.push(`${name} not numeric`); return; } if (!fn(v)) warnings.push(`${name} ${warnMsg}`); }
  num('REPRICE_MAX_ATTEMPTS', v=>v>=0, 'must be >= 0');
  num('RISK_MAX_SLIPPAGE_PCT', v=>v>=0 && v<=0.2, 'must be in [0,0.2]');

  // Emit warnOnce for each warning (stable id prefix)
  for (const w of warnings) warnOnce('cfg:'+w, w);
  for (const e of errors) warnOnce('cfg-err:'+e, e);
  if (errors.length) logWarn('[CONFIG] invalid trade-config detected', { errors, warnings });
  return { valid: errors.length === 0, errors, warnings, normalized: cfg };
}
