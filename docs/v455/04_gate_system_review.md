# v455 Gate System Review (v1)

## Scope
- ztb/trading/signal/calibration_map.py
- ztb/trading/signal/entry_system.py
- config/v455/gate_config.json
- scripts/v455/run_backtest.py

## Strengths
- Relative binning aligns the Gate with dynamic thresholds, reducing the prior binning mismatch.
- Gate logging (EV, cost, n_eff) is present in the backtest runner, which supports root-cause analysis.
- Configuration is separated into gate_config.json, enabling controlled parameter sweeps.

## Weaknesses / Risks
- Exploration is still effectively closed at cold start: avg_win/avg_loss are 0, so EV = -cost even with p_win_mean/ucb.
- Relative binning now requires threshold on process_signal/update_outcome; existing call sites/tests are not updated.
- Warm-up threshold relaxation changes shadow execution behavior vs env behavior, biasing early stats.
- Gate block reasons are derived from p_win_lcb even when probability_mode is ucb/mean.
- env_threshold is logged as the same value as threshold, so data alignment checks are not actually validated.
- gate_config uses fee_rate=0.0 and very low cost parameters, which can overstate performance and over-trade.

## Recommendations
- Add priors for avg_win/avg_loss (or an EV prior) so early EV is not always negative.
- Keep warm-up relaxation in the Gate only; do not use relaxed thresholds for shadow execution.
- Add default threshold parameter or update all call sites/tests to pass threshold.
- Log p_win_used based on probability_mode and compute block reasons from the same metric.
- Log both env_threshold (pre-warmup) and gate_threshold (post-warmup) to detect misalignment.
- Recalibrate cost parameters with a non-zero fee baseline for training realism.
- Add risk caps for early training (trade rate limit, max drawdown kill switch, max notional).

## Exploration Strategy (Optimistic but Safe)
- Stage 1: Cost-only gate for the first N trades (e.g., allow if cost < k * ATR).
- Stage 2: UCB/Thompson sampling on EV with capped position size.
- Stage 3: Full EV gate once n_eff >= n_min.
- Always cap trade frequency and position size during exploration to avoid runaway losses.
