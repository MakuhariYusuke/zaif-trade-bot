# Doc 19: Phase 5.6 Final Review (Post-Fix Verification)

## Verdict
Doc 18 says "all issues resolved", but the current code still has blocking defects and metric integrity problems. The results are not trustworthy yet.

## Critical Findings (Stop-Ship)
- Entry gate crash: the env checks `gate_result.allowed` but GateResult only defines `should_enter`. This will raise AttributeError when entry_gate is enabled. (`ztb/trading/environment/fast_intraday_env_v456.py:500`, `ztb/trading/environment/fast_intraday_env_v456.py:506`, `ztb/trading/signal/types.py:34`)
- Entry gate config is not wired into env_config, so the gate never actually enables. The "buy-only action" issue is therefore still unresolved. (`config/v458/base/config.yaml:87`, `ztb/training/utils/v457_config_utils.py:25`, `scripts/v456/train_v456_optimized.py:158`, `scripts/v458/run_walk_forward_v458.py:232`)
- Fee/slippage is double-counted: `trade_pnl` in the env is already net-of-cost, yet evaluator passes `fee_paid`/`slippage_paid` and the v457 reporter subtracts again. Net PnL/ROI/Sharpe are understated. (`ztb/trading/environment/fast_intraday_env_v456.py:579`, `ztb/evaluation/walk_forward/evaluator.py:528`, `scripts/v457/backtest_v457.py:149`)
- In-sample/out-of-sample contamination: the same reporter instance is reused for validation and test, so test stats include validation trades and balances. (`ztb/evaluation/walk_forward/evaluator.py:371`, `ztb/evaluation/walk_forward/evaluator.py:389`, `ztb/evaluation/walk_forward/evaluator.py:399`)

## Major Findings
- "close" trades are counted as "short": evaluator emits `trade_type = "close"` but reporter treats any non-long trade as short, skewing stats. (`ztb/evaluation/walk_forward/evaluator.py:521`, `ztb/evaluation/walk_forward/reporter.py:230`)
- Entry/exit logging is still wrong on reversals: `entry_price` updates only when the previous position is zero, so flip trades carry the old entry price. (`ztb/trading/environment/fast_intraday_env_v456.py:579`)
- BacktestReporter is duplicated in three places with different net/gross assumptions, which makes metrics inconsistent and brittle. (`ztb/evaluation/walk_forward/evaluator.py:53`, `ztb/evaluation/walk_forward/evaluator.py:147`, `scripts/v457/backtest_v457.py:47`, `ztb/evaluation/walk_forward/reporter.py:204`)
- Import error handler uses `logger` before initialization; missing dependencies will cause a NameError instead of a clean error message. (`scripts/v458/run_walk_forward_v458.py:26`)
- AB tests are effectively a no-op because `compare_multiple_evaluations([result])` always returns "Need at least 2 results". (`scripts/v458/run_walk_forward_v458.py:273`)
- Calibration map path is configured but never loaded; `load_state` exists but is not called anywhere. (`config/v458/base/config.yaml:88`, `ztb/trading/signal/entry_system.py:87`)

## Reuse / vXXX Assets to Leverage
- Backtest stats: standardize on the v457 reporter or move it into `ztb/evaluation/walk_forward/reporter.py` and delete duplicates. (`scripts/v457/backtest_v457.py`)
- AB test tooling: `tools/ab_test_runner.py`, `tools/run_ab_searches.py`, `experiments/v450/run_ab_test_threshold_v450.py`
- Prior stability criteria: `docs/v457/32_seed_stability_test.md`, `docs/v457/34_seed_stability_lost_alpha_review.md`
- Baseline engine: `ztb/analysis/baseline_comparison.py` (strategy keys `buy_hold`, `sma_crossover`)

## Recommended Fix Plan (Minimal, Deterministic)
1) Wire entry_gate config into env_config. Option A: move `entry_gate` under `training.environment`. Option B: merge top-level `entry_gate` into `extract_env_config`.
2) Fix the gate check to `gate_result["should_enter"]`.
3) Pick one PnL convention end-to-end:
   - If reporter expects gross, pass gross and keep fee/slippage subtraction there.
   - If env already returns net, stop subtracting in the reporter.
4) Use separate reporters for validation and test (or reset reporter between phases).
5) Handle "close" trades explicitly (separate type or skip; do not count as short).
6) Update `entry_price` on reversals (close old position, then set entry for new position).
7) If you want calibration to learn, call `entry_system.load_state` on init and `update_outcome` when a trade closes.
8) AB tests: collect two or more WalkForwardResult objects (different seeds) before calling `compare_multiple_evaluations`.

## Verification Checklist
- Run `scripts/v458/run_walk_forward_v458.py` with entry_gate enabled; confirm no AttributeError.
- Validate a single trade sequence: reporter net PnL matches env accounting (no double cost).
- Confirm val/test metrics are isolated (separate reporters).
- Baseline comparison returns per-strategy outputs and AB test runs on >=2 results.
