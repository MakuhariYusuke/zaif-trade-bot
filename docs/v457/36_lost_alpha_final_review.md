# 36. Lost Alpha Final Status Review

Target: `docs/v457/35_lost_alpha_final_status.md`

## Findings (ordered by severity)

1) Critical: MTF resampling leaks future data inside each bucket.
   - `resample()` aggregates the full 5m/15m/1h bucket and then `ffill` projects that value back to all 1m rows in the same bucket.
   - This means a 12:01 row sees a 12:04 close/high/low that was not yet available.
   - Evidence: `ztb/trading/environment/factory_v456.py:96`, `ztb/trading/environment/factory_v456.py:114`.
   - Recommendation: Use right-closed/right-labeled resample and shift by one bucket before ffill.

2) High: Ichimoku penalty is in raw JPY but applied to a reward already normalized by max_position.
   - `compute_hft_reward()` returns (pnl - cost - penalty) / max_position, but the injected penalty is not scaled.
   - Penalty strength changes with `max_position` and can be too weak when max_position < 1.0.
   - Evidence: `ztb/trading/environment/fast_intraday_env_v456.py:461`, `ztb/trading/environment/fast_intraday_env_v456.py:484`.
   - Recommendation: Convert the penalty into the same unit as the base reward (divide by max_position or inject before normalization).

3) High: Ichimoku signal calculation runs before column validation.
   - If a dataset is missing `high` or `low`, the env raises a KeyError before the existing missing-column error path.
   - Evidence: `ztb/trading/environment/fast_intraday_env_v456.py:39`, `ztb/trading/environment/fast_intraday_env_v456.py:160`.
   - Recommendation: validate required columns before `_calculate_ichimoku_signal`.

4) Medium: The injected "trend curriculum" is always-on, not stage-based.
   - The penalty is applied regardless of curriculum stage and never decays.
   - This diverges from doc 32's stage-weighted guidance and may block later-stage exploration.
   - Evidence: `ztb/trading/environment/fast_intraday_env_v456.py:463`.

5) Medium: Verification tests do not check for data leakage or reward scaling.
   - The MTF test only checks that 5m and 15m differ, not that they are causal.
   - Ichimoku test only checks non-zero signals; no check for penalty scale or alignment.
   - Evidence: `tests/verification/test_v457_lost_alpha.py:39`, `tests/verification/test_v457_lost_alpha.py:87`.

## Recommended Next Steps
1) Fix the MTF resampling to avoid lookahead and add a leakage-specific test.
2) Normalize the Ichimoku penalty to the same unit as `compute_hft_reward`.
3) Decide whether the trend penalty should be curriculum-gated or permanently on.
