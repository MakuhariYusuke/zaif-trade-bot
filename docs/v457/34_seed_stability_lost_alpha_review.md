# v457 Seed Stability + Lost Alpha Review (Docs 32-33)

Targets: `docs/v457/32_seed_stability_test.md`, `docs/v457/33_lost_alpha_recovery.md`

## Findings (ordered by severity)

1) Critical: The curriculum + Ichimoku wiring in doc 32 does not affect the current v457.4 training stack.
   - The v457.4 scripts build `FastIntradayEnvV456` via `create_fast_intraday_env_v456` and use `compute_hft_reward`,
     not `RewardCalculator`/`SignalIntegrator`.
   - Unless the training stack switches to the heavy environment or the signal-guidance logic is ported into
     `FastIntradayEnvV456`, the proposed curriculum updates will be a no-op.
   - Evidence: `scripts/v457/train_v457_4.py`, `ztb/trading/environment/fast_intraday_env_v456.py`,
     `ztb/trading/environment/components/calculators/reward_calculator.py`.

2) High: Cyclical time features are still zero-filled in the current environment.
   - Doc 33 claims cyclical features are computed from the DataFrame index, but `_build_observation` still fills zeros.
   - The env also `reset_index()` in `__init__`, so timestamp features are not available unless separately preserved.
   - Evidence: `ztb/trading/environment/fast_intraday_env_v456.py`.

3) High: MTF resampling is still not implemented in the factory.
   - Doc 33 claims true 5m/15m/1h resampling, but `calculate_mtf_features` still computes indicators on 1m data.
   - This keeps 27 MTF features highly redundant and does not deliver the proposed "bigger picture."
   - Evidence: `ztb/trading/environment/factory_v456.py`.

4) Medium: The verification script referenced in doc 33 is not present.
   - `test_fix_verification.py` is not in the repository, so the verification results are not reproducible.
   - This blocks independent validation of the MTF and cyclical fixes.

5) Medium: Signal guidance depends on feature name wiring that is not guaranteed in the v457.4 stack.
   - `SignalIntegrator` expects `feature_names` on config/env/observation_builder; otherwise it warns and skips.
   - Doc 32 assumes Ichimoku signals are available, but v457.4 feature wiring does not show this path.
   - Evidence: `ztb/trading/environment/components/signal_integrator.py`.

6) Low: Curriculum stage mapping may reduce guidance too early.
   - `_update_dynamic_weights` treats any stage beyond `balanced_transition` as "free market."
   - `BalanceCurriculumManager` includes `pnl_focused`, which might still need moderate guidance.
   - Evidence: `ztb/trading/environment/components/calculators/reward_calculator.py`,
     `ztb/trading/environment/components/reward/balance_curriculum.py`.

## Open Questions / Assumptions
- Are we staying on the `FastIntradayEnvV456` pipeline, or switching to the heavy env that uses `RewardCalculator`?
- Should Ichimoku guidance be gated by regime strength (e.g., only when trend strength / ADX is high)?
- Do we want the curriculum schedule to be purely automatic, or partially fixed by steps (seed stability focus)?
- Should guidance weights be scaled to avoid reward clipping (or should `reward_clip` be temporarily raised)?

## Recommended Next Steps
1) Decide the training stack.
   - If staying with `FastIntradayEnvV456`, port signal-guided reward logic into
     `compute_hft_reward` or a wrapper.
   - If switching to the heavy env, verify action mapping for 1D continuous actions.
2) Implement the cyclical feature integration in `FastIntradayEnvV456` with preserved timestamps.
3) Implement true MTF resampling in `EnvironmentFactory.calculate_mtf_features` and add a no-leak check
   (use right-closed resample + shift).
4) Re-run multi-seed stability tests after (2)-(3) are in place.
