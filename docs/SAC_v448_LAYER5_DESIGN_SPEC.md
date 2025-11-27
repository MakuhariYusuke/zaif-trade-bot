# SAC v448 Layer 5 Design Specification: Curriculum & MTF Optimization

## Overview
This document describes the design for Layer 5 features in SAC v448: the Curriculum Learning foundations and Multi-Timeframe (MTF) weight optimization. Focus is on ensuring stable policy learning for 1-minute timeframe training and to provide modular architecture for continued research.

Goals:
- Build on Layer 4's Trend-aware Balance improvements.
- Provide modular, testable components to tune and optimize multi-timeframe feature weights safely.
- Ensure curriculum progression is robust and can revert in emergencies.
- Provide interfaces for reward shaping and a candidate MTF optimizer that can be toggled in or out.

## Scope & Out-of-scope
Included:
- BalanceCurriculumManager (continued integration & evolution)
- MTFWeightManager (new): optimizer for feature weights across timeframes
- Integration into RewardCalculator and BehavioralPenaltyCalculator
- Config keys & defaults, tests and validation plan

Excluded:
- Full ablation studies; those will be executed through AB-run tools (outside this spec)
- GPU-specific optimizations or PyTorch GPU-only pipelines

## Key Components & Interfaces

1. BalanceCurriculumManager (Layer 3 -> Layer 5 evolution)
- Public API:
  - update(step, action_counts, recent_rewards, portfolio_values) -> status
  - get_current_stage() -> str
  - get_stage_config() -> dict
  - reset()
- Responsibilities:
  - Maintain stage state and transitions
  - Signal emergency revert to forced_balance
  - Provide stage-specific runtime configs to RewardCalculator

2. MTFWeightManager (New)
- Purpose: Tune `feature_weights` (e.g., {"1min": 0.30, "5min": 0.55, "15min": 0.15}) based on observed validation metrics.
- Interface:
  - get_weights() -> Dict[str,float]
  - update(step, metrics) -> None
  - reset() -> None
- Responsibilities:
  - Provide safe updates to weights constrained within config-specified min/max values.
  - Optionally perform offline candidate weight simulations (if enabled) using quick in-memory AB tests.
  - Expose methods and telemetry for CI to sample weights in quick jobs.

3. RewardCalculator (Integration)
- Integrates with: TrendDetector, BalanceCurriculumManager, MTFWeightManager
- New responsibilities:
  - Request `weights = mtf_manager.get_weights()` and blend features accordingly
  - Use curriculum stage config to adjust reward shaping dynamically

4. BehavioralPenaltyCalculator
- Receives `trend_signal` and uses `mtf_weight_adjustments` (if any) to influence balance targets
- Continues to use `consistency_min_actions`, `lookback+1` semantics, and `_get_recent_counts()` for counts

## Configuration
- New keys under `behavior` / `behavior_optimization` or `curriculum_learning`:
  - `mtf.weight_optimizer.enabled` (bool, default false)
  - `mtf.weight_optimizer.allowed_timeframes` (list, default ["1min","5min","15min"]) 
  - `mtf.weight_optimizer.min_weight_by_tf` (dict) and `max_weight_by_tf` (dict)
  - `mtf.weight_optimizer.learning_rate` (float)
  - `curriculum.stage_durations` (dict)
  - `curriculum.emergency_revert_threshold` (float, e.g., 0.35)

Default recommended `feature_weights` for 1-min setups:
- 1min: 0.30
- 5min: 0.55
- 15min: 0.15

## Acceptance Criteria
- Unit tests for updates and transitions exist and pass.
- Quick AB-run (3 seeds × 1000 steps) with `mtf.weight_optimizer.enabled: False` and `True` yields no bias collapse regressions.
- CI smoke test includes `run_child_trainer_wrapper.py --diagnostics-only` and `tools/training/ab_test_runner.py` quick-run for Layer 5 candidate configs.

## Test Plan
A. Unit Tests
- BalanceCurriculumManager: transitions, emergency revert, persisted stage configs
- MTFWeightManager: safe weight updates, min/max enforcement, reset behavior
- RewardCalculator integration test: receives stage config and weights, updates shaping calculations
- BehavioralPenaltyCalculator: ensures that `trend_signal` combined with varying weights adjusts targets correctly

B. Integration Tests (Quick/CI)
- `tools/training/ab_test_runner.py` quick run (3 seeds × 1000 steps) to validate stability and bias
- `run_child_trainer_wrapper.py --diagnostics-only` for import safety

## Migration Plan
- A detailed migration for existing configs will be handled via `ztb.utils.v4xx_config_converter`.
- Add `mtf` default values into the v448 config templates and document recommended values.

## Risk & Mitigations
- Risk: MTF optimizer may introduce unstable weight oscillations
  - Mitigation: Keep optimizer disabled by default; support a `test_mode` with shorter run durations and limited update frequency.
- Risk: Curriculum automation could inadvertently remove exploration too early
  - Mitigation: Add `min_exploration_steps` and tune thresholds with AB runs.

## Next Steps (Implementation Tasks)
- Implement `ztb/trading/environment/components/reward/mtf_weight_manager.py` with safe update logic.
- Add tests in `tests/unit/training/curriculum/` and `tests/unit/training/mtf/`.
- Integrate `MTFWeightManager` into `RewardCalculator` and `BehavioralPenaltyCalculator` and add integration tests.
- Add docs and CI smoke tests for quick runs.

## Implementation Sequence & Checklist (Recommended)
This is the order we will implement Layer 5 features to minimize risk and enable incremental validation.

1. (Complete) Add the safe MTF manager stub with min/max constraints and get_weights API
  - Implemented as `MTFWeightManager` with conservative no-op update.
  - Add unit tests for basic retrieval.

2. (Complete) Conservative optimizer update logic
  - Implement `MTFWeightManager.update()` with alpha smoothing, min/max enforcement, and normalization.
  - Add unit tests verifying updates and constraints.

3. (Complete) Telemetry & integration with RewardCalculator
  - Integrate manager into `RewardCalculator` and expose mtf_weights in `last_reward_components`.
  - Add unit/integration tests ensuring `mtf_weights` propagate and impact is observable.

4. (Complete) Add AB-run smoke wrapper and sample config
  - Add `tools/training/run_quick_mtf_ab.py` and `config/v448/mtf_mini_test.json` for quick validation.

5. (Short-term) Add CI quick-run job (3 seeds x 1000 steps) and ensure acceptance criteria
  - CI job toggles `mtf.weight_optimizer.enabled` True/False for AB comparison.
  - Add telemetry/metrics acceptance checks to pass/fail if bias collapse reappears.

6. (Medium-term) Implement a small offline optimizer or optimizer scheduler
  - Provide safe fallback, test with more AB-run variants and schedule (e.g., small trials per N steps).

7. (Long-term) Research & advanced optimizer
  - Bayesian or multi-armed bandit selection for MTF weights, longer AB-run testing.


## Notes
- Add telemetry for `mtf` updates: logs and structured logger fields `mtf_weights`, `mtf_update_step`, and `curriculum_stage`.
- Document expected behavior in `docs/SAC_v448_DEVELOPMENT_PLAN.md` and `docs/SAC_v448_IMPLEMENTATION_ROADMAP.md`.


---
*Created: 2025-11-27*