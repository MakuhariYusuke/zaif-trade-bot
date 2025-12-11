# MTF Optimizer — How to use

This file explains how to run the MTF optimizer locally and what the CI/nightly expectations are.

## Quick start (local)

1. Create or modify `config/v448/templates/mtf_optimizer_template.json` to your desired search parameters.

2. Run the quick MTF optimizer (example; TBD once `MTFOptimizer` is implemented):

```bash
python -m ztb.training.reward_function_optimizer.mtf_optimizer \
  --config config/v448/templates/mtf_optimizer_template.json \
  --seeds 3 --timesteps 2000 --candidates 10 --iterations 3
```

3. After runs complete, aggregate reports:

```bash
python tools/ci/evaluate_training_runs.py --out reports/mtf_optimizer_summary.json
```

## CI Nightly
- The CI `training-eval` job will run the optimizer nightly with `--seeds 3` and `--timesteps 4000`.
- Artifacts: `reports/` (individual training reports), `reports/mtf_optimizer_summary.json`, and logs will be uploaded.
 - The `tools/training/confirm_candidate.py` helper will be used to prefilter, verify top-N candidates, and optionally apply candidates.
   - It writes `reports/mtf_optimizer_summary.json` that CI or ops can use for gating and forensic purposes.
   - The helper also persists `reports/applied_candidate_<candidate_id>.json` when a candidate is applied to the runtime `MTFWeightManager`.

## Acceptance & Gating
- The MTF optimizer will store the best candidate in `best_model/` if composite score >= threshold.
- Gate conditions are configurable in `ci.yml` and `docs/SAC_v448_LAYER6_DESIGN_SPEC.md`.
 - Gate conditions are configurable and include a new `--min-reports` gate to ensure consistent multi-seed coverage (prefer `--min-reports == seeds`).

## Notes
- Prioritize Layer 4 stabilization before enabling solver-based optimizers in CI.
- Keep `ZTB_FORCE_TORCH_STUB` use to a minimum—prefer CPU-only PyTorch wheel in CI for reproducibility.

## Scheduler
- You can use the `run_mtf_scheduler.py` script to run a one-off scheduler (dry-run or apply candidate weights):

```bash
python tools/training/run_mtf_scheduler.py --config config/v448/templates/mtf_optimizer_template.json --dry-run
```

This will run the optimizer in dry-run mode (no real training) and optionally apply the best candidate's weights to the `MTFWeightManager` if not in dry-run mode.

### MTFScheduler

Use `ztb.training.reward_function_optimizer.mtf_scheduler.MTFScheduler` to register periodic or stage-change callbacks into the curriculum manager and apply candidates autonomously. The scheduler writes an `applied_candidate_<id>.json` file into `reports/` when a candidate is applied and will log failures when `MTFWeightManager.set_weights()` returns False.

Example:
```bash
python tools/training/run_mtf_scheduler.py --config config/v448/templates/mtf_optimizer_template.json --dry-run
```

Note: `BalanceCurriculumManager` calls registered listeners using **keyword args** (kwargs) and the event includes `previous_stage`, `new_stage`, `step`, and `emergency`. When creating callbacks with `MTFScheduler.create_stage_change_callback()`, the callback reads `new_stage` to decide whether to run (e.g., `kwargs.get('new_stage')`).

#### Confirm Candidate CLI (Two-Stage Adoption)

Use `tools/training/confirm_candidate.py` for a safer two-stage adoption flow (useful in CI/nightly):

1) Prefilter: short AB-runs to shortlist candidates
2) Verify: longer AB-runs for the top-N candidates
3) Gate: optional invocation of `tools/ci/check_optimizer_gates.py` to fail if gating thresholds not satisfied
4) Apply: when gated and `--apply` is used, write `reports/applied_candidate_<id>.json` with candidate metadata and telemetry

Example (local):
```bash
python tools/training/confirm_candidate.py --config config/v448/templates/mtf_optimizer_template.json \
  --candidates 10 --prefilter-seeds 1 --verify-seeds 3 --verify-timesteps 2000 --top-n 3 --gate-sharpe 0.5 \
  --gate-return 0.05 --min-reports 3 --apply
```

This flow uses the same gates as `tools/ci/check_optimizer_gates.py` and writes artifacts into `reports/` for auditing and CI artifact upload.
