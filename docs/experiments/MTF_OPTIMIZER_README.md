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

## Acceptance & Gating
- The MTF optimizer will store the best candidate in `best_model/` if composite score >= threshold.
- Gate conditions are configurable in `ci.yml` and `docs/SAC_v448_LAYER6_DESIGN_SPEC.md`.

## Notes
- Prioritize Layer 4 stabilization before enabling solver-based optimizers in CI.
- Keep `ZTB_FORCE_TORCH_STUB` use to a minimum—prefer CPU-only PyTorch wheel in CI for reproducibility.

## Scheduler
- You can use the `run_mtf_scheduler.py` script to run a one-off scheduler (dry-run or apply candidate weights):

```bash
python tools/training/run_mtf_scheduler.py --config config/v448/templates/mtf_optimizer_template.json --dry-run
```

This will run the optimizer in dry-run mode (no real training) and optionally apply the best candidate's weights to the `MTFWeightManager` if not in dry-run mode.
