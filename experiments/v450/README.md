# v450 experiments

This folder contains short-run experiments and utilities for v450, focused on:

- dynamic thresholding modes (fixed / volatility-based / z-score)
- action_discovery curriculum stage
- relative market regime detection
- quick A/B comparisons and synthetic data generation

Scripts
- run_short_training_v450.py: Quick validation run using `action_discovery` and z-score dynamic thresholds
- run_ab_test_threshold_v450.py: A/B test comparing fixed, volatility-based, and z_score dynamic thresholds using the parallel runner
- generate_range_data_v450.py: Generate synthetic datasets for range-bound, medium, and wide volatility scenarios

Config files
- Use `config/v450/base/config.yaml` or `config/v450/templates/sac_v450_template.json` as a baseline for experiments.

Notes
- These scripts are lightweight by design (short training) for iterative development and local debugging. Use `ztb.utils.parallel_experiments.run_parallel_experiments` for more thorough batched/backtest runs.
