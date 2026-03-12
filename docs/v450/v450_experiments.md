# v450 experiments

This document describes the `v450` experiments and configuration layout.

Purpose
- Validate dynamic thresholding (z-score-based with MAD fallback), curriculum `action_discovery`, and relative regime detection.

Directories
- `config/v450/` - base configuration and templates
- `experiments/v450/` - runnable experiment scripts

Recommended workflow
1. Generate Synthetic Data
    - `python experiments/v450/generate_range_data_v450.py`
2. Quick Validation
    - `python experiments/v450/run_short_training_v450.py`
3. A/B Test Thresholds
    - `python experiments/v450/run_ab_test_threshold_v450.py`
4. Range Tests
    - Generate synthetic range data with `python experiments/v450/generate_range_data_v450.py`
    - Run range-based v450 experiments using `python experiments/v450/run_range_tests_v450.py` (features and training)

Configuration Keys of interest (v450 specifics)
- `dynamic_threshold_mode` (fixed | volatility | z_score)
- `z_score_window` (history length used for z-score calculation)
- `z_score_threshold` (threshold value for dynamic z-based adaptation)
- `z_score_method` (std | mad)
- `regime_detection_config.use_relative` (switch to percentile-based relative regime detection)
- `regime_detection_config.reference_window` (size of reference window for percentile computation)
- `action_discovery` reward config (`scale`, `magnitude_bonus`, `direction_bonus`)

Notes & Tips
- Quick validation uses `UnifiedTrainer` and synthetic datasets by default if `data/` file is not found.
- For faster iteration, reduce `total_timesteps` and `buffer_size` in `sac_hyperparameters`.
- When running A/B tests, use `ztb.utils.parallel_experiments` to concurrently run multiple configs; check `experiments/v450/run_ab_test_threshold_v450.py` for a runnable example.
