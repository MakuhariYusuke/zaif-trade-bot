# v457 Index

## Documents
- docs/v457/00_V457_RESET_PLAYBOOK_20260116.md
- docs/v457/01_legacy_asset_analysis.md
- docs/v457/02_plan_review_prompt.md
- docs/v457/03_plan_review.md
- docs/v457/04_second_opinion_prompt.md
- docs/v457/05_second_opinion.md
- docs/v457/06_initial_validation_log.md
- docs/v457/07_initial_validation_review.md
- docs/v457/08_hold_freeze_analysis.md
- docs/v457/09_data_source_cleanup.md
- docs/v457/10_hold_freeze_review.md
- docs/v457/11_training_results_analysis.md
- docs/v457/12_training_results_followup.md
- docs/v457/13_training_results_success.md
- docs/v457/14_training_results_review.md
- docs/v457/15_short_term_profit_strategy_v458_concept.md
- docs/v457/16_short_term_profit_strategy_v458_review.md
- docs/v457/17_v457_enhancement_roadmap.md
- docs/v457/18_v458_strategy_roadmap_review.md
- docs/v457/19_v458_grid_search_results.md
- docs/v457/20_v458_grid_search_review.md
- docs/v457/21_v457_1_phase2_frequency_control.md
- docs/v457/22_v457_1_phase2_frequency_control_review.md
- docs/v457/23_v457_2_strategy_plan.md
- docs/v457/24_v457_2_strategy_plan_review.md
- docs/v457/25_v457_2_analysis.md
- docs/v457/26_v457_2_analysis_review.md
- docs/v457/27_v457_3_analysis.md
- docs/v457/28_v457_3_analysis_review.md
- docs/v457/29_v457_4_implementation.md
- docs/v457/30_v457_4_verification.md
- docs/v457/31_v457_4_verification_review.md
- docs/v457/32_seed_stability_test.md

## Configs
- config/v457/README.md
- config/v457/features.yaml
- config/v457/base/
- config/v457/experiments/
- config/v457/templates/

## Scripts (v457)
- scripts/v457/train.py
- scripts/v457/train_v457_2.py
- scripts/v457/train_v457_3.py
- scripts/v457/train_v457_4.py
- scripts/v457/backtest.py
- scripts/v457/backtest_v457.py
- scripts/v457/verify_stability.py
- scripts/v457/run_parallel_training.py

## Metrics (results_utils)
- backtest_metrics.json metrics: net_pnl, gross_pnl, total_fees, total_slippage
- action metrics: action_distribution, avg_abs_action, ttl_action_distribution, avg_ttl_action
- TTL metrics: ttl_forced_exits, cooldown_triggers, ttl_enabled
- reproducibility: seed, start_index, action_space_type, baseline_mode
