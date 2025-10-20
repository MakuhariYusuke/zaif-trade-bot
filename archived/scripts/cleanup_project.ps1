# Project Cleanup and Reorganization Plan
# Date: 2025-10-08
# Purpose: Archive old models, reorganize root Python files, optimize configs

## Phase 1: Create Archive Directories
New-Item -ItemType Directory -Force -Path "models\archived"
New-Item -ItemType Directory -Force -Path "checkpoints\archived"
New-Item -ItemType Directory -Force -Path "scripts\analysis"
New-Item -ItemType Directory -Force -Path "scripts\data"
New-Item -ItemType Directory -Force -Path "scripts\validation"
New-Item -ItemType Directory -Force -Path "scripts\debug"

## Phase 2: Archive Old Models (keeping only essential ones)
Write-Host "Archiving old training models..."

# Move old hyperparameter search results
Move-Item "models\aggressive_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\batch_size_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\clip_range_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\curriculum_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\ent_coef_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\gae_lambda_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\gamma_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\lagrange_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\learning_rate_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\lr_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\max_grad_norm_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\max_position_size_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\n_steps_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\progress_bar_test.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\reward_params_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\reward_scaling_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\risk_free_rate_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\scalping_15s_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\sell_mitigation_test.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\target_kl_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\test_high_entropy_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\transaction_cost_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue
Move-Item "models\vf_coef_test_*.zip" "models\archived\" -ErrorAction SilentlyContinue

## Phase 3: Archive Old Checkpoints (keeping only recent/active ones)
Write-Host "Archiving old training checkpoints..."

# Move old scalping experiments
Move-Item "checkpoints\scalping-10k-*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\scalping-fixed-*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\scalping-prod-*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\scalping-test-*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\scalping_15s_aggressive_*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\scalping_15s_balance_*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\scalping_15s_ultra_*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\scalping_full_*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\scalping_iterative_*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\scalping_training_*" "checkpoints\archived\" -ErrorAction SilentlyContinue

# Move old iterative experiments
Move-Item "checkpoints\iterative_*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\normal_1m_*" "checkpoints\archived\" -ErrorAction SilentlyContinue

# Move old smoke tests
Move-Item "checkpoints\smoke_test_*" "checkpoints\archived\" -ErrorAction SilentlyContinue

# Move old test runs
Move-Item "checkpoints\test-*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\train-*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\training-*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\trading_*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\unified_training_session" "checkpoints\archived\" -ErrorAction SilentlyContinue

# Move old mlreinforcement experiments
Move-Item "checkpoints\mlreinforcement*" "checkpoints\archived\" -ErrorAction SilentlyContinue

# Move debug/failsafe dumps
Move-Item "checkpoints\debug_*" "checkpoints\archived\" -ErrorAction SilentlyContinue
Move-Item "checkpoints\failsafe_dump" "checkpoints\archived\" -ErrorAction SilentlyContinue

## Phase 4: Reorganize Root Python Files
Write-Host "Reorganizing root Python files..."

# Move analysis scripts
Move-Item "ablation_study.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "analyze_data.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "analyze_duplicates.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "analyze_features.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "analyze_file_sizes.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "analyze_sell_bias.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "analyze_test_duplicates.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "analyze_training_logs.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "benchmark_checkpoint_compression.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "benchmark_comparison.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "benchmark_compression.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "benchmark_features.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "benchmark_memory_monitoring.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "compare_models.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "comprehensive_benchmark.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "feature_analysis.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "feature_importance.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "performance_attribution.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "regime_evaluation.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "risk_parity_analysis.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "strategy_robustness.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "trade_analysis.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "walkforward_analysis.py" "scripts\analysis\" -ErrorAction SilentlyContinue

# Move data generation scripts
Move-Item "create_backtest.py" "scripts\data\" -ErrorAction SilentlyContinue
Move-Item "create_test_dataset.py" "scripts\data\" -ErrorAction SilentlyContinue
Move-Item "create_test_file.py" "scripts\data\" -ErrorAction SilentlyContinue
Move-Item "fetch_real_btc_data.py" "scripts\data\" -ErrorAction SilentlyContinue
Move-Item "fetch_yahoo_btc_data.py" "scripts\data\" -ErrorAction SilentlyContinue
Move-Item "generate_enhanced_training_data.py" "scripts\data\" -ErrorAction SilentlyContinue

# Move validation scripts
Move-Item "validate_model_behavior.py" "scripts\validation\" -ErrorAction SilentlyContinue
Move-Item "check_features.py" "scripts\validation\" -ErrorAction SilentlyContinue
Move-Item "check_file.py" "scripts\validation\" -ErrorAction SilentlyContinue
Move-Item "check_imports.py" "scripts\validation\" -ErrorAction SilentlyContinue
Move-Item "check_models.py" "scripts\validation\" -ErrorAction SilentlyContinue
Move-Item "check_model_meta.py" "scripts\validation\" -ErrorAction SilentlyContinue

# Move debug scripts
Move-Item "debug_action_masking.py" "scripts\debug\" -ErrorAction SilentlyContinue
Move-Item "debug_deep_sell_bias.py" "scripts\debug\" -ErrorAction SilentlyContinue
Move-Item "debug_env.py" "scripts\debug\" -ErrorAction SilentlyContinue
Move-Item "debug_model_predictions.py" "scripts\debug\" -ErrorAction SilentlyContinue
Move-Item "debug_sell_action.py" "scripts\debug\" -ErrorAction SilentlyContinue
Move-Item "debug_tiebreaker.py" "scripts\debug\" -ErrorAction SilentlyContinue
Move-Item "monitor_memory.py" "scripts\debug\" -ErrorAction SilentlyContinue

# Remove temporary/obsolete files
Remove-Item "temp*.py" -ErrorAction SilentlyContinue
Remove-Item "fix_*.py" -ErrorAction SilentlyContinue
Remove-Item "clean_unified.py" -ErrorAction SilentlyContinue
Remove-Item "remove_callback.py" -ErrorAction SilentlyContinue
Remove-Item "investigate_sb3.py" -ErrorAction SilentlyContinue

# Move statistical/simulation scripts
Move-Item "bootstrap_confidence.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "cost_sensitivity.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "drawdown_recovery.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "monte_carlo_simulation.py" "scripts\analysis\" -ErrorAction SilentlyContinue
Move-Item "stress_test.py" "scripts\validation\" -ErrorAction SilentlyContinue

# Move test files (should be in tests/)
Move-Item "test_*.py" "tests\manual\" -ErrorAction SilentlyContinue
Move-Item "simple_backtest.py" "scripts\analysis\" -ErrorAction SilentlyContinue

# Move utility scripts
Move-Item "demo_auto_stop.py" "scripts\debug\" -ErrorAction SilentlyContinue
Move-Item "quick_training_test.py" "scripts\debug\" -ErrorAction SilentlyContinue
Move-Item "train_and_backtest_optimized.py" "scripts\training\" -ErrorAction SilentlyContinue

Write-Host "✅ Cleanup complete!"
Write-Host ""
Write-Host "Summary:"
Write-Host "  - Old models archived to: models\archived\"
Write-Host "  - Old checkpoints archived to: checkpoints\archived\"
Write-Host "  - Scripts reorganized to: scripts\{analysis,data,validation,debug}\"
Write-Host "  - Temporary files removed"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Review ppo_balanced_mem_optimized.json"
Write-Host "  2. Apply latest best practices to config files"
Write-Host "  3. Start training with clean environment"
