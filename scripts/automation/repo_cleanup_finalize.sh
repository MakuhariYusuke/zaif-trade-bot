#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${1:-}"

run_step() {
  local msg="$1"
  shift
  echo "[STEP] $msg"
  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "       (dry-run)"
    return 0
  fi
  "$@"
}

ensure_dir() {
  [[ -d "$1" ]] || mkdir -p "$1"
}

merge_dir() {
  local src="$1"
  local dst="$2"
  [[ -d "$src" ]] || return 0
  ensure_dir "$dst"
  run_step "Merge $src -> $dst" rsync -a --remove-source-files "$src"/ "$dst"/
  find "$src" -type d -empty -delete || true
  [[ -d "$src" ]] && run_step "Remove empty $src" rm -rf "$src" || true
}

move_file_archived() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  ensure_dir "archived"
  run_step "Move $f -> archived/" mv -f "$f" archived/
}

echo "== Repo cleanup finalize =="
echo "DryRun: ${DRY_RUN:-false}"

# 1) root temporary files
while IFS= read -r file; do
  run_step "Delete root file $file" rm -f "$file"
done < <(find . -maxdepth 1 -type f \
  \( -name 'action_analysis_*' -o -name 'all_objects*' -o -name 'big_objects*' -o \
     -name 'blob_sizes_*' -o -name 'stash_diff*' -o -name 'syntax_*' -o -name 'temp_*' -o \
     -name 'test_d0_*' -o -name 'training_*_log*' -o -name 'training_*txt' -o \
     -name 'scan_*' -o -name 'test_*' -o -name 'training_*' -o \
     -name 'backtest_gate_log*' -o -name 'backtest_trades_*' -o -name 'test_synthetic_dataset*' -o \
     -name 'tmp_*.py' -o -name 'temp_*.py' \) -printf '%P\n')

# 2) move root scripts
for f in \
  alert_system.py circuit_breaker.py emergency_stop.py health_checker.py inspect_model.py \
  market_data_simulator.py paper_trading_manager.py performance_monitor.py performance_validator.py \
  real_time_metrics.py recovery_system.py result_comparator.py risk_based_allocator.py rollback_manager.py \
  sac.py virtual_portfolio_manager.py test_reward_simplified.py test_scale_verification.py \
  test_short_step_training.py; do
  move_file_archived "$f"
done

for pattern in analyze_*.py backtest_v45*.py debug_*.py diagnose_*.py; do
  for f in $pattern; do
    [[ -f "$f" ]] && move_file_archived "$f"
  done
done

# 3) merges
merge_dir analysis_results results/analysis
merge_dir backtest_results results/backtest
merge_dir backtest_analysis_plots results/backtest/plots
merge_dir experiment_plots results/experiments/plots
merge_dir optimization_results results/optimization
merge_dir phase3_comparison_results results/phase3
merge_dir statistical_sampling_results results/statistical
merge_dir test_backtest_results results/test_backtest
merge_dir test_results results/test
merge_dir training_results results/training
merge_dir coverage results/coverage

merge_dir test_checkpoints checkpoints/test
merge_dir test_checkpoints_phase2 checkpoints/test_phase2
merge_dir best_model checkpoints/best
merge_dir models checkpoints/models
merge_dir models_test checkpoints/models_test

merge_dir eval_logs logs/eval
merge_dir sac_action_test_logs logs/sac_action_test
merge_dir tensorboard logs/tensorboard

merge_dir backtest_experiments archived/backtest_experiments
merge_dir config configs
merge_dir schema configs/schema
merge_dir jsonschema configs/jsonschema
merge_dir _stable_baselines3_shim ztb/compat/sb3_shim
merge_dir utils ztb/utils
merge_dir websockets ztb/api/websockets
merge_dir venues ztb/api/venues
merge_dir python archived/python
merge_dir src archived/src

if [[ -d bundles ]]; then
  ensure_dir archived
  run_step "Move bundles -> archived/bundles_legacy" mv -f bundles archived/bundles_legacy
fi

# 4) remove dirs
for d in \
  .tmp .tmp-strategies .tmp-utils-stats .hypothesis .mypy_cache .ruff_cache .pytest_cache .benchmarks \
  htmlcov build zaif_trade_bot.egg-info __pycache__ node_modules venv venv311 venv311_new .venv311 \
  zaif-trade-bot-mirror git-filter-repo v435 temp_scripts temp_model stable_baselines3 sb3_contrib; do
  [[ -e "$d" ]] && run_step "Delete dir $d" rm -rf "$d"
done

echo "== Done =="
echo "Next:"
echo "  1) git status --short"
echo "  2) python -m pytest tests/ -x --timeout=60"
