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

move_if_exists() {
  local src="$1"
  local dst_dir="$2"
  [[ -e "$src" ]] || return 0
  ensure_dir "$dst_dir"
  run_step "Move $src -> $dst_dir/" mv -f "$src" "$dst_dir/"
}

move_dir_if_exists() {
  local src="$1"
  local dst="$2"
  [[ -d "$src" ]] || return 0
  ensure_dir "$(dirname "$dst")"
  run_step "Move dir $src -> $dst" mv "$src" "$dst"
}

echo "== Organize training data files =="
echo "DryRun: ${DRY_RUN:-false}"

# 1) Backups to archive
for f in data/*.bak; do
  [[ -e "$f" ]] || continue
  move_if_exists "$f" "data/archives/datasets/backups"
done

# 2) Legacy snapshots / one-off exports
for f in \
  data/btc_jpy_1m_dataset \
  data/btc_jpy_1m_dataset_pre_long \
  data/btc_jpy_1m_dataset_expanded.csv \
  data/btc_jpy_1m_latest_7d_20251213_155436.csv \
  data/btc_jpy_1m_latest_7d_20251215_073955.csv \
  data/btc_jpy_1m_latest_7d_20251215_074136.csv \
  data/btc_jpy_1m_yahoo_20251207_090329.csv \
  data/btc_jpy_yahoo_real_20251021.csv \
  data/btc_jpy_yahoo_real_20251021_corrected.csv \
  data/btc_jpy_yahoo_real_20251021_fixed.csv \
  data/btc_jpy_yahoo_real_20251021_fixed_featured.csv; do
  move_if_exists "$f" "data/datasets/legacy/root_snapshots"
done

# 3) Synthetic range datasets to dedicated legacy bucket
for f in \
  data/range_choppy.csv \
  data/range_choppy_featured.csv \
  data/range_medium.csv \
  data/range_medium_featured.csv \
  data/range_tight.csv \
  data/range_tight_featured.csv \
  data/range_wide.csv \
  data/range_wide_featured.csv; do
  move_if_exists "$f" "data/datasets/legacy/synthetic_ranges"
done

# 4) Debug/test one-off outputs
for f in \
  data/debug_sell_bias_output.csv \
  data/test_featured.csv \
  data/btc_jpy_15m_from_test_minute.csv \
  data/btc_jpy_5m_from_test_minute.csv; do
  move_if_exists "$f" "data/datasets/legacy/test_outputs"
done

# 5) Root-level backtest artifact file
move_if_exists "data/backtest" "data/archives/backtest"

# 6) Nested accidental dump directory
if [[ -d "data/data" ]]; then
  if [[ ! -e "data/archives/workspace_dump/data_data_legacy" ]]; then
    move_dir_if_exists "data/data" "data/archives/workspace_dump/data_data_legacy"
  else
    move_dir_if_exists "data/data" "data/archives/workspace_dump/data_data_legacy_$(date +%Y%m%d_%H%M%S)"
  fi
fi

echo "== Done =="
echo "Recommended checks:"
echo "  1) find data -maxdepth 1 -type f | sort"
echo "  2) find data/datasets/legacy -maxdepth 3 -type f | sort"
