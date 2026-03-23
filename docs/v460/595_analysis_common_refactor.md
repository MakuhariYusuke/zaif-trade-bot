# 595# 分析スクリプト共通モジュール抽出 (analysis_common)

## 概要
15本の分析スクリプト（`scripts/v460/analysis/`、合計~6,396行）に散在していた
重複コード（~36%）を共通モジュール `analysis_common.py` に集約し、
先行4スクリプトをマイグレーションした。

## 背景
| 項目 | 重複箇所 | 影響スクリプト数 |
|------|----------|------------------|
| CLI `--results-dir / --date-from / --date-to` | 6〜10箇所 | 15 |
| `_get_pnl()` (fallback chain)  | 3変種 | 8+ |
| `AS_THRESHOLD_BPS / SEVERE_AS_THRESHOLD_BPS` | 定数定義 | 5+ |
| `_extract_filled() / _pnl_array()` | ヘルパー | 4+ |
| `_record_to_utc_hour()` | タイムスタンプ変換 | 3+ |

### ztb 既存ヘルパー調査結果
- **使える**: `ztb.utils.safety.safe_to_finite()` (9スクリプトで使用)、`ztb.metrics.fill_quality` (FillRecord, load/iter/filter)
- **使えない**: `analysis_formatters.py` (SAC/バックテスト向け、重複定義あり)、`analysis_errors.py` (未使用)、`analysis_utils.py` (CSV/DataFrame向け)

## 新規作成ファイル

### `scripts/v460/analysis/analysis_common.py` (~270行)
分析スクリプト専用の共通モジュール。`ztb.metrics.fill_quality` と `ztb.utils.safety` へ委譲。

| カテゴリ | エクスポート |
|----------|-------------|
| 型 | `Record`, `FloatArray` |
| 定数 | `DEFAULT_RESULTS_DIR`, `AS_THRESHOLD_BPS`, `SEVERE_AS_THRESHOLD_BPS`, `PNL_FIELD_PRIORITY` |
| CLI builder | `add_common_filter_args()`, `add_side_regime_args()`, `add_output_args()` |
| データ読込 | `load_and_filter_records(include_emergency=True)`, `load_records_from_args()` |
| PnL抽出 | `get_pnl()`, `extract_pnl_array()`, `extract_pnl_list()` |
| フィルタ | `extract_filled()` |
| 時刻変換 | `record_to_utc_hour()` |
| 出力 | `write_output()`, `write_json_output()` |

### `tests/unit/analysis/test_analysis_common.py` (~290行)
42テスト（39 pass / 3 skip）

## マイグレーション済みスクリプト (4/15)

### 1. `tail_loss_analysis.py` (~60行削減)
- 削除: `_extract_filled`, `_pnl_array`, `_record_to_utc_hour`, `_DEFAULT_RESULTS_DIR`, `sys.path.insert` hack
- `load_and_filter_records(include_emergency=False)` を使用

### 2. `analyze_fill_logs.py` (~35行削減)
- 削除: `load_records()`, `apply_filters()`, `_pnls` 本体
- `--data-dir` は後方互換性のため維持
- `write_output()` / `write_json_output()` で出力統一

### 3. `sha_comparison.py` (~20行削減)
- 削除: `_get_pnl()`, `AS_THRESHOLD_BPS`, `SEVERE_AS_THRESHOLD_BPS`, `safe_to_finite` import
- 全 `_get_pnl(` → `get_pnl(` 一括置換

### 4. `hour_matched_comparison.py` (~10行削減)
- 削除: `_get_pnl()`, `safe_to_finite` import
- 全 `_get_pnl(` → `get_pnl(` 一括置換

## テスト修正

### `tests/test_analyze_fill_logs.py`
- `apply_filters` / `load_records` import → `load_and_filter_records` に変更
- `TestApplyFilters` を `apply_fill_record_filters` 直接呼び出しに書き換え
- `test_date_filter_file` 削除（sys.exit テスト）

### `tests/v460/test_346_tail_loss_analysis.py`
- `_extract_filled` / `_pnl_array` / `_record_to_utc_hour` → `analysis_common` 経由 (`as` alias)

## テスト結果
- 分析テスト: **92 passed, 4 skipped** ✅
- 全体: **3024 passed, 1 failed (既知・無関係), 127 skipped** ✅
- E2E: `analyze_fill_logs --date-from 2026-03-23 --date-to 2026-03-24` → 341件, 128 filled, exit 0 ✅

## 未マイグレーション (11スクリプト)
`stopgap_daily_report`, `side_regime_dashboard`, `compare_regime_ab`,
`hindsight_filter`, `reproduce_152_metrics`, `oracle_baseline`, `oracle_test`,
`ab_offset_comparison`, `vg_and_trend` 等 — 優先度に応じて順次対応。
