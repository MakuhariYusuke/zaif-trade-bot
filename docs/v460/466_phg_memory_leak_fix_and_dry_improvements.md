# 466# メモリリーク修正・DRY改善・ConfigMap集約

> 日時: 2025-01  
> 前提: 465# (c52ca2deb) モデル退化ガード実装後のコード品質向上

## 概要

465# セルフレビューで特定したメモリリーク・DRY違反・型重複を修正。

## 変更一覧

### 1. メモリリーク修正 (`retrain_scheduler.py`)

#### 1a. `_evaluate_wf_multi` ループ内リーク

**問題**: per-window ループで `lgbm_model`, `imputer`, `scaler`, `X_train_sc`, `X_test_sc`, `X_train`, `y_train`, `X_val`, `y_val`, `X_test` が蓄積。M ウィンドウなら O(M) 個のモデル + O(M×N) の scaled array が同時にメモリ上に存在。

**修正**: ループ末尾に `del` を追加し各ウィンドウのオブジェクトを即時解放。

#### 1b. `retrain_model` 本体の中間 DataFrame

**問題**: `pnl_data`, `X_base`, `X_full`, `enriched`, `X_sc`, `X_imp` が関数終了まで保持。

**修正**: 各変数を最終参照後に `del` で即時解放:
- `del pnl_data` — len チェック後
- `del X_base` — y_target 抽出後
- `del X_full` — ドロップ統計記録後
- `del enriched` — WF eval 完了後
- `del X_sc, X_imp` — Pipeline 構築後

#### 1c. `import lightgbm as lgb` 遅延化

**問題**: `retrain_model` の training path 入口で `import lightgbm as lgb` を行っていたが、`lgb` は `early_stop > 0` の場合（callbacks 生成）のみ使用。lightgbm 未インストール環境で D1/D2 テストが不必要に失敗。

**修正**: import を `if early_stop > 0:` ブロック内に移動。

### 2. ConfigMap 集約 (`ztb/utils/types.py`)

**問題**: `ConfigMap = dict[str, object]` が 8 ファイルで重複定義:
- `retrain_scheduler.py`
- `v4xx_config_converter.py`
- `sac_algorithm.py`
- `reward_function_optimizer.py`
- `parameter_space.py`
- `optimization_engine.py`
- `evaluation_engine.py`
- `config_builder.py`

**修正**: `ztb/utils/types.py` に集約し、全ファイルを `from ztb.utils.types import ConfigMap` に統一。

### 3. DRY改善: `_resolve_early_stopping` ヘルパー

**問題**: `early_stopping_rounds` と `n_estimators` の解決ロジックが3箇所で重複:
- `_evaluate_wf_multi` (l.657)
- `_evaluate_wf_single` (l.810)
- `retrain_model` Step 5 (l.1614)

**修正**: `_resolve_early_stopping(cfg) -> (int, int)` ヘルパーに統合。

### 4. テスト修正

D1/D2 テストの `_base_cfg` に `early_stopping_rounds: 0` を追加。lightgbm 未インストール環境でも D1/D2 ガードテストが正常に動作するように。

## テスト結果

- `test_retrain_hot_reload.py`: 64 passed, 22 skipped (lightgbm 未インストール分)
- v460 全体: 3674 passed, 9 skipped

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/ml/retrain_scheduler.py` | メモリリーク修正、DRY改善、ConfigMap import |
| `ztb/utils/types.py` | `ConfigMap` 追加 |
| `ztb/utils/v4xx_config_converter.py` | ConfigMap import 統一 |
| `ztb/training/algorithms/sac/sac_algorithm.py` | ConfigMap import 統一 |
| `ztb/training/reward_function_optimizer/reward_function_optimizer.py` | ConfigMap import 統一 |
| `ztb/training/reward_function_optimizer/parameter_space.py` | ConfigMap import 統一 |
| `ztb/training/reward_function_optimizer/components/optimization_engine.py` | ConfigMap import 統一 |
| `ztb/training/reward_function_optimizer/components/evaluation_engine.py` | ConfigMap import 統一 |
| `ztb/training/core/config_builder.py` | ConfigMap import 統一 |
| `tests/unit/v460/test_retrain_hot_reload.py` | D1/D2 テスト設定修正 |
| `docs/v460/466_phg_memory_leak_fix_and_dry_improvements.md` | 本ドキュメント |
