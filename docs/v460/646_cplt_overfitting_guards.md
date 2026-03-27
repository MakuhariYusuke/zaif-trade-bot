# 646# 過学習防止ガード追加

## 背景
645# で退化した sell モデル (n=229, 300木, 定数出力 24%) を無効化したが、
過学習の根本原因はトレーニングパイプライン自体にあった。

### 根本原因分析
1. **n_estimators がサンプル数に対して過大**: `early_stopping_rounds=20` 有効時に `lgbm_n_estimators_max=300` が使用され、229 サンプルに対して 300 木を訓練
2. **Early stopping の val set が小さすぎ**: 229 × 0.2 = 46 サンプルでは過学習の検出精度が低い
3. **D2 ガード (pred_std) が訓練データのみで検証**: 過学習モデルは訓練データで分散があるが、未知データで定数出力になる
4. **`side_min_samples=50` が低すぎ**: 50 サンプルでのモデル学習は危険

## 変更内容

### P0-A: サンプル数ベース n_estimators 上限
- **ファイル**: `scripts/v460/ml/retrain_scheduler.py` (`_resolve_early_stopping`)
- **ロジック**: `n_est = min(n_est, max(30, n_samples // 2))`
- **効果**: n=229 → 300木 が 114木 に制限 (比率 2.0+)
- **適用箇所**: 最終学習 + WF eval (multi/single) 全3箇所

### P0-B: D2 ガードの診断フィールド追加
- **ファイル**: `scripts/v460/ml/retrain_scheduler.py` (D2 ガード部)
- **ゲート判定**: 全訓練データの `pred_std` を使用 (変更なし)
- **追加**: eval_set (early stopping val split) の `pred_std_val` を診断フィールドとして記録
- **注意**: eval_set は LightGBM モニタリング専用で訓練データから除外されない (真の OOS ではない)
- **新フィールド**: `pred_std_val`, `pred_std_ratio` (val/train の分散比率、将来の異常検知指標)

### P0-C: side_min_samples 引き上げ
- **変更前**: `side_min_samples: 50`
- **変更後**: `side_min_samples: 200`
- **変更箇所**: デフォルト設定 + YAML (`configs/v460/fill_test.yaml`)
- **根拠**: 229 サンプルで 300 木は過学習必至。200 なら n_est_cap = 100 木で穏当

### YAML 変更
| キー | Before | After | 理由 |
|------|--------|-------|------|
| `side_min_samples` | 50 | 200 | 過学習防止に必要な最低サンプル数 |

## テスト
- `tests/unit/v460/test_646_overfitting_guards.py`: 13 テスト全パス
  - `TestResolveEarlyStopping`: 9 テスト (cap 適用、下限保証、大サンプル無cap、None 後方互換、229 defect case)
  - `TestSideMinSamplesDefault`: 2 テスト (デフォルト値確認)
  - `TestPredStdOOS`: 2 テスト (OOS フィールド存在、eval_set 使用確認)
- v460 全体: 4163+ passed, 0 failed

## 定量的効果 (229 サンプルの売りモデルの場合)
| 指標 | Before | After |
|------|--------|-------|
| n_estimators | 300 | 114 (cap) |
| samples/trees 比率 | 0.76 | 2.01 |
| side_min_samples | 50 (通過) | 200 (棄却 → 統一モデルにフォールバック) |
| D2 検証データ | 訓練データ | 訓練データ (+ val split 診断) |

## セルフレビュー所見

### FINDING-1 (修正済): eval_set ≠ held-out
LightGBM `fit(X, y, eval_set=[(X_val, y_val)])` は `X` 全体で訓練し、eval_set は
モニタリング専用。初版で eval_set を "OOS" と称して D2 ゲート判定に使用していたが、
真の OOS ではないため全訓練データの `pred_std` でゲート判定するよう修正。
eval_set の `pred_std_val` は診断フィールドとして保持。

### buy 側モデル状況
- buy モデルは 600# で既に null (統一モデルフォールバック)
- 原因: stale (2/24, 1ヶ月超) + C2 統計ゲート拒否 (p=0.51)
- `.degenerate_bak` ファイルも存在 — 退化検出履歴あり
- 統一モデル (3/27, corr=+0.10) で運用中、追加対応不要
- 646# の P0-A/C ガードは buy 側再学習時にも同様に適用される

## 影響範囲
- `retrain_model()` の最終学習
- `_evaluate_wf_multi()` / `_evaluate_wf_single()` の WF 評価
- side 別モデル学習の最小サンプル数ゲート
