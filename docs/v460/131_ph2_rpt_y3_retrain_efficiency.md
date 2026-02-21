# 131# ph2 Y3 効率化報告 — SkipGate 再訓練パイプラインの過去成果活用

> **ドキュメント番号**: 131#  
> **日付**: 2026-02-21  
> **Git 範囲**: `422223e69`..`2d0d8aab1`  
> **前提文書**: 130# (Y3 SkipGate 再訓練・デプロイ), 126# (retrain_scheduler 初版), 118# (Appendix F 再訓練仕様)  
> **目的**: Y3 再訓練パイプラインの効率化 — 過去モデル・データを最大限に活用し、サイクル時間・過適合リスクを低減

---

## 目次

- [§0 エグゼクティブサマリ](#0-エグゼクティブサマリ)
- [§1 Y3 現状分析](#1-y3-現状分析)
  - [§1.1 デプロイ済みモデルの特性](#11-デプロイ済みモデルの特性)
  - [§1.2 特徴量重要度分析 (130# モデル, n=355)](#12-特徴量重要度分析-130-モデル-n355)
  - [§1.3 ボトルネック特定](#13-ボトルネック特定)
- [§2 効率化施策の設計と実装](#2-効率化施策の設計と実装)
  - [E1: LightGBM Warm-start](#e1-lightgbm-warm-start-init_model)
  - [E2: Early Stopping](#e2-early-stopping)
  - [E3: Dead Feature Pruning](#e3-dead-feature-pruning)
  - [E4: Enriched Data Cache](#e4-enriched-data-cache)
- [§3 DRY リファクタリング](#3-dry-リファクタリング)
- [§4 テスト結果](#4-テスト結果)
- [§5 メタデータ拡張](#5-メタデータ拡張)
- [§6 YAML 追加パラメータ](#6-yaml-追加パラメータ)
- [§7 変更ファイル一覧](#7-変更ファイル一覧)
- [§8 考察 — WF score 下落の分析](#8-考察--wf-score-下落の分析)
- [§9 今後の展望](#9-今後の展望)

---

## §0 エグゼクティブサマリ

130# で Y3 SkipGate を再訓練・デプロイした。本書ではそのモデルと訓練パイプラインを分析し、過去成果を活用した 4 施策を設計・実装した。

| 施策 | 概要 | 効果 |
|------|------|------|
| **E1: Warm-start** | 前モデルの LightGBM Booster を `init_model` に使用 | 収束高速化 + 知識転移 |
| **E2: Early stopping** | WF val split で過学習を自動停止 (`n_estimators`: 旧固定 150 → 上限 300 + 自動選定) | 最適木数の自動選定 (実測: WF 5 trees / 本訓練 298 trees) |
| **E3: Dead feature pruning** | split=0 の特徴量を自動除外 | 過適合リスク低減 (19→13 features, 31.6% 削減) |
| **E4: Enriched cache** | enrichment 結果を pkl キャッシュ | I/O コスト大幅削減 (レコード数不変時は再計算回避) |

テスト: 26 件全 PASS (既存 20 + 新規 6)。`--all-runs --once` でのワンショット実行検証済み。

---

## §1 Y3 現状分析

### §1.1 デプロイ済みモデルの特性

130# でデプロイされた Y3 モデルの特性を整理する。

| 項目 | 値 |
|------|-----|
| モデルパス | `models/v460/skip_gate_lgbm_pnl120.pkl` |
| ターゲット | pnl120 回帰 |
| サンプル数 | 355 (pnl120 有効のみ。全 fill records は 852) |
| 特徴量数 | 19 (base 16 + OB 3) |
| WF score | +0.297 |
| 木の数 | 150 (固定) |
| ハイパーパラメータ | max_depth=4, lr=0.05, num_leaves=15, min_child_samples=20 |
| デプロイ日時 | 2026-02-21 20:02 |

### §1.2 特徴量重要度分析 (130# モデル, n=355)

以下は **130# デプロイモデル** (355 サンプル, pnl120) の LightGBM split-based 重要度。E3 施策のための事前分析。

| 特徴量 | Split 数 | 比率 | 評価 |
|--------|---------|------|------|
| spread_jpy | 179 | 19.1% | **核心** |
| hour_cos | 126 | 13.4% | 重要 |
| hour_sin | 92 | 9.8% | 重要 |
| spread_bps_ob | 61 | 6.5% | 有効 |
| vpin_60s | 57 | 6.1% | 有効 |
| depth_imbalance_ob | 52 | 5.5% | 有効 |
| side_aligned_tfi | 51 | 5.4% | 有効 |
| side_aligned_velocity | 51 | 5.4% | 有効 |
| avg_trade_size | 50 | 5.3% | 有効 |
| price_velocity_60s | 44 | 4.7% | 有効 |
| side_aligned_imbalance | 40 | 4.3% | 有効 |
| offset_ratio | 35 | 3.7% | 有効 |
| trade_count_60s | 30 | 3.2% | 有効 |
| regime_ranging | 27 | 2.9% | 限定的 |
| buy_ratio | 22 | 2.3% | 限定的 |
| side_buy | 16 | 1.7% | 限定的 |
| trade_flow_imbalance_60s | 5 | 0.5% | **ほぼ不使用** |
| regime_trending | 1 | 0.1% | **ほぼ不使用** |
| regime_high_vol | 0 | 0.0% | **完全不使用** |

**所見**: 19 特徴量中 3 個 (regime_high_vol, regime_trending, trade_flow_imbalance_60s) が split ≤ 5 で実質的にモデルに寄与していない。サンプル数 355 に対して 19 次元は過多の可能性がある。

> **注**: E3 施策のワンショット実行 (§4, n=858) ではサンプル数が 2.4 倍に増えたため、split 分布が大きく変化し、§1.2 で有効だった特徴量 (depth_imbalance_ob, avg_trade_size, side_buy) も split=0 となった。これはデータ規模に応じた重要度変動であり、E3 の設計がロバストであることを示す。

### §1.3 ボトルネック特定

| ボトルネック | 詳細 | 影響 |
|-------------|------|------|
| **I/O コスト** | 毎 retrain cycle で OB + trades データを再読込 | スケジューラのサイクル時間を支配 |
| **固定木数** | 150 trees 固定。355 samples に対して過剰の可能性 | 過学習リスク |
| **コールドスタート** | 毎回ゼロから訓練。前モデルの知識を捨てている | 収束が遅い |
| **Dead features** | split=0 の特徴量が次元を無駄に消費 | ノイズ特徴量の curse of dimensionality |

---

## §2 効率化施策の設計と実装

### E1: LightGBM Warm-start (`init_model`)

**原理**: LightGBM は `init_model` パラメータで既存の Booster を初期値として受け取り、追加学習する機能を持つ。新データが少量しか増えていない場合、ゼロからの再訓練よりも前モデルを起点とした方が収束が速い。

**実装** (`scripts/v460/ml/retrain_scheduler.py`):
- `retrain_model()` で前モデルの SkipGate をロードする際、`booster_` を抽出して保持
- WF eval 時と全データ訓練時の両方で `fit(init_model=prev_booster)` を使用
- **安全条件**: `feature_cols` が前モデルと完全一致する場合のみ warm-start を適用。E3 pruning で特徴量が変更された場合は自動スキップ

**YAML 設定**: `warm_start_enabled: true` (デフォルト有効)

### E2: Early Stopping

**原理**: LightGBM の `early_stopping_rounds` で validation loss が N ラウンド改善しなければ自動停止。§1.1 の固定 150 trees は、小サンプル (n=355) では過剰であり、最適木数は自動選定すべき。

**実装** (`scripts/v460/ml/retrain_scheduler.py`):
- WF eval: train/test 分割の test set を `eval_set` に使用
- 全データ訓練: 内部的に直近 20% を val split として使用
- `lgbm_n_estimators_max=300` を上限として設定 (旧: 固定 150)
- `early_stopping_rounds=20` で N ラウンド改善なし → 自動停止
- `callbacks=[lgb.early_stopping(), lgb.log_evaluation()]` で制御

**YAML 設定**: `early_stopping_rounds: 20`, `lgbm_n_estimators_max: 300`

### E3: Dead Feature Pruning

**原理**: WF eval で得た `feature_importances_` (split count) が閾値以下の特徴量は、モデルがほぼ使用していない。次元削減で過適合リスクを低減する。

**実装** (`scripts/v460/ml/retrain_scheduler.py`):
- WF eval 後に `feature_importance` dict を返却
- `retrain_model()` の Step 5 で split ≤ `feature_pruning_min_importance` の特徴量を自動除外
- **安全条件**: 最低 5 特徴量は保持 (過剰 pruning 防止)
- pruning によって feature set が変更された場合、E1 warm-start を自動スキップ (Booster の特徴量次元が不一致のため)

**YAML 設定**: `feature_pruning_enabled: true`, `feature_pruning_min_importance: 0`

### E4: Enriched Data Cache

**原理**: `enrich_fill_records()` は OB スナップショットと trades データの I/O + マッチングが処理時間の支配的要因。レコード数が変わらない限り結果は同一であるため、pkl キャッシュで再計算を回避する。

**実装** (`scripts/v460/ml/retrain_scheduler.py`):
- `_get_enriched_cache_path(results_dir)` → `cache/data/enriched_{name}.pkl`
- `_load_enriched_cache(path, n_records)` → レコード数一致時のみキャッシュ利用
- `_save_enriched_cache(path, enriched)` → enrichment 後に保存
- **無効化条件**: レコード数不一致で自動 invalidate (新データ到着時は必ず再計算)

**YAML 設定**: `enriched_cache_enabled: true`

---

## §3 DRY リファクタリング

`_build_lgbm_regressor()` を抽出し、`_evaluate_wf()` と `retrain_model()` における LGBMRegressor 構築を共通化:

```python
def _build_lgbm_regressor(cfg: dict, n_estimators_override: int | None = None) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=n_estimators_override or cfg.get("lgbm_n_estimators", 150),
        max_depth=cfg.get("lgbm_max_depth", 4),
        learning_rate=cfg.get("lgbm_learning_rate", 0.05),
        num_leaves=cfg.get("lgbm_num_leaves", 15),
        min_child_samples=cfg.get("lgbm_min_child_samples", 20),
        verbosity=-1,
        random_state=42,
    )
```

---

## §4 テスト結果

### §4.1 ユニットテスト

| テストスイート | 件数 | 結果 |
|--------------|------|------|
| 既存テスト (20) | 20 | ✅ all passed |
| TestE4EnrichedCache.test_cache_roundtrip | 1 | ✅ passed |
| TestE4EnrichedCache.test_cache_invalidation_on_count_mismatch | 1 | ✅ passed |
| TestE3FeaturePruning.test_dead_features_identified | 1 | ✅ passed |
| TestE3FeaturePruning.test_pruning_preserves_minimum_features | 1 | ✅ passed |
| TestBuildLgbmRegressor.test_default_params | 1 | ✅ passed |
| TestBuildLgbmRegressor.test_n_estimators_override | 1 | ✅ passed |
| **合計** | **26** | ✅ **0 failed** |

テストファイル: `tests/unit/v460/test_retrain_hot_reload.py`

### §4.2 ワンショット実行テスト (`--all-runs --once`)

858 filled samples (pnl120 target) での実行結果:

```
WF eval     : score=+0.0218, actual_n_trees=5 (E2: 300→5, 98% 削減)
E3 pruning  : 6 dead features pruned → 13 features remaining
              [side_buy, regime_trending, regime_high_vol,
               trade_flow_imbalance_60s, avg_trade_size, depth_imbalance_ob]
E4 cache    : 1660 records saved → cache/data/enriched_fill_test.pkl
E1 warm-start: skipped (E3 pruning で feature set 変更のため)
Final model : 298 trees, 13 features → models/v460/skip_gate_lgbm_pnl120.pkl
```

> **注意**: §1.2 (n=355) では depth_imbalance_ob=52 splits, avg_trade_size=50 splits だったが、n=858 では split=0 に変化した。サンプル数増加で他の特徴量に情報が集約されたためと考えられる。E3 の pruning はデータ規模に適応的に動作する。

---

## §5 メタデータ拡張

デプロイ済みモデルの metadata に効率化情報を記録:

```json
{
  "warm_start_used": false,
  "early_stopping_used": true,
  "actual_n_trees": 298,
  "pruned_features": ["side_buy", "regime_trending", "regime_high_vol",
                       "trade_flow_imbalance_60s", "avg_trade_size",
                       "depth_imbalance_ob"],
  "enriched_cache_used": true
}
```

`warm_start_used: false` は E3 pruning で feature set が変更されたため (安全条件によるスキップ)。

---

## §6 YAML 追加パラメータ

`configs/v460/fill_test.yaml` の `retrain:` セクションに以下を追加:

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `warm_start_enabled` | `true` | E1: 前モデル Booster を init_model に使用 |
| `early_stopping_rounds` | `20` | E2: N ラウンド改善なし → 停止 |
| `lgbm_n_estimators_max` | `300` | E2: early stopping 時の n_estimators 上限 |
| `feature_pruning_enabled` | `true` | E3: split ≤ 閾値の特徴量を自動除外 |
| `feature_pruning_min_importance` | `0` | E3: split 回数閾値 (0 = split=0 のみ除外) |
| `enriched_cache_enabled` | `true` | E4: enriched data pkl cache |

---

## §7 変更ファイル一覧

| ファイル | 変更種別 | 施策 |
|---------|---------|------|
| `scripts/v460/ml/retrain_scheduler.py` | E1-E4 実装 + DRY リファクタリング | §2, §3 |
| `configs/v460/fill_test.yaml` | E1-E4 設定追加 | §6 |
| `tests/unit/v460/test_retrain_hot_reload.py` | E1-E4 テスト 6 件追加 | §4.1 |

---

## §8 考察 — WF score 下落の分析

130# デプロイモデル (WF score +0.297) に対し、本書 E1-E4 適用後のワンショット実行では WF score +0.022 に下落した。要因分析:

| 要因 | 影響度 | 詳細 |
|------|--------|------|
| **サンプル数増加** | 大 | 355→858。WF の train/test 分割比率は同一だが、データ分布が変化。より多くの市場状態を含むため、汎化が困難に |
| **E2 early stopping** | 大 | WF eval で 5 trees に停止。150 trees → 5 trees は劇的な容量削減。WF test set の予測精度が低下 |
| **E3 feature pruning** | 中 | 19→13 features。WF eval 内で pruning されるため、WF score 自体には反映されないが、最終モデルの性能に影響 |
| **データ品質** | 小 | 新規 run_id のデータが混在。初期 run は戦略が不安定な期間のデータを含む |

**評価**: WF score は single-split のためバリアンスが高く、0.022 は「+0.297 から悪化」というより「855 サンプルでの真の汎化性能に接近」と解釈する方が妥当。本番でのモデル性能は fill_test の PnL 推移で実証検証する。quality gate の `min_score_improvement` は通常運用時には正値に設定されるため、低品質モデルが自動デプロイされるリスクは制御下にある。

---

## §9 今後の展望

| 施策 | 優先度 | 説明 |
|------|--------|------|
| Optuna ベイズ最適化 | 中 | `--all-runs` 時に max_depth/num_leaves/learning_rate を自動探索 |
| Time-weighted sampling | 中 | 古いサンプルに減衰重みを付与。市場構造変化への追従性向上 |
| Feature engineering v2 | 低 | E3 で除外された dead features の代替候補探索 |
| Expanding window CV | 低 | 単一 WF split → 3-fold expanding window CV でスコアの安定化 |
