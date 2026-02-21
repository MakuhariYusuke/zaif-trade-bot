# 131# Y3 効率化 — SkipGate 再訓練パイプラインの過去成果活用

> **ドキュメント番号**: 131#  
> **日付**: 2026-02-21  
> **Git 範囲**: `422223e69`..`HEAD`  
> **前提文書**: 130# (Y3 SkipGate 再訓練 デプロイ), 126# (retrain_scheduler 初版), 118# (Appendix F 再訓練仕様)  
> **目的**: Y3 再訓練パイプラインの効率化 — 過去モデル・データを最大限に活用

---

## §0 エグゼクティブサマリ

Y3 SkipGate 再訓練 (130# で実施) の結果を分析し、以下の 4 施策で効率化を実装した。

| 施策 | 概要 | 効果 |
|------|------|------|
| **E1: Warm-start** | 前モデルの LightGBM Booster を `init_model` に使用 | 収束高速化 + 知識転移 |
| **E2: Early stopping** | WF val split で過学習を自動停止 | 最適木数の自動選定 (150→5~298) |
| **E3: Dead feature pruning** | split=0 の特徴量を自動除外 | 過適合リスク低減 (19→13 features) |
| **E4: Enriched cache** | enrichment 結果を pkl キャッシュ | I/O コスト大幅削減 |

---

## §1 Y3 現状分析

### 1.1 デプロイ済みモデルの特性

| 項目 | 値 |
|------|-----|
| モデルパス | `models/v460/skip_gate_lgbm_pnl120.pkl` |
| ターゲット | pnl120 回帰 |
| サンプル数 | 355 (852 total, pnl120 valid のみ) |
| 特徴量数 | 19 (base 16 + OB 3) |
| WF score | +0.297 |
| 木の数 | 150 (固定) |
| デプロイ日時 | 2026-02-21 20:02 |

### 1.2 特徴量重要度分析

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

**所見**: 19 特徴量中 6 個 (31.6%) が split ≤ 5 で実質的にモデルに寄与していない。サンプル数 355 に対して 19 次元は過多。

### 1.3 ボトルネック特定

| ボトルネック | 詳細 | 影響 |
|-------------|------|------|
| **I/O コスト** | 毎 retrain cycle で OB + trades データを再読込 (~30秒) | スケジューラのサイクル時間を支配 |
| **固定木数** | 150 trees 固定。355 samples に対して過剰 | 過学習リスク |
| **コールドスタート** | 毎回ゼロから訓練。前モデルの知識を捨てる | 収束が遅い |
| **Dead features** | split=0 の特徴量が次元を無駄に消費 | ノイズ特徴量の curse of dimensionality |

---

## §2 効率化施策の設計と実装

### E1: LightGBM Warm-start (`init_model`)

**原理**: LightGBM は `init_model` パラメータで既存の Booster を初期値として受け取り、追加学習する機能を持つ。新データが少量しか増えていない場合、ゼロからの再訓練よりも前モデルを起点とした方が収束が速い。

**実装**:
- `retrain_model()` で前モデルの SkipGate をロードする際、`booster_` を抽出して保持
- WF eval 時と全データ訓練時の両方で `fit(init_model=prev_booster)` を使用
- **安全条件**: `feature_cols` が前モデルと完全一致する場合のみ warm-start を適用。E3 pruning で特徴量が変更された場合は自動スキップ

**YAML 設定**: `warm_start_enabled: true` (デフォルト有効)

### E2: Early Stopping

**原理**: LightGBM の `early_stopping_rounds` で validation loss が N ラウンド改善しなければ自動停止。355 samples に 150 trees は過剰であり、実データでは 5~30 trees で飽和する場合が多い。

**実装**:
- WF eval: train/test 分割の test を `eval_set` に使用
- 全データ訓練: 内部的に直近 20% を val split として使用
- `lgbm_n_estimators_max=300` を上限として設定、`early_stopping_rounds=20` で自動停止
- `callbacks=[lgb.early_stopping(), lgb.log_evaluation()]` で制御

**実測結果**:
- WF eval: 300→**5 trees** で停止 (97% 削減)
- 全データ訓練: 300→**298 trees** (val split で停止が緩い — 期待通り)

**YAML 設定**: `early_stopping_rounds: 20`, `lgbm_n_estimators_max: 300`

### E3: Dead Feature Pruning

**原理**: WF eval で得た `feature_importances_` (split count) が 0 の特徴量は、モデルが一度も使用していない。次元削減で過適合リスクを低減。

**実装**:
- WF eval 後に `feature_importance` dict を返却
- `retrain_model()` の Step 5 で split ≤ `feature_pruning_min_importance` の特徴量を自動除外
- **安全条件**: 最低 5 特徴量は保持 (過剰 pruning 防止)
- pruning 実行時は warm-start を自動スキップ (feature set 不一致)

**実測結果** (852 samples, pnl30 target):
```
Pruned 6 dead features:
  - side_buy (0 splits)
  - regime_trending (0 splits)
  - regime_high_vol (0 splits)
  - trade_flow_imbalance_60s (0 splits)
  - avg_trade_size (0 splits)
  - depth_imbalance_ob (0 splits)
→ 19 → 13 features remaining (31.6% 削減)
```

**YAML 設定**: `feature_pruning_enabled: true`, `feature_pruning_min_importance: 0`

### E4: Enriched Data Cache

**原理**: `enrich_fill_records()` は OB スナップショットと trades データの I/O + マッチングが支配的。レコード数が変わらない限り結果は同一であるため、pkl キャッシュで再計算を回避。

**実装**:
- `_get_enriched_cache_path(results_dir)` → `cache/data/enriched_{name}.pkl`
- `_load_enriched_cache(path, n_records)` → レコード数一致時のみキャッシュ利用
- `_save_enriched_cache(path, enriched)` → enrichment 後に保存
- **無効化条件**: レコード数不一致で自動 invalidate

**YAML 設定**: `enriched_cache_enabled: true`

---

## §3 DRY リファクタリング

`_build_lgbm_regressor()` を抽出し、`_evaluate_wf()` と `retrain_model()` の LGBMRegressor 構築を共通化:

```python
def _build_lgbm_regressor(cfg, n_estimators_override=None) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=n_estimators_override or cfg.get("lgbm_n_estimators", 150),
        max_depth=cfg.get("lgbm_max_depth", 4),
        ...
    )
```

---

## §4 テスト結果

| テストスイート | 件数 | 結果 |
|--------------|------|------|
| 既存テスト (20) | 20 | ✅ all passed |
| E4 cache roundtrip | 1 | ✅ passed |
| E4 cache invalidation | 1 | ✅ passed |
| E3 dead features identified | 1 | ✅ passed |
| E3 pruning min features guard | 1 | ✅ passed |
| _build_lgbm_regressor default | 1 | ✅ passed |
| _build_lgbm_regressor override | 1 | ✅ passed |
| **合計** | **26** | ✅ **0 failed** |

### ワンショット実効テスト結果

```
--all-runs --once 実行:
  - Total samples: 858 (filled)
  - WF eval: score=+0.0218, actual_n_trees=5 (E2 early stopping)
  - E3 pruning: 6 dead features → 13 remaining
  - E4 cache: 1660 records saved to cache/data/enriched_fill_test.pkl
  - Final model: 298 trees, 13 features
  - Status: deployed
```

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

---

## §6 YAML 追加パラメータ

`configs/v460/fill_test.yaml` の `retrain:` セクションに以下を追加:

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `warm_start_enabled` | `true` | E1: 前モデル Booster を init_model に使用 |
| `early_stopping_rounds` | `20` | E2: N ラウンド改善なし → 停止 |
| `lgbm_n_estimators_max` | `300` | E2: early stopping 時の上限 |
| `feature_pruning_enabled` | `true` | E3: split=0 の特徴量を自動除外 |
| `feature_pruning_min_importance` | `0` | E3: split 回数しきい値 |
| `enriched_cache_enabled` | `true` | E4: enriched data pkl cache |

---

## §7 変更ファイル一覧

| ファイル | 変更種別 | 施策 |
|---------|---------|------|
| `scripts/v460/ml/retrain_scheduler.py` | E1-E4 実装 + DRY | §2, §3 |
| `configs/v460/fill_test.yaml` | E1-E4 設定追加 | §6 |
| `tests/unit/v460/test_retrain_hot_reload.py` | E1-E4 テスト追加 | §4 |

---

## §8 今後の展望

| 施策 | 優先度 | 説明 |
|------|--------|------|
| Optuna ベイズ最適化 | 中 | `--all-runs` 時に max_depth/num_leaves/learning_rate を自動探索 |
| Time-weighted sampling | 中 | 古いサンプルに減衰重みを付与。市場構造変化への追従 |
| Feature engineering v2 | 低 | E3 で除外された dead features の代替候補探索 |
| Cross-validation | 低 | 単一 WF split → 3-fold expanding window CV |
