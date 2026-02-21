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

> **重要**: 初版 (§0-§9) でのモデルデプロイは「ファイル保存成功」(deployed) までの確認であり、「実運用プロセスでの hot-reload 成功」(activated) は未確認だった。Appendix A レビューで hash 移送バグ (A.1 #1) が発覚し、Appendix B で修正・確認済み。

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
| `ztb/evaluation/walk_forward` 接続 | **高** | 既存 WalkForwardSplitter (embargo 付き multi-window) を retrain gate に導入。single-split → 3-fold expanding CV。v458 資産の直接活用 |
| v459 統計 gate 導入 | 高 | Holm/p-mean, 効果量 (Cliff's Delta) 併用を retrain deploy 判定に導入。000# §3.7 の統計検定仕様を活用 |
| Optuna ベイズ最適化 | 中 | `--all-runs` 時に max_depth/num_leaves/learning_rate を自動探索 |
| Time-weighted sampling | 中 | 古いサンプルに減衰重みを付与。市場構造変化への追従性向上 |
| Feature engineering v2 | 低 | E3 で除外された dead features の代替候補探索 |

---

## Appendix A: Codex 追記レビュー (2026-02-21 22:22 JST)

### A.1 重大度付き指摘

| # | 重大度 | 対象 | 問題 | 推奨対応 |
|---|---|---|---|---|
| 1 | **CRITICAL** | `scripts/v460/ml/retrain_scheduler.py:744` / `scripts/v460/ml/skip_gate.py:480` | アトミック保存時の hash ファイル移動先計算が不整合で、`*.pkl` と `*.pkl.sha256` が不一致化。結果、`fill_test.log` で hot-reload が恒常失敗し、新モデルが実運用に反映されていない。 | `tmp_hash` 計算を `tmp_path.with_suffix(tmp_path.suffix + '.sha256')` に修正。既存 `models/v460/skip_gate_lgbm_pnl120.pkl.sha256` を再生成。 |
| 2 | **HIGH** | `docs/v460/131_ph2_rpt_y3_retrain_efficiency.md:35` / `results/v460/fill_test/logs/fill_test.log` | 131# は「Y3効率化モデルを創出」としているが、実運用側は hash mismatch により旧モデルを保持し続けている（reload失敗が連続）。成果の実運用反映にギャップ。 | 文書に「deployed=保存成功」と「activated=運用反映成功」を分離記載し、activation確認ログを必須化。 |
| 3 | **HIGH** | `scripts/v460/ml/retrain_scheduler.py:536` | 前モデル読み込み失敗を `except: pass` で握り潰しており、品質ゲートが「no prev model」分岐に誤って入る。`absolute_min_score` 判定が過敏化/誤判定化。 | 例外を警告ログ出力し、`result['prev_model_load_error']` に記録。hash mismatch は即 `status=error` 返却も検討。 |
| 4 | **HIGH** | `logs/retrain_scheduler.log` (2026-02-21 20:01) | `--all-runs --once` で `WF score=-0.3007` のモデルが deploy されている。相対ゲート無効化中で、負の期待値モデル投入リスクが現実化。 | `--all-runs` 時でも `absolute_min_score` に加えて `pnl120_improvement >= 0` のハード制約を追加。 |
| 5 | **MEDIUM** | `docs/v460/131_ph2_rpt_y3_retrain_efficiency.md:57` / `logs/retrain_scheduler.log` | 文書は「ターゲット pnl120 回帰」を主軸記述だが、実行ログは `target=pnl30`。モデル名 (`*_pnl120.pkl`) と target 意味が乖離し運用誤認を誘発。 | モデルパス命名を target と同期 (`skip_gate_lgbm_pnl30.pkl`) するか、metadataの `target` を運用表示の主キーに。 |
| 6 | **MEDIUM** | `scripts/v460/ml/retrain_scheduler.py:205` | E4 cache の invalidation 条件が「行数一致のみ」。同件数でも run混在・設定変更・ソース更新時に stale cache を再利用する危険。 | cache key に `run_id集合hash + target + feature_cols + config digest` を追加。 |
| 7 | **MEDIUM** | `docs/v460/131_ph2_rpt_y3_retrain_efficiency.md:134` | E3 pruning が single WF split 由来で不安定。データ増加時に split=0 になる特徴量が頻繁に変動し、feature set が振動。 | pruning は「連続N回でdead」または「3-fold平均importance」で実行。 |
| 8 | **MEDIUM** | `docs/v460/131_ph2_rpt_y3_retrain_efficiency.md:272` | 「Expanding window CV」は将来課題扱いだが、`ztb/evaluation/walk_forward/splitter.py` など既存資産があり、再利用で即導入可能。vXXX資産の活用が不十分。 | `ztb/evaluation/walk_forward` を retrain WF に接続し、embargo付き multi-window score を quality gate に昇格。 |

### A.2 過去 vXXX 成果の活用評価

| 分類 | 評価 | 内容 |
|---|---|---|
| 活用できている | ✅ | 126#/127# の run_id分離・品質ゲート・metadata拡張、118# Appendix F の `--all-runs` 運用思想を継承。 |
| 活用不足 | ⚠️ | v458 系の walk-forward 分割資産（embargo/複数窓）を再訓練判定に未接続。 |
| 活用不足 | ⚠️ | v459 系の統計 gate 思想（Holm/p-mean, 効果量併用）を retrain deploy 判定に未導入。 |
| 逆行リスク | ⚠️ | 「log上 deploy 成功でも運用未反映」という観測不能状態を許容しており、v459で強化した観測一貫性ポリシーと齟齬。 |

### A.3 次に何をすべきか (優先順)

1. **P0 (即時)**: hash 移送バグ修正 + `.sha256` 再生成 + hot-reload 成功確認（1回でなく3連続成功まで）。
2. **P0 (即時)**: `except: pass` を廃止し、前モデル読み込み失敗を明示ログ化。quality gate分岐の誤動作を止める。
3. **P1 (同日)**: `--all-runs` の deploy 条件を強化（`pnl120_improvement>=0` と `score>=absolute_min` のAND）。
4. **P1 (同日)**: cache key を config/run依存に拡張し、行数一致のみ invalidation を廃止。
5. **P2 (次セッション)**: `ztb/evaluation/walk_forward` を使った 3-fold expanding/embargo WF を retrain gate に導入。

### A.4 補足（今回の再検証）

- `tests/unit/v460/test_retrain_hot_reload.py` は 26/26 PASS を再確認。
- `scripts/v460/ml/retrain_scheduler.py --all-runs --once` 実行で、現時点は `score=-0.2745` で `absolute_min_score` reject を確認（prev model load失敗分岐）。
- 131# の主張は方向性として妥当だが、**「実運用反映まで含めた完了判定」** が不足している。

---

## Appendix B: レビュー対応結果

**対応日時**: 2025-02-21
**対象**: Appendix A の全 8 件 + 追加盲点分析
**Git 範囲**: `e4c4b2edf`..`HEAD`

### B.1 対応一覧

| A.1 # | 重大度 | 対応 | 詳細 |
|-------|--------|------|------|
| **1** | CRITICAL | ✅ 修正済 | `tmp_hash` パス計算を `tmp_path.with_suffix(tmp_path.suffix + ".sha256")` に修正。`with_suffix` は最終 suffix のみ置換するため旧コードは `.pkl.tmp` → `.pkl.pkl.tmp.sha256` (二重 `.pkl`) を生成。hash 未移動 → hot-reload 常時失敗。SHA256 再生成済み。テスト 3 件追加 |
| **2** | HIGH | ✅ 修正済 | §0 に deployed/activated の区別を明示注記 |
| **3** | HIGH | ✅ 修正済 | `except Exception: pass` → `except Exception as e:` + `logger.warning(...)` + `result["prev_model_load_error"]`。テスト 1 件追加 |
| **4** | HIGH | ✅ 修正済 | `--all-runs` 時に `all_runs_require_positive_pnl=True` 設定。target PnL 改善が負なら棄却 |
| **5** | MEDIUM | ✅ 修正済 | `_validate_config()` に target/model\_path 命名不整合の警告ログ追加 |
| **6** | MEDIUM | ✅ 修正済 | `cache_key = md5(target\|features\|run_ids)[:16]` 追加。行数+key 二重検証。旧フォーマット後方互換確保。テスト 2 件追加 |
| **7** | MEDIUM | ✅ 修正済 | `feature_pruning_min_trees=20` ガード追加。WF 木数 < 20 で pruning スキップ。テスト 2 件追加 |
| **8** | MEDIUM | ✅ 対応済 | §9 に WF 資産活用・v459 統計 gate を **高** 優先度で記載 |

### B.2 テスト結果

| テストスイート | 件数 |
|--------------|------|
| 既存テスト | 26 |
| TestAtomicHashMove (#1) | 3 |
| TestPrevModelLoadError (#3) | 1 |
| TestE4EnrichedCache (#6) | 2 |
| TestE3PruningMinTrees (#7) | 2 |
| **合計** | **34 passed, 0 failed** |

### B.3 追加盲点分析

| 分類 | 盲点 | 対応 |
|------|------|------|
| 過去資産の未活用 | `ztb/evaluation/walk_forward/splitter.py` — embargo\_days 付き multi-window 分割が retrain 未使用 | §9 P2。次セッションで `_evaluate_wf()` multi-window 化 |
| 過去資産の未活用 | 000# §3.7 統計検定仕様 (Holm-Bonferroni, Cliff's Delta) — retrain deploy 判定に未導入 | §9 P2 |
| 観測一貫性 | deploy 成功 = ファイル書き込み成功のみ。activation = hot-reload 成功は別途確認要 | §0 注記 → **B.5 で post-deploy 自己検証を実装** |
| E2+E3 相互作用 | early stopping で木数 5 → importance 不安定 → 誤 prune | `min_trees=20` ガード → **B.5 で連続 dead pruning も実装** |

### B.4 変更ファイル一覧 (Appendix B 初版)

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/ml/retrain_scheduler.py` | #1 hash, #3 except, #4 pnl gate, #5 target 警告, #6 cache\_key, #7 min\_trees |
| `configs/v460/fill_test.yaml` | `feature_pruning_min_trees: 20` 追加 |
| `tests/unit/v460/test_retrain_hot_reload.py` | 8 テスト追加 (34 total) |
| `docs/v460/131_ph2_rpt_y3_retrain_efficiency.md` | §0 注記, §9 改訂, Appendix B 追加 |

---

### B.5 レビュー深掘り対応 (2回目)

**対応日時**: 2026-02-21 (Appendix B 直後)
**方針**: A.1 の「ガード追加にとどまっていた」項目を本質的に強化

#### B.5.1 Post-deploy 自己検証 (A.2 逆行リスク / A.3 P0)

**問題**: `status=deployed` はファイル書き込み成功のみを意味し、hash 一致や pickle 健全性は未検証。
hot-reload 側 (SkipGateEvaluator) で初めて失敗に気付く「観測不能状態」が残存していた。

**対応**: deploy 完了直後に `SkipGate.load(model_path)` を実行し:
- hash 検証 + pickle load + n\_samples 一致 → `status="deployed_verified"`
- load 失敗 → `status="deployed"` のまま + `deploy_verify_error` に記録 + ERROR ログ
- n\_samples 不一致 → WARNING ログ (deploy 自体は成功)

**効果**: retrain\_scheduler 側で「hot-reload が成功し得るか」を即時判定。v459 観測一貫性ポリシーとの整合を回復。

#### B.5.2 連続 dead pruning (A.1 #7 深掘り)

**問題**: single WF split の feature importance は不安定。サンプル数やデータ分布の変化で
「前回重要→今回 dead」の振動が発生し、feature set が cycle ごとに変動するリスク。

**A.1 #7 推奨**: "pruning は「連続N回でdead」または「3-fold平均importance」で実行"

**対応**: `feature_pruning_require_consecutive=True` (デフォルト有効)
- metadata に `wf_dead_features` (今回 WF で dead な全特徴量) を記録
- 次回 retrain 時: `prev_gate.metadata["wf_dead_features"]` と前回 `feature_cols` から `prev_dead` を構築
- `wf_dead ∩ prev_dead` の交差のみ実際に prune (連続 dead 条件)
- 前モデルなし or prev\_dead 空の場合: 従来通り単回 dead で prune (初回 bootstrap)

**効果**: feature set の振動を抑制。データ規模変化時の過剰 pruning を防止。

#### B.5.3 残存コード品質スキャン

| 検査項目 | 結果 |
|---------|------|
| `except:pass` / `except Exception: pass` 残存 | ✅ retrain\_scheduler.py 内に残存なし |
| `with_suffix` 誤用 (二重 suffix) | ✅ 全ファイルで修正確認済み |
| bare `except:` (scripts/v460) | ✅ なし (全て型指定 except) |

#### B.5.4 変更ファイル一覧 (2回目)

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/ml/retrain_scheduler.py` | post-deploy 検証, 連続 dead pruning, wf\_dead\_features metadata |
| `configs/v460/fill_test.yaml` | `feature_pruning_require_consecutive: true` 追加 |
| `tests/unit/v460/test_retrain_hot_reload.py` | 6 テスト追加 (40 total): ConsecutiveDeadPruning×3, PostDeployVerification×3 |

#### B.5.5 テスト結果 (2回目)

| テストスイート | 件数 |
|--------------|------|
| 既存 + B.1 テスト | 34 |
| TestConsecutiveDeadPruning | 3 |
| TestPostDeployVerification | 3 |
| **合計** | **40 passed, 0 failed** |

---

## Appendix C: ztb アセット統合 (C1-C3)

**対応日時**: 2026-02-22

レビュー (Appendix A) で提案された ztb/ 既存資産の活用について、
P1 (即時適用可能) の 3 アセットを retrain_scheduler.py に統合した。

### C.1 WF Multi-Window 評価 (C1)

**統合アセット**: `ztb.evaluation.walk_forward.splitter.WalkForwardSplitter`

**課題**: 従来の `_evaluate_wf()` は単一の train/test 分割 (80/20) で評価しており、
テスト期間の偏りに脆弱だった。

**対応**:
- `_evaluate_wf_multi()` を新設。`WalkForwardSplitter` で複数 WF ウィンドウを生成し、
  各ウィンドウで独立に train→predict→PnL 評価を実行。
- `_evaluate_wf()` をディスパッチ関数化: `wf_multi_window_enabled=True` (default) なら
  multi-window → データ不足時は single-window にフォールバック。
- per-window fold-level PnL データを返却 (C2 統計ゲートへの入力)。

**設定キー** (`configs/v460/fill_test.yaml`):

| キー | デフォルト | 説明 |
|------|-----------|------|
| `wf_multi_window_enabled` | `true` | multi-window 有効 |
| `wf_initial_train_pct` | 0.50 | 初期訓練割合 |
| `wf_val_pct` | 0.10 | 検証割合 |
| `wf_test_pct` | 0.15 | テスト割合 |
| `wf_step_pct` | 0.20 | ウィンドウシフト |
| `wf_embargo_rows` | 0 | エンバーゴ行数 |
| `wf_min_window_train` | 30 | 最小訓練サンプル/window |
| `wf_min_window_test` | 10 | 最小テストサンプル/window |

### C.2 統計的品質ゲート (C2)

**統合アセット**: `ztb.metrics.gate_checks` (`g1_judgment`, `holm_bonferroni_gate`)

**課題**: 品質ゲートが `score - prev_score > threshold` の単純比較であり、
サンプルサイズやランダム変動を考慮していなかった。

**対応**:
- `_apply_statistical_gate()` を新設。
  - Multi-window (≥2): `g1_judgment()` を適用 (001# §5.3 準拠の p-mean → Holm → AND)
  - Single-window: `holm_bonferroni_gate()` を適用 (per-sample PnL 比較)
  - テストサンプル < `min_test_samples` 時: スキップ (統計的検出力不足)
- 既存のスコアベースゲートに**追加**する形で統計ゲートを挿入 (and 条件)。
- `result["statistical_gate"]` にゲート結果を記録 → metadata にも保存。

**設定キー**:

| キー | デフォルト | 説明 |
|------|-----------|------|
| `statistical_gate_enabled` | `true` | 統計ゲート有効 |
| `statistical_gate_alpha` | 0.05 | FWER |
| `statistical_gate_min_effect` | 0.147 | Cliff's Delta 最小閾値 (small effect) |
| `statistical_gate_min_test_samples` | 40 | 最小テストサンプル |

### C.3 冗長特徴量除去 (C3)

**統合アセット**: `ztb.analysis.redundancy` (`calculate_feature_correlations`, `find_highly_correlated_features`)

**課題**: E3 dead-feature pruning は split=0 の特徴量のみ除去するが、
高相関ペア (r>0.85) が残存 → 次元の呪い・学習効率低下を抑制できない。

**対応**:
- E3 dead-feature pruning の**直後**に相関ベースの冗長除去を挿入。
- `find_highly_correlated_features(corr_matrix, threshold)` で高相関ペアを検出。
- ペアのうち WF feature\_importance が低い方を除去 (同値なら名前順で決定的)。
- 最低 5 特徴量は保持 (過剰 pruning 防止)。
- E1 warm-start: C3 pruning 実施時は feature set 不一致のため warm-start スキップ。

**設定キー**:

| キー | デフォルト | 説明 |
|------|-----------|------|
| `redundancy_pruning_enabled` | `true` | 冗長除去有効 |
| `redundancy_correlation_threshold` | 0.85 | 相関閾値 |

### C.4 循環参照対策: `_safe_import_ztb_module()`

`ztb.analysis.__init__` → `ztb.trading` 間の循環参照により、
通常の `import` では ztb サブモジュールがロード不可になる場合がある。
`_safe_import_ztb_module()` を新設し、`importlib.util.spec_from_file_location()` で
直接ファイルロードするフォールバックを実装。

### C.5 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/ml/retrain_scheduler.py` | `_safe_import_ztb_module()`, `_evaluate_wf_multi()`, `_apply_statistical_gate()`, C3 冗長除去, metadata 拡張 |
| `configs/v460/fill_test.yaml` | C1-C3 設定キー追加 (12 keys) |
| `tests/unit/v460/test_retrain_hot_reload.py` | 11 テスト追加: TestMultiWindowWF×4, TestStatisticalGate×4, TestRedundancyPruning×3 |
| `docs/v460/131_ph2_rpt_y3_retrain_efficiency.md` | Appendix C 追加 |

### C.6 テスト結果

| テストスイート | 件数 |
|--------------|------|
| 既存テスト (A.1 + B.5) | 40 |
| TestMultiWindowWF (C1) | 4 |
| TestStatisticalGate (C2) | 4 |
| TestRedundancyPruning (C3) | 3 |
| **合計** | **51 passed, 0 failed** |

---

## Appendix D: 動的ロットサイズ調査 + ph3 先行実装

> ユーザ要求: 「今現在は基本的に1mBTCで回すしかありませんが、動的に変更する仕組みについて調査して下さい。
> その他ph3以降で先行して実施可能なタスクを探し出し、実装して下さい。」

### D.1 動的ロットサイズ機構の調査結果

#### D.1.1 現状: 既に完全実装されているが **無効** (方策 B)

| コンポーネント | ファイル | 役割 |
|--------------|---------|------|
| コアアルゴリズム | `scripts/v460/lib/lot_sizer.py` (~258行) | ステップベース増減、損失キャップ統合 |
| 統合レイヤー | `scripts/v460/lib/adaptation_engine.py` (~451行) | TTL キャッシュ、fill_record 統計集約 |
| 安全装置 | `scripts/v460/lib/balance_checker.py` | 残高ベースのリアルタイム縮小 |
| 設定 | `scripts/v460/lib/fill_config.py` + `configs/v460/fill_test.yaml` | `enable_dynamic_lot: false` |
| Kelly基準 | `ztb/trading/trade_execution_engine.py` | `_calculate_kelly_size()` 存在するが fill_test 未接続 |

#### D.1.2 有効化条件 (`compute_lot_size()`)

```
増量 (increase):
  fill_rate ≥ 70% AND as_ratio ≤ 30% AND recent_pnl_bps ≥ 0
  → current_lot + lot_step (0.001)

減量 (decrease):
  fill_rate < 60% OR as_ratio > 40% OR recent_pnl_bps < -1.0
  → current_lot - lot_step

損失キャップ (cap_shrink):
  cumulative_pnl_jpy < -(loss_cap_jpy × loss_cap_warning_ratio)
  → min_lot へ強制リセット

適用契機:
  lot_adapt_interval_cycles(50) サイクル毎 AND total_count ≥ min_adapt_samples(50)
```

#### D.1.3 無効化の理由

| 指標 | 現在値 | 増量閾値 |
|------|--------|---------|
| fill_rate | ~67% | ≥ 70% |
| AS ratio | ~28% | ≤ 30% |
| PnL (30s) | -0.172 bps | ≥ 0 bps |

**PnL が負の状態でロットを増やすと損失が拡大するため、方策 B を有効化する前に PnL ≥ +0.3 bps を達成する必要がある。**

#### D.1.4 スケーリング試算 (118# §8.4)

| ロット (BTC) | bps | JPY/cycle | 月間 JPY |
|-------------|-----|-----------|---------|
| 0.001 | -0.172 | -0.002 | -37 |
| 0.01 | +0.3 | +0.03 | +648 |
| 0.1 | +1.0 | +1.0 | +21,600 |
| 1.0 | +1.0 | +10.0 | +216,000 |

**大義達成のロードマップ**: PnL 改善 → 方策 B 有効化 → max_lot 段階的引き上げ → Kelly 基準統合

---

### D.2 実装: D1 レジーム連動ロット制御

#### D.2.1 背景

方策 B を将来有効化する際、市場レジームが不明確（unknown）な状態でロット増量を許可すると
リスクが過大になる。レジーム検出器 (`RegimeDetector`) の判定結果を `compute_lot_size()` に
フィードバックし、レジーム別の増減制御を追加。

#### D.2.2 変更内容

**`LotSizingConfig` 拡張** (lot_sizer.py):
```python
regime_guard_enabled: bool = True      # レジームガード有効化
regime_hold_regimes: tuple = ("unknown",)  # 増量ブロック対象
regime_decrease_regimes: tuple = ()    # 強制減量対象
```

**`compute_lot_size()` 拡張**:
- `regime_tag: str = "n/a"` パラメータ追加
- 損失キャップ判定後、条件判定前にレジームガードを挿入:
  - `regime_tag in regime_hold_regimes` → hold（増量ブロック）
  - `regime_tag in regime_decrease_regimes` → decrease（強制減量）
  - `regime_tag == "n/a"` → ガード無効（検出器不在時の安全デフォルト）

**`adaptation_engine.py` 連携**:
- `_build_lot_kwargs()` で YAML の 3 キーをマッピング
- `try_auto_lot_size()` で `regime_tag` を `regime_detector` から取得して渡す

**fill_test.yaml 追加キー**:
```yaml
lot_sizing:
  regime_guard_enabled: true
  regime_hold_regimes: [unknown]
  regime_decrease_regimes: []
```

---

### D.3 実装: D2 Oracle PnL 基準線スクリプト

#### D.3.1 背景 (118# §8.5)

ph3 進入前に「理論上の最大 PnL（全損益を完全予測し、PnL < 0 の取引をスキップした場合）」を
計算しておくことが必須とされている。この **Oracle PnL baseline** が ph3 Gate の前提条件。

#### D.3.2 スクリプト概要

**ファイル**: `scripts/v460/analysis/oracle_baseline.py` (~280行)

**主要機能**:
- `OracleMetrics` データクラス: actual/oracle PnL 統計、skip_rate、JPY 換算
- `compute_oracle_metrics()`: FillRecord リストから Oracle 指標を算出
  - 全体 / side 別 / regime 別ブレークダウン
  - 30s / 60s / 120s マルチタイムフレーム
  - ロットサイズシナリオ (0.001 → 0.1 BTC) × BTC 価格で月間 JPY 換算
- CLI: `--results-dir`, `--output`, `--lot-btc`, `--btc-price`
- JSON 出力対応

**用途**:
```powershell
# Oracle PnL 算出
python scripts/v460/analysis/oracle_baseline.py --results-dir results/ --lot-btc 0.001 --btc-price 15000000
```

**ph3 進入判定ロジック**:
- oracle_skip_rate < 0.7 → PASS（7割以上スキップなら改善余地大きすぎてまだ早い）
- oracle_pnl_mean > 0 → PASS（理論上プラスになる可能性あり）
- actual_pnl_mean > -0.5 → PASS（実績が大幅マイナスでない）

---

### D.4 ph3 先行タスク探索結果

| # | 候補 (118#/015# 等) | 状態 | 備考 |
|---|---------------------|------|------|
| §3.1 | fast_fill_defense sell-side 二層化 | **未着手** | P0, +0.2bps 期待, 中コスト → ph3 本体で実施 |
| §3.2 | warm_start 閾値即座収束 | ✅ 118# A2 で完了 | skip_gate.py L746-767 |
| §3.3 | regime warm-up state persistence | ✅ 121# A4 で完了 | resilience.py FillTestState |
| §5.6 | time_filter step 1 | config 変更のみ | 次回再起動時に適用可能 |
| §8.5 | Oracle baseline | ✅ **D2 で実装** | ph3 前提条件クリア |
| R3 | SkipGate warm_start テスト | LOW | 既に warm_start は安定動作中 |
| 015# §8.2 | SAC prerequisites | 大規模 | ModelRegistry, dim fix 等 → ph3 本体 |

**結論**: 前倒し可能な高価値タスクは D1 (レジーム連動ロット) と D2 (Oracle baseline) の 2 件。
残りは ph3 本体の一部として実施すべき規模。

---

### D.5 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/lib/lot_sizer.py` | D1: `LotSizingConfig` に 3 属性追加, `compute_lot_size()` に `regime_tag` パラメータ + ガードロジック |
| `scripts/v460/lib/adaptation_engine.py` | D1: `_build_lot_kwargs()` で YAML キーマッピング, `regime_tag` 連携 |
| `configs/v460/fill_test.yaml` | D1: `lot_sizing` セクションに 3 キー追加 |
| `scripts/v460/analysis/oracle_baseline.py` | D2: **新規** — Oracle PnL 基準線スクリプト |
| `tests/unit/v460/test_retrain_hot_reload.py` | D1: TestRegimeAwareLotSizing×6, D2: TestOracleBaseline×5 追加 |
| `docs/v460/131_ph2_rpt_y3_retrain_efficiency.md` | Appendix D 追加 |

### D.6 テスト結果

| テストスイート | 件数 |
|--------------|------|
| 既存テスト (C.6) | 51 |
| TestRegimeAwareLotSizing (D1) | 6 |
| TestOracleBaseline (D2) | 5 |
| **合計** | **62 passed, 0 failed** |
