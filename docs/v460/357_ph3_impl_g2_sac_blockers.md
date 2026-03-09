# 357# ph3 G2 SAC ブロッカー実装

> **356a# B1/B3/B4 ブロッカー解消 — G2 SAC 訓練基盤の構築**
> 作成日: 2026-03-09

---

## §1 概要

356a# で特定された ph3 ブロッカー 5 件 (B1–B5) のうち、即時実装可能な 3 件 (B1/B3/B4) を解消した。
これにより G2 SAC 4-seed 訓練 → gate 自動判定の end-to-end パイプラインが構築された。

| Blocker | 内容 | 状態 |
|---|---|---|
| **B1** | g2_sac_train.yaml 未作成 | ✅ 解消 |
| B2 | 特徴量選定 | ⏳ B1 の YAML で 12 特徴量を仮選定 (P3A-3 で精査) |
| **B3** | feature_columns 未注入 | ✅ 解消 |
| **B4** | multi-seed wrapper 未実装 | ✅ 解消 |
| B5 | 3-way 設計判断 | 📝 356a# で C パスを主方針に決定 (コード変更不要) |

---

## §2 B1: g2_sac_train.yaml 作成

**ファイル**: `configs/v460/experiments/g2_sac_train.yaml`

G2-train gate 用の SAC 訓練実験設定ファイルを新規作成。
G1 YAML (g1_xgb_h5_direction.yaml) の構造を踏襲しつつ、SAC 固有のセクションを追加。

### 設計判断

| 項目 | 値 | 根拠 |
|---|---|---|
| `_gate` | `G2-train` | 001# §4.1 gate 体系 |
| `_task` | `sac_train` | task ディスパッチ対応 |
| seeds | `[42, 123, 456, 789]` | G2 gate 要件: 4-seed |
| gamma | `0.80` | v459 チューニング済み短期 γ |
| buffer_size | `100,000` | v459 値 |
| learning_starts | `100` | v459 早期学習開始 |
| total_timesteps | `50,000` | 初期検証用（本番は 200K+） |
| features | 12 個 | FeatureRegistry から選定 |

### 特徴量選定 (12 個)

FeatureRegistry の v430–v459 実績ベースで選定。v460 マイクロストラクチャ特徴量 (bid_ask_spread 等) は
HeavyTradingEnv の observation space との整合性が未検証のため除外。

```
price_velocity, micro_trend, volume_surge, rsi_14,
bb_position, macd_signal, atr_ratio, obv_slope,
vwap_deviation, momentum_5m, volatility_ratio, trade_intensity
```

### EnvironmentConfig 互換性

YAML の `environment` セクションは `EnvironmentConfig` dataclass のフィールド名に正確に対応:
- `initial_portfolio_value` (≠ `initial_balance`)
- `use_continuous_actions: true`
- `action_space_type: "continuous_1d"`
- `transaction_cost: 0.001`

---

## §3 B3: feature_columns → EnvironmentConfig 注入

**ファイル**: `scripts/v460/lib/tasks/sac_train.py` — `_create_training_env()`

### 問題

YAML `features.selected` で指定した特徴量が `HeavyTradingEnv` の observation space に
反映されず、デフォルト全特徴量が使用されていた。

### 修正

```python
# 356a# B3: 明示的に feature_names を注入
if feature_columns:
    env_config.feature_names = feature_columns
```

### 可観測性

`env_info` dict に `feature_columns_injected` boolean を追加し、
注入の有無をログ・マニフェストで追跡可能にした。

---

## §4 B4: G2 multi-seed ディスパッチ + gate 判定

**ファイル**: `scripts/v460/run_experiment.py`

3 つの新関数を追加し、既存の `run()` / `_evaluate_gate()` を拡張。

### 4.1 `_run_multi_seed(cfg, seeds, task_fn)`

- 4 seed を順次実行し、各 seed の `gross_roi` / `ic_mean` を抽出
- `_compute_convergence()` で 30K step 以降の ROI 変動を算出
- 集約結果 dict (`seed_results`, `convergence`, `raw_results`) を返却

### 4.2 `_compute_convergence(all_checkpoint_metrics, window_start=30000)`

- 全 seed のチェックポイントメトリクスから `window_start` 以降の ROI 値を収集
- `roi_variance_pct_after_30k = (max - min) × 100` を算出
- データ不足 (< 2 点) の場合は 0.0 を返却

### 4.3 `_evaluate_g2_from_results(results, thresholds)`

run_gate_check.py の G2 判定ロジックを dict 入力で再現。4 つの検査項目:

| Check | 条件 | デフォルト閾値 |
|---|---|---|
| **E1** positive_seed_ratio | gross_roi > 0 の seed 比率 ≥ 75% | `min_positive_seed_ratio: 0.75` |
| **E2** ic_seed_std | IC の seed 間標準偏差 ≤ 0.03 | `max_ic_seed_std: 0.03` |
| **E3** convergence | 30K 以降 ROI range ≤ 5% | `max_roi_variance_pct: 5.0` |
| **E4** worst_seed_roi | 最悪 seed の ROI > -2% | `worst_seed_min_roi: -0.02` |

全 4 検査が PASS → `gate_result: "PASS"`, いずれか FAIL → `gate_result: "FAIL"`

### 4.4 `run()` / `_evaluate_gate()` 拡張

- `run()`: `"G2" in gate and len(seeds) > 1` で `_run_multi_seed()` にディスパッチ
- `_evaluate_gate()`: G2 gate 検出時に `_evaluate_g2_from_results()` を呼び出し

---

## §5 テスト

**ファイル**: `tests/unit/v460/test_356_g2_sac_blockers.py` — **22 tests, ALL PASSED**

| Test Class | Tests | 検証内容 |
|---|---|---|
| `TestB1YamlExists` | 7 | YAML 存在・構造・seeds 数・特徴量・ハイパラ・env |
| `TestB3FeatureInjection` | 3 | feature_names 注入/非注入/env_info 記録 |
| `TestB4G2GateEvaluation` | 7 | E1-E4 個別 FAIL + 全 PASS + 空結果 + 境界値 |
| `TestConvergenceComputation` | 3 | 基本計算・空入力・window 外データ |
| `TestMultiSeedDispatch` | 2 | G2/G1 識別ロジック |

---

## §6 枝番付与 (356# → 356a/356b)

000# §5 の命名規則に従い、356# を枝番化:

| 枝番 | 種別 | ファイル | 内容 |
|---|---|---|---|
| 356a | plan | `356_ph3_plan_sac_training.md` | SAC 訓練計画 (旧名: `...with_vxxx_assets.md`) |
| 356b | rpt | `356_ph3_rpt_sac_asset_inventory.md` | SAC 資産インベントリ (ファイル名に `rpt` type 追加) |

同時に index.md の 336# (6 行重複) と 253# (rev+impl 結合) にも枝番を付与:
- 336 → 336a–336f
- 253 → 253a/253b

---

## §7 残課題

| ID | 内容 | 次ステップ |
|---|---|---|
| B2 | 特徴量精査 (12 → 最適セット) | P3A-3: FeatureRegistry 全量評価 |
| B5 | SAC 3-way (A/B/C) 方針 | 356a# で C 決定済、コード変更不要 |
| P3A-1 | 実データ取得 + 前処理 | data/ 配下のパイプライン構築 |
| P3A-2 | HeavyTradingEnv 統合テスト | feature_columns 注入の E2E 検証 |

---

## §8 変更ファイル一覧

| ファイル | 種別 | 内容 |
|---|---|---|
| `configs/v460/experiments/g2_sac_train.yaml` | NEW | B1: G2 SAC 訓練 YAML |
| `scripts/v460/lib/tasks/sac_train.py` | MOD | B3: feature_columns 注入 |
| `scripts/v460/run_experiment.py` | MOD | B4: multi-seed + convergence + G2 gate |
| `tests/unit/v460/test_356_g2_sac_blockers.py` | NEW | 22 tests |
| `docs/v460/356_ph3_plan_sac_training.md` | RENAME | 枝番化 (356a) |
| `docs/v460/356_ph3_rpt_sac_asset_inventory.md` | RENAME | 枝番化 (356b) + rpt type 追加 |
| `docs/v460/index.md` | MOD | 枝番更新 (253/336/356) |
| `docs/v460/357_ph3_impl_g2_sac_blockers.md` | NEW | 本ドキュメント |
