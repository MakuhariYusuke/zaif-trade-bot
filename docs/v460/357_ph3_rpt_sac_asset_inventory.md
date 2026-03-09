# 356# ph3 SAC Training 資産インベントリ

**Date**: 2026-03-09
**Purpose**: ph3 SAC Training に向けた全関連資産の棚卸し
**Scope**: v430–v459 系列 + v460

---

## 1. SAC Training Config 一覧

### 1.1 v460 設定 (現行)

| ファイル | 形式 | 概要 |
|---------|------|------|
| [configs/v460/base.yaml](../../configs/v460/base.yaml) | YAML | v460 全実験共通ベース |
| [configs/v460/gate_thresholds.yaml](../../configs/v460/gate_thresholds.yaml) | YAML | G0–G4 閾値定義 |
| [configs/v460/fill_test.yaml](../../configs/v460/fill_test.yaml) | YAML | Fill test 設定 |
| [configs/v460/experiments/g1_*.yaml](../../configs/v460/experiments/) | YAML | G1 XGBoost 実験 ×5 |

**v460 base.yaml の SAC ハイパラ** ([base.yaml](../../configs/v460/base.yaml) L48-54):
```yaml
sac:
  total_steps: 50000
  seeds: [42, 123, 456, 789]
  batch_size: 256
  lr: 0.0003
  gamma: 0.99
  tau: 0.005
```

> **注意**: v460 には `g2_sac_train.yaml` 実験 YAML がまだ存在しない（ph3 ブロッカー）

### 1.2 旧バージョン SAC Config (configs/ ルートレベル)

| ファイル | 形式 | 用途 |
|---------|------|------|
| [configs/sac_test_config.yaml](../../configs/sac_test_config.yaml) | YAML | v435 テスト用 |
| configs/sac_test_100steps.json | JSON | 100 step テスト |
| configs/sac_test_1ksteps.json | JSON | 1K step テスト |
| configs/sac_v434_2_integrated_config.json | JSON | v434 統合設定 |
| configs/sac_v446_base_template.json | JSON | v446 テンプレート |

**sac_test_config.yaml ハイパラ** ([sac_test_config.yaml](../../configs/sac_test_config.yaml) L44-53):
```yaml
sac_hyperparameters:
  learning_rate: 3.0e-04
  buffer_size: 1000000
  learning_starts: 1000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  ent_coef: 0.1
  target_update_interval: 1
  train_freq: 1
```

### 1.3 バージョン別 SAC Config (抜粋)

**v427** ([configs/v427/sac_v427_default_config.json](../../configs/v427/sac_v427_default_config.json)):
- `learning_rate: 0.0003`, `buffer_size: 50000`, `batch_size: 256`
- `gamma: 0.99`, `tau: 0.005`, `ent_coef: 0.01`, `target_entropy: -2.0`
- データ: `data/btc_jpy_real_dataset.csv`
- 環境: `initial_balance: 200,000`, `continuous_actions: true`

**v444** (configs/v444/ — 16+ バリエーション):
- asymmetric targets, balanced penalty, regime adaptation 等の実験
- `sac_v444_default.json`, `sac_v444_6_optimized_config.json` 等

**v446** ([configs/v446/sac_v446_base_config.json](../../configs/v446/sac_v446_base_config.json)):
```json
"learning_rate": 0.0003, "buffer_size": 1000000,
"batch_size": 256, "tau": 0.005, "gamma": 0.99,
"ent_coef": "auto", "ent_coef_init": 1.0, "target_entropy": "auto"
```
- `total_timesteps: 100000`, データ: `data/btc_jpy_5m_dataset.csv`

**v459** ([configs/v459/base/config.yaml](../../configs/v459/base/config.yaml) L18-29):
```yaml
sac_hyperparameters:
  gamma: 0.80              # 短期指向の割引率
  ent_coef: "auto"
  learning_rate: 3.0e-4
  batch_size: 256
  buffer_size: 100000
  learning_starts: 100
  tau: 0.005
  train_freq: 1
  gradient_steps: 1
```
- `total_timesteps: 10000` (検証用短期)
- 環境: `FastIntradayEnvV456`, `initial_balance: 10,000,000`



### 1.4 ハイパラ変遷サマリ

| パラメータ | v427 | v446 | v459 | v460 |
|-----------|------|------|------|------|
| learning_rate | 3e-4 | 3e-4 | 3e-4 | 3e-4 |
| batch_size | 256 | 256 | 256 | 256 |
| buffer_size | 50K | 1M | 100K | (未指定→動的) |
| gamma | 0.99 | 0.99 | **0.80** | 0.99 |
| tau | 0.005 | 0.005 | 0.005 | 0.005 |
| ent_coef | 0.01 | auto | auto | (未指定) |
| total_steps | 10K | 100K | 10K | 50K |
| seeds | 1 | 1 | [42,123,456,777] | [42,123,456,789] |

> **所見**: `lr`, `batch_size`, `tau` は全バージョンで安定。`gamma` は v459 で 0.80 に変更されたが v460 では 0.99 へ回帰。`buffer_size` は v460 の `sac_train.py` で `min(raw, 2×timesteps)` に動的調整される。

---

## 2. 過去の Training Results & Models

### 2.1 保存済みモデル (models/)

| ファイル | バージョン | 説明 |
|---------|-----------|------|
| models/sac_model.zip | 不明 | 汎用 SAC モデル |
| models/e0_diag_model.zip | v459 | Phase E0 診断モデル |
| models/e0_test_model.zip | v459 | Phase E0 テストモデル |
| models/e1_cf1_zero_cost_model.zip | v459 | Counterfactual 1 (コスト 0) |
| models/e1_cf3_zero_cost_low_thr_model.zip | v459 | CF3 (コスト 0, 低閾値) |
| models/e1_oracle_cost0.001_model.zip | v459 | Oracle (0.1% コスト) |
| models/e1_oracle_cost0.0_model.zip | v459 | Oracle (0% コスト) |
| models/e2a_base_model.zip | v459 | E2α ベース |
| models/e2a_hold5/15/30_model.zip | v459 | E2α hold 変形 |
| models/e2a_realcost_model.zip | v459 | E2α 実コスト |
| models/e2b_seed42/123/456/789_model.zip | v459 | E2β 4-seed |

**models/v460/ ディレクトリ**: SAC モデルなし。SkipGate 関連 .pkl のみ。
- `skip_gate.pkl`, `skip_gate_as.pkl`, `pnl_regressor_070_candidate3.pkl` 等 (ph2 成果物)

> **結論**: v460 SAC 学習モデルは未生成。v459 の SAC モデルは 5 ファイル (.zip) 残存。

### 2.2 results/v460/ (v460 実験結果)

| ファイル | 内容 |
|---------|------|
| g0_result.json | G0 データゲート結果 |
| v460_g1info_seed42_*.json | G1 XGBoost 4 回分 |
| 065_* | ハイパラスイープ・WF 結果等 |
| fill_test/ | G1.1 約定テスト結果 |
| ab_regime/ | AB テスト結果 |
| sac_dependency_graph.md | SAC 依存関係図 |

### 2.3 results/ (旧バージョン SAC 結果 — 抜粋)

v444 関連が最多 (11 ファイル: `sac_v444_training_results_*.json`)。v430, v435, v445, v501-v503 等も存在。

| バージョン | ファイル例 | 件数 |
|-----------|-----------|------|
| v444 | sac_v444_training_results_20251106_*.json | 11 |
| v445 | sac_v445.3_*, sac_v445.4_* | 7 |
| v430 | sac_v430_backtest_*.json | 6 |
| v427 | sac_v426_*, sac_v427_* | 3 |
| v501-503 | paper_trade_sac_v50x_*.json | 3 |

### 2.4 training_results/

**空ディレクトリ** — 訓練結果は results/ に統合されている。

---

## 3. Feature Pipeline 資産

### 3.1 ztb/features/ ディレクトリ構造

```
ztb/features/
├── core/
│   ├── base.py           # 基底クラス
│   ├── engine.py         # 計算エンジン
│   └── registry.py       # FeatureRegistry (832行)
├── generators/
│   ├── technical/
│   │   ├── momentum/     # RSI, MACD, CCI, Stochastic, ROC, Williams_R, ...
│   │   ├── oscillator/
│   │   ├── trend/        # KAMA, PSAR, HeikinAshi, ...
│   │   ├── volatility/   # ZScore, ...
│   │   └── volume/       # Chaikin_AD, ReturnMA, ...
│   ├── adaptive/
│   │   └── selection.py  # AdaptiveFeatureSelector
│   └── multi_timeframe/  # MTF feature system
├── microstructure.py     # ★ v460 10 特徴量 (リアル板データ版)
├── hft_proxies.py        # HFT proxy 特徴量
├── scalping.py           # スキャルピング特徴量
├── registry.py           # 旧 registry (互換)
├── feature_engine.py
├── feature_set_config.py
├── feature_set_manager.py
├── unified_feature.py
└── ...
```

### 3.2 FeatureRegistry の仕組み

**実装**: [ztb/features/core/registry.py](../../ztb/features/core/registry.py)

- クラスメソッドベースのシングルトンレジストリ
- `@FeatureRegistry.register("FeatureName")` デコレータで特徴量関数を登録
- `compute_features_batch(df, feature_names)` でバッチ計算
- チャンク処理・並列処理・メモリ最適化対応
- ~191 個の OHLCV ベース技術指標が登録済み

**登録済み特徴量の例**:
- Momentum: RSI, RSI_M1/M5/M15/H1/H4/D1, MACD, CCI, Stochastic, ROC, Williams_R, Ultimate_Oscillator
- Trend: KAMA, PSAR, PSAR_Trend, HeikinAshi_Color (MTF 変形含む)
- Volume: Chaikin_AD_Oscillator, ReturnMA_Short, ReturnMA_Medium
- Volatility: ZScore

### 3.3 v460 の 10 Microstructure 特徴量

**定義**: [scripts/v460/build_features.py](../../scripts/v460/build_features.py) L63-74 + [ztb/features/microstructure.py](../../ztb/features/microstructure.py) L131-142

```python
V460_FEATURES = [
    "bid_ask_spread",       # (high-low)/mid (proxy) / (ask-bid)/mid (real)
    "depth_imbalance",      # CLV proxy / (bid_vol_5-ask_vol_5) (real)
    "trade_flow_imbalance", # signed volume proxy / (buy-sell)/total (real)
    "vwap_deviation",       # (close - VWAP_proxy) / close
    "trade_intensity",      # volume / rolling_mean(volume)
    "order_flow_toxicity",  # VPIN 近似: |buy-sell|/total
    "price_impact",         # |Δclose| / volume, smoothed
    "micro_return_vol",     # log return rolling std
    "bid_depth_slope",      # buy_vol / bid_range
    "ask_depth_slope",      # sell_vol / ask_range
]
```

**2 モード**:
- **proxy**: OHLCV → `build_proxy_features()` → 技術的 proxy (ph1 検証用)
- **real**: raw orderbook/trades JSONL.gz → `add_microstructure_features()` → 実データ特徴量

### 3.4 v460 特徴量と FeatureRegistry の橋渡し — 断絶あり

> **355# B2 ブロッカー**: v460 の 10 microstructure 特徴量と FeatureRegistry の ~191 OHLCV 特徴量の間に adapter/injection 機構がない。

- `FeatureRegistry` は `@register` デコレータで OHLCV→技術指標を算出
- v460 microstructure 特徴量は `build_features.py` / `microstructure.py` で独自生成
- SAC 訓練時にどちらの特徴量系統を使うかの設計判断が必要
- `sac_train.py` は `cfg.features.selected` からカラム名リストを取得し、Parquet の列を直接使用 → FeatureRegistry は経由しない

---

## 4. SAC Trainer 実装

### 4.1 SACTrainer (統合訓練器)

**ファイル**: [ztb/training/unified_trainer/algorithms/sac_trainer.py](../../ztb/training/unified_trainer/algorithms/sac_trainer.py) (1969 行)

**コンストラクタ** (L61-98):
```python
class SACTrainer(BaseAlgorithmTrainer):
    def __init__(
        self,
        config: ConfigDict,
        env: HeavyTradingEnv | None = None,
        logger: logging.Logger | None = None,
        gradient_accumulation_steps: int = 1,
        system_optimizer: object | None = None,
        optimizer_tracker: OptimizerFeatureTracker | None = None,
    ):
```

- `BaseAlgorithmTrainer` を継承
- SB3 `SAC` モデルをラップ
- `TrainingCheckpointManager` で 1000 step 毎にチェックポイント保存
- `MarketRegimeClassifier` による適応的学習 (オプション)
- `StructuredLogger` による JSON ログ
- feature_set の解決・伝播ロジック内蔵

### 4.2 v460 task_sac_train (軽量版)

**ファイル**: [scripts/v460/lib/tasks/sac_train.py](../../scripts/v460/lib/tasks/sac_train.py) (372 行)

**読み取る config キー**:
```
cfg.training.total_timesteps     → 50,000 (default)
cfg.sac_hyperparameters.*        → lr, buffer_size, batch_size, tau, gamma, ent_coef 等
cfg.data.v460_features_path      → 特徴量 Parquet パス
cfg.data.ohlcv_path              → フォールバック
cfg.features.selected            → 使用特徴量カラム名リスト
cfg.environment                  → EnvironmentConfig kwargs
cfg.output.model_dir             → models/v460 (default)
cfg.training.checkpoint_interval → 10,000 (default)
cfg.evaluation.n_episodes        → 1 (default)
cfg.seed                         → 42 (default)
```

**処理フロー** ([sac_train.py](../../scripts/v460/lib/tasks/sac_train.py)):
1. `load_parquet(data_path)` でデータロード
2. replay buffer を `min(raw_buffer, max(timesteps×2, 10000))` に動的調整
3. `_create_training_env(df, cfg)` → `HeavyTradingEnv(df=df, config=EnvironmentConfig(**env_cfg))`
4. `_create_sac_model(env, sac_cfg, seed)` → SB3 `SAC("MlpPolicy", env, ...)`
5. `_train_with_checkpoints()` → checkpoint_interval 毎に Learn
6. `_evaluate_trained_model()` → deterministic predict, mean_reward 収集
7. `model.save(models/v460/sac_v460_seed{seed}.zip)`
8. `_save_model_schema()` → FeatureSchemaManager で特徴量メタデータ保存

### 4.3 学習定数 (training/constants.py)

| 定数 | 値 |
|------|-----|
| DEFAULT_LEARNING_RATE_SAC | 3e-4 |
| DEFAULT_BATCH_SIZE_SAC | 256 |
| DEFAULT_BUFFER_SIZE_SAC | 1,000,000 |
| DEFAULT_LEARNING_STARTS_SAC | 1,000 |
| DEFAULT_GAMMA | 0.99 |
| DEFAULT_TAU | 0.005 |
| DEFAULT_TRAIN_FREQ | 1 |
| DEFAULT_GRADIENT_STEPS | 1 |
| DEFAULT_TARGET_UPDATE_INTERVAL | 1 |
| DEFAULT_TOTAL_TIMESTEPS_SAC | 100,000 |

---

## 5. HeavyTradingEnv 特徴量ハンドリング

### 5.1 EnvironmentConfig.feature_names

**ファイル**: [ztb/trading/environment/utils/config.py](../../ztb/trading/environment/utils/config.py) L201

```python
@dataclasses.dataclass
class EnvironmentConfig:
    feature_set: str = "full"
    feature_names: list[str] | None = None  # Explicit feature list (overrides feature_set)
    correlation_reduction: bool = True
    target_feature_count: int | None = None
    use_continuous_actions: bool = False     # True for SAC
    initial_portfolio_value: float = 200_000.0
    ...
```

### 5.2 特徴量セットアップフロー

**ファイル**: [ztb/trading/environment/heavy_env/mixins/initialization.py](../../ztb/trading/environment/heavy_env/mixins/initialization.py) L200-

`_initialize_features_and_spaces()` の動作:

1. `config.feature_names` が設定されている場合:
   - そのリストをそのまま `self.features` に設定
   - DataFrame に存在しないカラムがあれば `ValueError`
   - **→ カスタム特徴量カラムを渡せる** ✅

2. `config.feature_names` が `None` の場合:
   - DataFrame の全カラム (ts/timestamp/exchange/pair/episode_id 除外) を自動検出
   - `feature_set` に応じて `FeatureSetConfig` フィルタリング
   - MTF 特徴量の動的追加 (オプション)
   - 相関削減 (`correlation_reduction=True` のとき)
   - `max_features_limit` 制限

3. その後:
   - Adaptive feature selection (オプション)
   - observation_space = `Box(low, high, shape=(n_features,))`

### 5.3 sac_train.py からの特徴量渡し —  現状の課題

[sac_train.py](../../scripts/v460/lib/tasks/sac_train.py) L169-174:
```python
feature_columns = [str(col) for col in selected_raw] if isinstance(selected_raw, list) else []
env_config = EnvironmentConfig(**env_cfg) if env_cfg else EnvironmentConfig()
env = HeavyTradingEnv(df=df, config=env_config)
```

**問題**: `feature_columns` は計算されるが `EnvironmentConfig` に渡されていない。`env_cfg` dict に `feature_names` キーが含まれていない限り、HeavyTradingEnv は DataFrame 全カラムを自動検出する。

**修正方針**: `env_config.feature_names = feature_columns` を明示的に設定する必要がある。

---

## 6. Gate G2 インフラ

### 6.1 G2 閾値 (gate_thresholds.yaml)

```yaml
g2_train:
  min_positive_seed_ratio: 0.75   # 3/4 seed で gross > 0
  max_ic_seed_std: 0.03           # IC の seed 間 σ 上限
  convergence_window_start: 30000 # 収束判定開始地点
  max_roi_variance_pct: 5.0       # 30K 以降の ROI 変動上限
  worst_seed_min_roi: -0.02       # worst seed の ROI 下限
```

### 6.2 run_g2_judgment() 実装

**ファイル**: [scripts/v460/run_gate_check.py](../../scripts/v460/run_gate_check.py) L248-318

**入力 JSON フォーマット**:
```json
{
  "seed_results": [
    {"seed": 42, "gross_roi": 0.05, "ic_mean": 0.03},
    {"seed": 123, "gross_roi": -0.01, "ic_mean": 0.02},
    ...
  ],
  "convergence": {
    "roi_variance_pct_after_30k": 3.5
  }
}
```

**4 つのチェック**:
| # | チェック | 閾値 | 判定 |
|---|---------|------|------|
| E1 | positive_seed_ratio | ≥ 0.75 | gross_roi > 0 の seed / 全 seed |
| E2 | ic_seed_std | ≤ 0.03 | ic_mean の stdev |
| E3 | convergence | ≤ 5.0% | 30K 以降の ROI 変動 |
| E4 | worst_seed_roi | > -0.02 | 最低 seed の ROI |

**CLI**:
```bash
python scripts/v460/run_gate_check.py --gate G2 --results-path results/v460/g2_train_results.json
```

---

## 7. 既存 SAC Training ドキュメント

### 7.1 docs/algorithms/ (v421–v427)

| ドキュメント | 内容 |
|-------------|------|
| [SAC_v427_TRAINING_README.md](../../docs/algorithms/SAC_v427_TRAINING_README.md) | v427 SAC 訓練手順 |
| [SAC_v427_LEARNING_PLAN.md](../../docs/algorithms/SAC_v427_LEARNING_PLAN.md) | v427 学習計画 |
| [SAC_V425_IMPROVEMENT_PLAN.md](../../docs/algorithms/SAC_V425_IMPROVEMENT_PLAN.md) | v425 改善計画 |
| [SAC_V424_DEEP_ANALYSIS_REPORT.md](../../docs/algorithms/SAC_V424_DEEP_ANALYSIS_REPORT.md) | v424 深層分析 |
| [SAC_V421_IMPROVEMENT_PLAN.md](../../docs/algorithms/SAC_V421_IMPROVEMENT_PLAN.md) | v421 改善計画 |
| [SAC_TUNING_ANALYSIS_REPORT.md](../../docs/algorithms/SAC_TUNING_ANALYSIS_REPORT.md) | チューニング分析 |
| [SAC_ROOT_CAUSE_ANALYSIS.md](../../docs/algorithms/SAC_ROOT_CAUSE_ANALYSIS.md) | 根本原因分析 |
| [SAC_PARAMETER_TUNING_GUIDE.md](../../docs/algorithms/SAC_PARAMETER_TUNING_GUIDE.md) | パラメータガイド |
| [SAC_ENVIRONMENT_DIAGNOSTICS_REPORT.md](../../docs/algorithms/SAC_ENVIRONMENT_DIAGNOSTICS_REPORT.md) | 環境診断 |
| [SAC_ENTROPY_ANALYSIS.md](../../docs/algorithms/SAC_ENTROPY_ANALYSIS.md) | エントロピー分析 |
| [SAC_BIAS_REPORT.md](../../docs/algorithms/SAC_BIAS_REPORT.md) | バイアス報告 |

### 7.2 docs/v459/ Phase 3 関連 (10 ファイル)

| ドキュメント | 内容 |
|-------------|------|
| 24_phase3_specification.md | Phase 3 仕様書 |
| 25_phase3_specification_review.md | 仕様レビュー |
| 27_phase3_implementation_plan_phase4_ready.md | 実装計画 |
| 28_phase3_day1_implementation_complete.md | Day 1 完了 |
| 29_phase3_existing_implementation_review.md | 既存実装レビュー |
| 30_phase3_day3_reward_config_complete.md | 報酬設定完了 |
| 31_phase3_action_space_analysis.md | アクション空間分析 |
| 32_phase3_action_space_fix_complete.md | アクション空間修正 |
| 33_phase3_execution_status.md | 実行ステータス |
| 43_phase3.5_verification_results.md | 検証結果 |

### 7.3 v460 プロジェクト提案書の ph3/G2 記載

**ファイル**: [docs/v460/000_ph0_plan_project_proposal.md](../../docs/v460/000_ph0_plan_project_proposal.md) §2, §3.4

**Phase 定義**: `ph3 = SAC 学習安定性検証 → G2-train Gate`

**§3.4 G2-train**:
- 4 seed × 50K steps
- gross > 0 の seed 比率 ≥ 75%
- IC の seed 間 σ ≤ 0.03
- 30K 以降で ROI 変動 ≤ 5%
- worst-seed ROI > −2%
- FAIL 時: 学習器・報酬設計見直し。G1-info 再確認。

---

## 8. データパイプライン

### 8.1 data/v460/ 構造

```
data/v460/
├── features/
│   ├── btc_jpy_1m_v460_features.parquet      # proxy 特徴量 (OHLCV 由来)
│   └── btc_jpy_1m_v460_real_features.parquet  # real 特徴量 (板/約定由来)
└── raw/
    ├── orderbook/
    │   └── 20260213.jsonl.gz ... 20260309.jsonl.gz (24 日分)
    └── trades/
        └── 20260213.jsonl.gz ... 20260309.jsonl.gz (23 日分)
```

### 8.2 データ生成フロー

```
[生データ収集]
MarketDataCollector.collect_tick()  → orderbook snapshot + trades
    ↓ flush_raw()
data/v460/raw/orderbook/{date}.jsonl.gz
data/v460/raw/trades/{date}.jsonl.gz
    ↓ aggregate_to_1min()
1 分集約 DataFrame (mid_price, spread, bid_vol_5, ask_vol_5, ...)
    ↓ add_microstructure_features()      [real モード]
    ↓ build_proxy_features()             [proxy モード]
data/v460/features/btc_jpy_1m_v460_*.parquet
    ↓
SAC Training (task_sac_train → load_parquet → HeavyTradingEnv)
```

### 8.3 MarketDataCollector

**ファイル**: [ztb/data/market_data_collector.py](../../ztb/data/market_data_collector.py) L136-

- `IBroker` adapter 経由で Coincheck/Bitflyer/Zaif からデータ取得
- `collect_tick()`: orderbook (depth=10) + recent_trades (limit=100)
- `flush_raw()`: JSONL.gz 日次ローテーション保存
- `aggregate_to_1min()`: raw JSONL → 1 分足 DataFrame → Parquet
- 出力カラム: mid_price, spread, best_bid, best_ask, bid_vol_5, ask_vol_5, depth_imbalance, buy_volume, sell_volume, trade_count, vwap, trade_flow_imbalance

### 8.4 既存 OHLCV データ (data/ ルート)

| ファイル | サイズ/行数 | 用途 |
|---------|-----------|------|
| btc_jpy_1m_v451_optimized_features.parquet | — | v460 proxy 特徴量のソース |
| btc_jpy_1m_v459_expanded_features.parquet | — | v459 拡張特徴量 |
| btc_jpy_1m_v451.csv | — | v451 生データ |
| btc_jpy_real_dataset.csv | — | v427/v446 学習用 |
| btc_jpy_5m_dataset.csv | — | v446 5分足学習用 |

---

## 9. ph3 ブロッカーサマリ (355# 再掲 + 追加)

| # | ブロッカー | 重要度 | 状態 |
|---|----------|--------|------|
| **B1** | g2_sac_train.yaml 実験 YAML 未作成 | Critical | 未着手 |
| **B2** | 特徴量次元体系の断絶 (microstructure ↔ FeatureRegistry) | High | 設計判断必要 |
| **B3** | sac_train.py の feature_columns が EnvironmentConfig に未渡し | Medium | コード修正必要 |
| **B4** | G2 結果 JSON の convergence フィールド生成コード未実装 | Medium | _train_with_checkpoints に ROI 追跡要 |
| **B5** | replay buffer 動的調整の妥当性検証 | Low | テスト必要 |

---

## 10. 推奨アクション

1. **g2_sac_train.yaml 作成**: base.yaml の sac セクション + features.selected + environment セクションを統合
2. **特徴量方針決定**: v460 microstructure 10 特徴量のみ (obs_dim=10) vs FeatureRegistry 併用
3. **sac_train.py 修正**: `env_config.feature_names = feature_columns` の明示設定
4. **convergence 追跡実装**: checkpoint 毎の ROI 計算 + roi_variance_pct_after_30k 算出
5. **4-seed 訓練実行**: `for seed in [42, 123, 456, 789]: run_experiment --seed {seed}`
