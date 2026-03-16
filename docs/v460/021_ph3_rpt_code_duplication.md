# 021# ph3 — コード重複 & リファクタリング分析レポート

> 調査日: 2026-02-14
> 対象: `ztb/training/`, `ztb/trading/`, `scripts/v460/`, `ztb/utils/`
> 参照: `018#`(実装済み修正)
> 備考: 元 `018_duplication_refactoring_analysis.md` — 018# と番号重複のため 021# に分離

---

## 1. SAC Trainer 実装の重複 (Critical — 最重要統合対象)

### 発見: **7つの独立した SAC 訓練実装**が存在

| # | ファイル | 行数 | 参照元 | 状態 |
|---|---------|------|--------|------|
| 1 | `ztb/training/unified_trainer/algorithms/sac_trainer.py` | **1,759** | v459/v460 scripts, tests (主流) | **Active — primary** |
| 2 | `ztb/training/trainers/sac_trainer.py` | **735** | `core/algorithm_trainer.py`, multimodal, online_learning | Active — legacy |
| 3 | `ztb/training/sac_trainer.py` | **317** | `unified_resume.py` のみ | Semi-dead |
| 4 | `scripts/v460/lib/tasks/sac_train.py` | **300** | `run_experiment.py` | Active — 新規追加(017#) |
| 5 | `ztb/training/sac_v427_advanced_trainer.py` | **504** | なし(壊れた import あり) | **Dead** |
| 6 | `ztb/training/train_v430_advanced.py` | **578** | なし | **Dead** |
| 7 | `ztb/training/train_sac_v432_unified.py` | **88** | なし | **Dead** |

**加えて**:
- `ztb/training/algorithms/sac/sac_algorithm.py` (923行) — SACアルゴリズム低レイヤ
- `ztb/training/sac.py` (414行) — 旧スクリプト
- `ztb/training/sac_utils.py` (307行) + `ztb/training/utils/sac_utils.py` (190行) — **utilsも2重**
- `ztb/training/sac_v430_training_optimizations.py` (334行) — 旧最適化
- `ztb/training/adaptive_sac_core.py` (617行) — V433適応学習

### 共通パターン (全実装に重複):
1. **SB3 SAC モデル生成**: `SAC("MlpPolicy", env, learning_rate=..., buffer_size=..., ...)` — 少なくとも4ファイルで独立実装
2. **チェックポイント保存**: `create_checkpoint_callback()` を呼ぶロジック — 3箇所
3. **モデル保存**: `model.save(str(path))` + metadata JSON — 4箇所
4. **訓練ループ**: `model.learn(total_timesteps=..., callback=..., reset_num_timesteps=False)` — 5箇所
5. **メトリクス収集**: `model.logger.name_to_value` からの critic_loss/actor_loss 抽出 — 3箇所

### 推奨:
```
統合先: ztb/training/unified_trainer/algorithms/sac_trainer.py (既に 1,759行で巨大)
        ↓ リファクタリング後
        sac_trainer.py (コア訓練ロジック ~600行)
        sac_config.py  (設定バリデーション ~200行)
        sac_analysis.py (結果解析 ~200行)
        sac_callbacks.py (メトリクス・チェックポイント ~300行)
        
削除候補:
  - ztb/training/sac_trainer.py → unified_resume.py の import を #1 に変更
  - ztb/training/sac_v427_advanced_trainer.py → 壊れた import、参照なし
  - ztb/training/train_v430_advanced.py → 参照なし
  - ztb/training/train_sac_v432_unified.py → 参照なし
  - ztb/training/train_sac_v432_1_advanced_position_management.py → 参照なし
```

---

## 2. 環境作成ロジックの重複 (High)

### 発見: `HeavyTradingEnv` の生成が **15箇所以上**で独立実装

| ファイル | 行 | パターン |
|---------|------|---------|
| `scripts/v460/lib/tasks/sac_train.py` | L124-133 | `HeavyTradingEnv(df=df, config=EnvironmentConfig(**env_cfg))` |
| `ztb/utils/backtest_init_utils.py` | L72, L107 | `HeavyTradingEnv(data, env_config)` |
| `ztb/training/train_v430_full.py` | L116 | `HeavyTradingEnv(df=df, config=env_config_obj, random_start=True)` |
| `ztb/training/utils/training_utils.py` | L39 | `HeavyTradingEnv(df=df, config=config)` |
| `ztb/training/utils/simple_reward.py` | L50 | `HeavyTradingEnv(...)` |
| `ztb/training/unified_trainer/trainer.py` | L2306 | `HeavyTradingEnv(...)` (動的import) |
| `ztb/analysis/evaluator/evaluator.py` | L162 | `_create_env()` |
| `ztb/analysis/evaluation/paper_trading_evaluator.py` | L74 | `create_environment()` |
| `ztb/evaluation/paper_trading.py` | L42 | `_create_environment()` |
| `scripts/v456/train_mlp_v456.py` | L112 | `create_environment()` |
| `scripts/v456/train_mlp_v456_improved.py` | L177 | `create_environment()` |
| `scripts/v456/train_mlp_v456_integrated.py` | L177 | 同上 |
| `scripts/v456/train_mlp_v456_phase2_complete.py` | L193 | 同上 |
| `scripts/v444/train/train_sac_v444_2_simple.py` | L70 | 同上 |
| `ztb/training/binary_search/base_optimizer.py` | L324 | `create_environment()` |

### `EnvironmentConfig` の生成も 3箇所以上で散在:
- `scripts/v460/lib/tasks/sac_train.py` L131: `EnvironmentConfig(**env_cfg)`
- `ztb/training/unified_trainer/trainer.py` L2408: `EnvironmentConfig(**env_config_dict)`
- `ztb/training/config/configuration_manager.py` L364: `EnvironmentConfig(...)`
- `ztb/training/train_v430_full.py` L106: `EnvironmentConfig(...)`

### 推奨:
```python
# ztb/training/env_factory.py (新規 — 統一ファクトリ)

def create_training_env(
    df: pd.DataFrame,
    env_cfg: dict | EnvironmentConfig | None = None,
    *,
    random_start: bool = False,
    wrap_vec: bool = False,
) -> HeavyTradingEnv | DummyVecEnv:
    """全訓練・バックテスト・評価で共有する環境生成."""
    ...
```

---

## 3. `load_model` の重複 (Medium)

### 発見: `ztb/utils/training_utils.py` に **`load_model` が2つ定義** されている

| 行 | シグネチャ | 備考 |
|----|----------|------|
| L180 | `load_model(model_path, algorithm=None, verbose=True)` | 自動検出付き、Optional返却 |
| L414 | `load_model(model_path, algorithm="SAC")` | 簡易版、例外送出 |

同一ファイル内で同名関数が2回定義されており、**L414 が L180 を上書きしている**。
呼出元は L414 版のみが有効であり、L180 の高機能版（自動検出・verbose・None安全）は**到達不能デッドコード**。

### 推奨:
- L414 を削除し、L180 の高機能版に統一
- `backtest_init_utils.py` 等の呼出元テストを確認

---

## 4. God Object: `UnifiedTrainer` (Critical)

### 発見: `ztb/training/unified_trainer/trainer.py` — **2,835行、65メソッド**

責務が過大:
- 訓練実行 (SAC/PPO/DQN 全アルゴリズム)
- 連合学習 (`_execute_federated_training`, `_federated_average`)
- 混合精度 (`_apply_mixed_precision`, `_step_optimizer`)
- 分散訓練 (`_setup_distributed_training`)
- アンサンブル管理 (`_initialize_ensemble_system`, `_setup_ensemble_training`)
- V433適応学習 (`_initialize_v433_components`, 7 methods)
- メモリ監視 (`_start_memory_monitoring`, `_monitor_training_memory`)
- 異常検知 (`_run_anomaly_detection`)
- メタ学習 (`_run_meta_learning_adaptation`)
- 継続学習 (`_run_continual_learning`, `_prepare_task_data`)
- マルチ期間バックテスト (`run_multi_period_backtest`, `_identify_market_periods`, etc.)
- 特徴一致性バリデーション (`_validate_feature_consistency` ~150行)
- 環境作成 (`_create_v433_training_environment`, `_create_backtest_environment`)

### 推奨 — 分割案:
```
UnifiedTrainer (コア: ~400行)
  ├── FederatedTrainingMixin   → ztb/training/mixins/federated.py
  ├── AdaptiveTrainingMixin    → ztb/training/mixins/adaptive_v433.py  
  ├── EnsembleTrainingMixin    → (既存 ensemble_mixin.py と統合)
  ├── MemoryMonitorMixin       → ztb/training/mixins/memory_monitor.py
  ├── AdvancedFeaturesMixin    → ztb/training/mixins/advanced_features.py
  ├── BacktestRunnerMixin      → ztb/training/mixins/backtest_runner.py
  └── FeatureValidator         → ztb/training/validation/feature_validator.py
```

---

## 5. God Object: `SACTrainer` (algorithms/) (High)

### 発見: `ztb/training/unified_trainer/algorithms/sac_trainer.py` — **1,759行、33メソッド**

`_execute_sac_training()` 単体が **~800行** あり、以下を内包:
- データローディング
- 環境作成・ラップ
- SB3モデル生成 / チェックポイントリストア
- コールバック設定
- 訓練実行
- モデル保存
- 結果分析
- エラーリカバリ (`_attempt_emergency_save`, `_retry_training_with_reduced_params`)

### 推奨:
```
SACTrainer (~400行)
  ├── SACModelFactory     → create_sac_model(), load_model()
  ├── SACCallbackBuilder  → _setup_callbacks(), checkpoint config
  ├── SACResultAnalyzer   → analyze_results(), _calculate_final_action_distribution()
  └── SACErrorRecovery    → _attempt_error_recovery(), _cleanup_on_memory_error()
```

---

## 6. `scripts/v460/lib/tasks/sac_train.py` の重複 (017# で追加)

### 発見: 既存の SACTrainer ロジックを部分的に再実装

| 機能 | `sac_train.py` (v460) | 既存 `SACTrainer` (unified) |
|------|----------------------|----------------------------|
| モデル生成 | `_create_sac_model()` L155-177 | `_execute_sac_training()` 内で同等ロジック |
| 環境生成 | `_create_training_env()` L116-152 | 複数箇所で類似実装 |
| チェックポイント訓練 | `_train_with_checkpoints()` L184-213 | `TrainingProgressCallback` が同等機能 |
| 評価 | `_evaluate_trained_model()` L216-257 | `validate_training()` で類似実装 |
| Schema保存 | `_save_model_schema()` L259-300 | なし(新機能 — ✅ 正当) |

### 推奨:
- `_create_sac_model` → `ztb/training/sac_model_factory.py` に統一
- `_create_training_env` → 環境ファクトリ (#2) に統一
- `_train_with_checkpoints` → 既存コールバック機構に委譲
- `_evaluate_trained_model` → `ztb/evaluation/` の既存評価器を使用
- `_save_model_schema` → そのまま維持 (v460 新機能)

---

## 7. `sac_utils` の重複 (Medium)

### 発見: SAC ユーティリティが2ファイルに分散

| ファイル | 行数 |
|---------|------|
| `ztb/training/sac_utils.py` | 307 |
| `ztb/training/utils/sac_utils.py` | 190 |

両方とも SAC ハイパーパラメータ関連のヘルパーを含む。

### 推奨:
- `ztb/training/utils/sac_utils.py` に統合
- `ztb/training/sac_utils.py` は import リダイレクトのみ残し、非推奨化

---

## 8. デッドコード (Medium)

### 8.1 到達不能ファイル (参照0件)

| ファイル | 行数 | 理由 |
|---------|------|------|
| `ztb/training/sac_v427_advanced_trainer.py` | 504 | 壊れた import (`SACv427MarketAdaptiveSystem`) |
| `ztb/training/train_v430_advanced.py` | 578 | 外部参照なし |
| `ztb/training/train_sac_v432_unified.py` | 88 | 外部参照なし |
| `ztb/training/train_sac_v432_1_advanced_position_management.py` | 70 | 外部参照なし |
| `ztb/training/sac_v430_training_optimizations.py` | 334 | 外部参照なし |
| `ztb/training/scripts/train_sac_v395*.py` (6ファイル) | ~240 | 旧スクリプト群 |
| `ztb/training/scripts/train_sac_v431_advanced.py` | 0 | **空ファイル** |
| `ztb/training/sac_utils_scripts.py` | 401 | 外部参照未確認 |

### 8.2 `load_model` 上書き (L180 が L414 に隠蔽)
→ セクション3 参照

### 推奨:
- `archived/` に移動、または削除
- 推定回収: **~2,600行** 削減

---

## 9. 巨大ファイル (>500行) — リファクタリング優先度

| ファイル | 行数 | 推奨 |
|---------|------|------|
| `unified_trainer/trainer.py` | **2,835** | Mixin 分割 (#4) |
| `unified_optimizer.py` | **2,293** | 重複する reward optimizer 統合 |
| `unified_trainer/algorithms/sac_trainer.py` | **1,759** | 機能分離 (#5) |
| `reward_function_optimizer.py` | **1,485** | Strategy パターン適用 |
| `binary_search/base_optimizer.py` | **1,099** | Template Method 分離 |
| `algorithms/sac/sac_algorithm.py` | **923** | SACTrainer と責務重複確認 |
| `scripts/paper_trade.py` | **936** | 環境生成ロジック分離 |
| `callbacks/base/learning_callback.py` | **778** | コンポジション化 |

---

## 10. `config_loader` / `config_manager` の重複 (Low-Medium)

| ファイル | 役割 |
|---------|------|
| `ztb/utils/config_manager.py` | `ConfigManager` クラス (264行) |
| `ztb/utils/config_loader.py` | `load_config()` 等 (112行) |
| `ztb/utils/config_utils.py` | `load_config_unified()` |
| `scripts/v460/lib/config_loader.py` | v460 専用ローダー |
| `ztb/training/unified_trainer/config.py` | `UnifiedTrainerConfig` |
| `ztb/training/unified_trainer/config_manager.py` | 別の ConfigManager |
| `ztb/training/unified_trainer/components/config_manager.py` | さらに別の ConfigManager |
| `ztb/training/config/configuration_manager.py` | `ConfigurationManager` (630行) |

**8ファイルに設定ロジックが散在**。

### 推奨:
- 設定ローディング: `ztb/utils/config_loader.py` に一本化
- 設定バリデーション: `ztb/config/schemas/` に Pydantic/dataclass モデル集約
- trainer 固有設定: `ztb/training/config/` のみ

---

## 11. 命名不一致 (Low)

| パターン | 例 |
|---------|-----|
| Trainer クラス名 | `SACTrainer`, `SACAlgorithmTrainer`, `SACv430AdvancedTrainer`, `SACv427AdvancedTrainer`, `SACv435Trainer` |
| train メソッド | `train()`, `run()`, `run_training()`, `train_sac_v432()` |
| 環境作成関数 | `_create_training_env`, `_create_env`, `create_environment`, `_create_v433_training_environment`, `make_env` |
| configローダー | `_load_config`, `load_config`, `load_config_unified`, `get_training_config` |

---

## 12. 循環 Import リスク (Low)

- `ztb/training/sac_trainer.py` L29: `from ztb.training.unified_trainer import UnifiedTrainer`
  - 同時に `unified_trainer/trainer.py` が `ztb.training.trainers.sac_trainer` を動的 import
  - **潜在的な循環 import** (現在は lazy import でガードされているが脆弱)

- `ztb/training/unified_resume.py` L199: `from ztb.training.sac_trainer import SACTrainer`
  - この `SACTrainer` は `unified_trainer` の `SACTrainer` とは**別クラス**
  - 同名異クラスによる混乱リスク

---

## 優先実装計画

### P0 (即時 — 017# 残課題として)
1. `ztb/utils/training_utils.py` の `load_model` 重複解消 (L414 削除)
2. デッドファイル 8件を `archived/` へ移動 (~2,600行回収)

### P1 (次チケット)
3. `sac_train.py` (v460) のヘルパーを共有ファクトリに統合
4. `UnifiedTrainer` の Mixin 分割開始 (federated, v433_adaptive 優先)
5. `sac_utils.py` 2ファイル統合

### P2 (中期)
6. `SACTrainer` (algorithms/) の800行関数分割
7. 環境ファクトリ `env_factory.py` 作成
8. config loader 一本化

### P3 (長期)
9. `UnifiedTrainer` 完全 Mixin 分割
10. 命名規約統一
11. `reward_function_optimizer` Strategy パターン適用
