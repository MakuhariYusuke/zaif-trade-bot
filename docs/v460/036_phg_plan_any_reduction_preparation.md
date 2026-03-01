# 036# Any削減マスター（計画・進捗・方針 一元化）

| key | value |
|---|---|
| 番号 | 036 |
| フェーズ | phg (cross-gate) |
| 種別 | master |
| 作成日 | 2026-02-14 |
| 目的 | `Any` 削減の方針・進捗・次アクションを1文書に統合する |
| スコープ | `ztb/`, `scripts/v460/` |

---

## 1. 結論（要点）

1. 先に可視化と受け皿を整え、後から削減を進める方針は有効。  
2. `scripts/v460` は `Any` type-debt を 0 まで削減済み。  
3. `ztb/optimization/model_compression.py` / `ztb/analysis/core/analyzer.py` / `ztb/utils/checkpoint.py` まで `Any=0` 化を拡張。  
4. `ztb/training/callbacks/supervised/supervised_callbacks.py` と `ztb/training/callbacks/reinforcement/sac/sac_callbacks.py` を継承ベースへ再編し `Any=0` 化。  
5. 次の主戦場は `ztb/trading`, `ztb/utils`, `ztb/analysis`。  

---

## 2. ベースライン（Step0）

実行コマンド:

```bash
python scripts/quality/any_inventory.py --top 25 --json-out results/type_any_inventory_v460_prep.json
```

初期計測:

| 指標 | 値 |
|---|---|
| scanned_files | 1,280 |
| any_total_tokens | 5,185 |
| any_import_tokens | 664 |
| any_annotation_tokens | 4,371 |
| any_alias_tokens | 65 |
| any_type_debt_tokens | 4,436 |
| any_runtime_tokens | 85 |

`type_debt` 上位ディレクトリ:

1. `ztb/trading` (1204)
2. `ztb/training` (1100)
3. `ztb/analysis` (577)
4. `ztb/utils` (451)
5. `scripts/v460` (39)

---

## 3. 既存実装の再利用原則

1. `ztb/types/common.py`
   - `ConfigValue`, `ConfigDict`, `is_config_dict`
   - 今回追加: `ConfigSection`, `JSONDict`, `MetricsDict` など移行用別名
2. `ztb/trading/types.py`
   - `MarketState`, `PositionSignal`
3. `ztb/adaptation/monitoring/types.py`
   - `RiskMetrics`, `TradingPerformanceMetrics`

原則:
- 新規に `dict[str, Any]` を追加しない。
- 先に既存型で受けられるか確認し、足りない場合のみ共通層へ追記する。

---

## 4. 実施ログ（統合）

### Step1: 下準備

1. `scripts/quality/any_inventory.py` を追加。  
2. `ztb/types/common.py` に移行用共通別名を追加。  
3. `scripts/v460/lib/config_loader.py` を `ConfigSection` + `is_config_dict` 化。  

### Step2: `scripts/v460/lib` 前半

1. `scripts/v460/lib/manifest.py` の `Any` 撤去。  
2. `scripts/v460/lib/evaluator.py` を Protocol (`FitPredictModel`) 化。  
3. `scripts/v460/lib/tasks/feature_info.py` の `Any` 撤去 + `np` 未 import 修正。  

### Step3: `scripts/v460/lib` 後半

1. `scripts/v460/lib/tasks/sac_train.py` の `Any` 撤去（局所 Protocol 化）。  
2. `scripts/v460/lib/data_loader.py` の `Any` 撤去（`NaNRatioCheck` 追加）。  

### Step4: `scripts/v460` 全域

1. `scripts/v460/monitor_fill_test.py` の `Any` 撤去（`TypedDict` 化）。  

### Step5: `ztb/evaluation/unified_evaluation.py`

1. `Any` を 0 まで削減。  
2. `TimeSeriesWindow = Any` のプレースホルダ依存を除去し、遅延 import で実クラス解決。  
3. `from ztb.analysis.common.types import EvaluationResult` の重複 import を解消（ローカル dataclass と衝突していた）。  

### Step6: 重複ロジック共通化（v460）

1. `scripts/v460/lib/config_access.py` を追加。  
2. `feature_info.py` / `sac_train.py` の section取得・数値変換を共通化。  

### Step7: 既存実装への置換（重複除去）

1. `scripts/v460/run_experiment.py` の Gate閾値読込を  
   `load_gate_thresholds()` へ統一（手書き YAML 読込を削除）。  

### Step8: `ztb/metrics/metrics.py` の `Any` 全撤去

1. `NDArray[Any]` を `NDArray[np.generic]` へ置換。  
2. `Dict[str, Any]` を `dict[str, object]` へ置換。  
3. 対象ファイル単体の `any_type_debt_tokens` を **0** 化。  

### Step9: `ztb/trading/comprehensive_backtest.py` の `Any` 全撤去

1. `Dict[str, Any]` / `List[Any]` / `Optional[Any]` を `object` 系へ置換。  
2. 既存ロジックを変えずに型注釈のみ整理。  
3. 対象ファイル単体の `any_type_debt_tokens` を **0** 化。  

### Step10: `ztb/trading/real_data_validation.py` の `Any` 全撤去

1. `Dict[str, Any]` / `Optional[Any]` を `dict[str, object]` / `Optional[object]` へ置換。  
2. テスト用フォールバッククラスの `Any` も `object` 化。  
3. 対象ファイル単体の `any_type_debt_tokens` を **0** 化。  

### Step11: 型定義整理 + 大型ファイル削減

1. `ztb/types/common.py` に `ObjectMap` / `ObjectList` / `ObjectRecords` を追加。  
2. `ztb/analysis/common/types.py` を共通 alias (`ConfigSection`, `MetricsDict`, `ObjectMap`) へ寄せ、`Any` を全撤去。  
3. `ztb/training/unified_optimizer.py` の `Dict[str, Any]` 系注釈を `object` ベースへ置換し、`Any` を全撤去。  
4. `ztb/trading/comprehensive_backtest.py` で `RiskMetrics` 未 import を修正（例外握り潰し依存を解消）。  

### Step12: 共通型定義の追加整理

1. `ztb/types/common.py` の `Dict[str, Any]` 系を `object` ベースへ置換。  
2. `Logger` / `spaces` / `DataLoader` の fallback を `object` 化し、`Any` import を削除。  
3. `ztb/types/common.py` 単体の `any_type_debt_tokens` を **0** 化。  

### Step13: `ztb/training/unified_trainer/reporting.py` 整理

1. `Dict[str, Any]` / `List[Dict[str, Any]]` を `ObjectMap` / `ObjectRecords` へ統一し、`Any` を全撤去。  
2. `print_summary()` にあった表示ロジックが `save_ensemble_report()` 配下で死んでいた問題を修正。  
3. `print_summary()` を `ObjectMap` 互換で安全に参照する実装へ変更。  
4. `log_training_start()` の `config["training"]["total_timesteps"]` 直参照を安全参照へ変更（KeyError予防）。  

### Step14: 追加ホットスポット削減 + 既存不具合修正

1. `ztb/optimization/model_compression.py` を再構成し、重複定義と API 不整合を解消。  
2. `KnowledgeDistillationCompressor` の二重定義を解消し、`__all__` に公開 API を整列。  
3. `compress_model(**kwargs)` で `ModelCompressor(**kwargs)` が失敗しうる問題を修正（`__init__` 追加）。  
4. `ztb/analysis/core/analyzer.py` と `ztb/utils/checkpoint.py` を `ObjectMap`/`object` 系へ寄せ、`Any` を全撤去。  

### Step15: コールバック基底の整合性回復

1. `ztb/training/callbacks/shared/base/learning_callback.py` を `ObjectMap` ベースへ統一し、`Any` を全撤去。  
2. `LearningCallback` の抽象メソッド定義崩れ（`@abstractmethod` 重複、`on_training_start/on_epoch_start` 欠落）を修正。  
3. 関連確認中に検出した `ztb/training/callbacks/unsupervised/unsupervised_callbacks.py` の既存構文エラーを修正。  

### Step16: supervised callback 群の継承整理 + `Any` 全撤去

1. `ztb/training/callbacks/supervised/supervised_callbacks.py` を再構成し、`_MonitoredCallback` / `_BaseSupervisedMetricsCallback` を導入。  
2. `EarlyStoppingCallback` と `ModelCheckpointCallback` の監視ロジック重複を継承側へ集約。  
3. `ClassificationMetricsCallback` / `RegressionMetricsCallback` の抽出処理を基底へ集約し、`Any` を全撤去。  
4. 履歴配列を上限付き append に統一し、長時間実行時のメモリ肥大を抑制。  

### Step17: SAC callback 群の継承整理 + 既存不具合解消

1. `ztb/training/callbacks/reinforcement/sac/sac_callbacks.py` を再編し、重複 `pass` 実装とイベント処理分岐を整理。  
2. `on_epoch_end` 内の未定義変数参照（`q_loss`, `action_entropy`, `action_std`, `reward` など）を修正。  
3. `Any` を全撤去し、履歴配列を上限付きに統一。  
4. 既存クラス構成は維持したまま `create_sac_target_updater()` を追加し、生成 API を補完。  

### Step18: callback no-op 基底の共通化

1. `ztb/training/callbacks/shared/base/learning_callback.py` に `NoOpMemoryOptimizedCallback` を追加。  
2. `supervised_callbacks.py` / `sac_callbacks.py` のローカル no-op 基底を削除し、共通基底へ置換。  
3. 継承重複を削減し、以後の callback 置換を既存実装ベースで進められる状態を整備。  

### Step19: 次フェーズ4ファイルの `Any` 全撤去 + 互換修正

1. `ztb/analysis/v4xx_unified_analyzer.py` を `ObjectMap` / `ObjectRecords` ベースへ再構成し、`Any` を全撤去。  
2. `analyze_multi_period_backtest()` の `list`/`dict` 入力を後方互換で吸収し、既存テスト想定とズレる不整合を修正。  
3. `ztb/utils/type_guards.py` を `ConfigSection` / `object` / `NDArray[np.generic]` へ統一し、`Any` を全撤去。  
4. `ztb/analysis/regime/v444_regime_analyzer.py` の regime payload を `ConfigSection` / `ObjectMap` へ寄せ、`Any` を全撤去。  
5. `ztb/trading/real_data_validator.py` を互換 API 維持で整理し、`Any` を全撤去。合わせて `AnomalyDetector` 継承不整合・未定義 `np` 参照・到達不能コードを修正。  

### Step20: interface 共通化 + 設定ローダ整理（重複削減/継承整理）

1. `ztb/trading/strategies/action_signal_guide/interfaces/common_types.py` を追加し、`PayloadMap`/`MetricsMap` など共通 alias と `IActionSignalGuideInterface`（marker 基底）を導入。  
2. `adaptation_interfaces.py` / `ml_interfaces.py` / `portfolio_interfaces.py` を共通 alias + 基底継承に統一し、重複注釈を削減。  
3. 上記3 interface ファイルの `Any` を全撤去。  
4. `ztb/config/loaders/priority_loader.py` を `ObjectMap` ベースへ移行し、`Any` を全撤去。  
5. `load_env` / `load_cli` / `load_yaml` の例外吸収パターンを `_safe_load()` へ共通化し、重複ロジックを削減。  

### Step21: `job_manager` 再構成（`Any` 全撤去 + 状態整合性修正）

1. `ztb/experiments/job_manager.py` を `JobConfig` / `JobResult` / `JobStateRecord` (`TypedDict`) ベースへ再構成し、`Any` を全撤去。  
2. `execute_job()` で manifest に `running` を明示反映し、`get_job_status()` で実行中検知できない不整合を修正。  
3. `run_all_jobs()` で `_can_skip_job()` を実運用へ接続し、既完了ジョブの再実行を回避。  
4. 非同期失敗時に result/manifest/DB を同時更新する `_register_async_failure()` を追加し、状態散逸を抑制。  
5. `JobStateDB.save_job_state()` を UPSERT 化して既存開始時刻や指標を保全し、`INSERT OR REPLACE` 起因の履歴欠落リスクを低減。  
6. 並列実行セットアップ失敗時に sequential fallback する経路を追加し、実行不能で全停止するリスクを低減。  

### Step22: `performance_optimizer` の継承導入 + `Any` 全撤去 + 不具合修正

1. `ztb/trading/performance_optimizer.py` を `OptimizationComponentBase` 継承構成へ再編し、`LatencyOptimizer` / `MemoryOptimizer` / `CPUOptimizer` の最適化ステップ収集ロジックを `_collect_applied_steps()` へ共通化。  
2. `safe_execute(..., error_msg=..., default_return=...)` というシグネチャ不一致（実運用で `TypeError` 化しうる）を解消し、`_safe_optimize()` 経由で正しい呼び出しへ統一。  
3. `LatencyOptimizer._analyze_bottlenecks()` の cProfile 出力解析を修正し、ヘッダ行混入や先頭空白行での解析漏れを回避。  
4. `MemoryOptimizer` に停止可能な監視制御 (`_monitoring_stop_event`, `stop_memory_monitoring`) を追加し、停止不能な監視スレッド増殖リスクを低減。  
5. `PerformanceOptimizationSystem.stop_performance_monitoring()` から `stop_memory_monitoring()` を連動呼び出しし、監視停止時の後始末を統一。  
6. `get_performance_report()` の空配列 `np.mean` 依存を解消し、`_evaluate_target_achievement()` の 0除算リスクをガード。  
7. 当該ファイルの `Any` を全撤去し、`any_type_debt_tokens=0` を達成。  

### Step23: `end_to_end_validator` の継承導入 + `Any` 全撤去 + 実行不整合修正

1. `ztb/trading/end_to_end_validator.py` を `StageValidatorBase` 継承構成へ再編し、`Component/Integration/Performance/SystemHealth` の4 validator に共通集計（tests_run/pass/fail/duration）を集約。  
2. `StageExecutionResult` / `ValidationStatusPayload` / `ValidationHistoryPayload` / `PipelineLike` (`Protocol`) を導入し、stage payload と pipeline 参照を型固定。  
3. `ValidationMetrics({})` の誤用（`tests_run` に `dict` が入る潜在不整合）を解消し、`ValidationMetrics(...)` を明示初期化へ統一。  
4. `SystemHealthValidator.validate_system_health` の二重定義を解消し、health_checks が実行されず常時 `PASSED` になりうる不具合リスクを修正。  
5. `run_end_to_end_validation()` で `validate_components([])` / `validate_integrations([])` を呼んで毎回テスト集合を空にしていた不整合を修正。  
6. pipeline 履歴登録を `pipeline_id` で重複排除し、`None` pipeline の履歴汚染を抑制。  
7. 当該ファイルの `Any` を全撤去し、`any_type_debt_tokens=0` を達成。  

### Step24: `unified_trainer/trainer` の重点改善（不具合修正 + メモリ/実行効率 + `Any` 全撤去）

1. `ztb/training/unified_trainer/trainer.py` の `Any` 注釈を `object` ベースへ置換し、当該ファイルの `any_type_debt_tokens=0` を達成。  
2. `_execute_training()` の `OperationMemoryTracker` 呼び出しを `_safe_memory_tracking()` context 化し、例外時に `__exit__` が漏れるリスクを解消。  
3. `_execute_training()` に `finally: _stop_memory_monitoring()` を追加し、失敗経路でも監視スレッドが残留しないよう修正。  
4. メモリ監視の閾値判定順 (`>95` と `>90`) を修正し、critical 判定が到達不能になる不具合を解消。  
5. `_start_memory_monitoring()` を「生存スレッドのみ再利用」に変更し、停止済み thread 参照で再起動できない不整合を修正。  
6. `_monitor_training_memory()` に `total_steps<=0` ガードを追加し、0除算リスクを解消。  
7. `_validate_feature_consistency()` で header 列情報を `feature_cache` にキャッシュし、同一ファイルの重複I/Oを削減。  
8. 同メソッドで `data_path` fallback (`training.data_config.data_path` -> `config.data_path`) と `max_features` の `int` 正規化を追加し、設定揺れによる誤判定を低減。  
9. `_cleanup_memory()` で `feature_cache.clear()` を併用し、トレーニング終了後のキャッシュ残留によるメモリ保持を抑制。  

### Step25: 水平展開（`sac_trainer` / `system_optimizer`）+ 不具合修正

1. `ztb/training/unified_trainer/algorithms/sac_trainer.py` の `Any` 注釈を `object` ベースへ置換し、当該ファイルの `any_type_debt_tokens=0` を達成。  
2. `validate_training(model_path=...)` が引数を無視する不具合を修正し、明示指定パスを優先するように変更。  
3. `SACTrainer(test_config, self.logger)` / `SACTrainer(config_override, self.logger)` の位置引数誤渡しを `logger=` 指定へ修正。  
4. checkpoint callback の重複登録を抑止し、同一 save path の二重追加を回避。  
5. `training_time` に下限 (`1e-9`) を入れ、SPS算出時の 0除算リスクを低減。  
6. `ztb/training/system_optimizer.py` の `Any` 注釈を `Callable[..., object]` / `object` ベースへ置換し、当該ファイルの `any_type_debt_tokens=0` を達成。  
7. `optimize_training_step()` の計測を `time.perf_counter()` 化し、短時間区間の測定精度を改善。  
8. memory tracker の enter/exit を例外安全化し、初期化失敗時でも学習継続できるよう修正。  
9. performance history の二重 append を解消し、統計の過大計上を防止。  
10. `cache_io_operation()` の cache hit 時二重 `get` を削減し、ホットパスの冗長処理を解消。  

### Step26: `ensemble_system` の型固定 + 実行安定化（不具合修正/メモリ管理）

1. `ztb/training/unified_trainer/ensemble_system.py` の `Any` を全撤去し、`ConsensusRequirementConfig` / `StabilityVotingConfig` / `AdaptationConfig` / `PredictionInfo` (`TypedDict`) で payload 契約を明示。  
2. `members` 設定値が初期化時に実質無視される問題を修正し、`config.members` 件数を基準に member 生成（specialization は循環割当）するよう変更。  
3. 不正な `specialization` 文字列で初期化が失敗するリスクを修正し、未知値は `sideways` へフォールバック。  
4. `weighted_confidence` の `total_weight=0` 時に `normalized_weights` で 0除算となる不具合を修正し、多数決 fallback を導入。  
5. 不正な `voting_mechanism` 文字列で `VotingMechanism(...)` が `ValueError` を投げる経路を修正し、多数決 fallback を導入。  
6. `decision_log` / `performance_history` の append を共通化し、上限超過時に head を削除する bounded 管理へ統一。  
7. `update_member_performance()` で performance 履歴を記録するようにし、`performance_history_size` が常時 0 になりやすい観測性不足を改善。  
8. `save_ensemble_state()` の config 直列化を `asdict()` 化し、dataclass 状態の保存整合性を改善。  
9. `load_ensemble_state()` を堅牢化し、破損/不正形式 JSON・不正 member payload・未知 specialization を安全に吸収しつつ復元できるよう修正。  
10. 当該ファイルの `any_type_debt_tokens=0` を達成。  

### Step27: `analysis/promotion` の型固定 + 評価安全性強化

1. `ztb/analysis/promotion.py` の `Any` を全撤去し、`ObjectMap` / `ObjectRecords` ベースへ統一（当該ファイル `any_type_debt_tokens=0`）。  
2. 評価値の数値化ヘルパ（`_coerce_metric`）を導入し、文字列や不正値が混入した場合でも比較演算で `TypeError` を起こさず安全に fail 判定するよう修正。  
3. 比較演算を `_compare_values` に集約し、criterion 実装間の重複ロジックを削減。  
4. 失敗時スコア計算を `_bounded_ratio` に共通化し、負値・0除算・過大値の混入で異常スコアになるリスクを低減。  
5. YAML 設定読込を厳格化し、root が mapping 以外の場合は明示的に `ValueError` を返すよう修正。  
6. `_compile_criteria()` の cache key 生成を `json.dumps(..., default=str)` 化し、非JSON値混入時のシリアライズ失敗を回避。  
7. カテゴリ/通知/webhook 設定の dict 参照を `_as_object_map` 経由に統一し、設定型揺れ時の `.get` 連鎖例外を予防。  
8. webhook retry 設定に下限ガード（`max_attempts >= 1`, `backoff_seconds >= 0`）を追加し、無効設定で通知処理がスキップ/暴走するリスクを低減。  

### Step28: `analysis/promotion` の保守性改善（責務分割/重複削減）

1. `evaluate_promotion()` を小さな責務関数へ分解（`_evaluate_criterion_group`, `_should_promote`, `_resolve_status_result`, `_build_evaluation_details`, `_notify_promotion_result`）。  
2. criterion詳細 payload 生成を `_build_criterion_detail()` へ共通化し、hard requirement と regular criterion の重複ロジックを削減。  
3. `RatioCriterion` を `NumericCriterion` 継承へ寄せ、同一比較ロジックの重複実装を解消。  
4. engine 側の criterion 生成分岐を `CriterionPluginManager` に一本化し、型追加時の変更点を単一箇所へ集約。  
5. 未使用のしきい値最適化メソッドを除去し、読み手が追う分岐数を削減。  
6. `Any` 負債は維持（当該ファイル `any_type_debt_tokens=0` のまま）。  

### Step29: `heavy_env/core` の `Any` 全撤去 + 継承重複削減 + 転用可否調査

1. `ztb/trading/environment/heavy_env/core.py` の `Any` 注釈を全撤去し、当該ファイル `any_type_debt_tokens=0` を達成。  
2. `deep_merge_dict` を `ObjectMap` ベースへ型固定し、再帰マージ時の key/value 取り回しを `object` 系へ統一。  
3. `reset/step/_get_observation/_get_info/get_legal_actions/action_mask/get_action_masks` の返却型を具体化し、環境I/O契約を明確化。  
4. `enable_market_regime_adaptation()` の `adaptation_config` と内部 `regime_stats` を `object` ベースへ統一し、設定 payload の型揺れを抑制。  
5. `FlipHeavyTradingEnv.enable_market_regime_adaptation()` の重複実装を `super()` 委譲へ整理し、派生クラス側はデバッグ責務のみに縮小。  
6. `HeavyTradingEnv._build_initial_regime_stats()` を導入し、`Heavy/Flip` 間での regime stats 初期化重複を削減。  
7. **転用可否調査**: Step28 で分割した `promotion` の評価メソッド群は、現状コードベースでは `ztb/analysis/promotion.py` 以外に同等の実装先がなく、即時横展開先は限定的（`ztb/evaluation/promotion.py` は fallback wrapper 中心）。将来 2nd promotion engine 追加時に mixin 化での再利用が妥当。  

### Step30: `utils/env_metrics` の `Any` 全撤去 + 抽出仕様の共通化

1. `ztb/utils/env_metrics.py` の `Any` を全撤去し、`object` / `ObjectMap` ベースへ統一（当該ファイル `any_type_debt_tokens=0`）。  
2. `resolve_env/unwrap_env/extract_*` 系 API の入出力型を具体化し、呼び出し側契約を明確化。  
3. メトリクス抽出を `_BASE_METRIC_SPECS` / `_OPTIONAL_METRIC_SPECS` + `_populate_metric_specs()` へ集約し、重複 `_set_first_attr` 呼び出しを削減。  
4. `_set_first_attr()` の `hasattr + getattr` 二重参照を単一 `getattr` に整理し、例外安全性を維持したまま冗長処理を削減。  
5. optional 抽出で重複していた `gross_pnl/net_pnl/fees/slippage` の再設定を削除し、同一キーの上書き経路を整理。  
6. `trainer.py` 経由の既存利用互換を維持したまま、型注釈負債を純減。  

### Step31: `candlestick_patterns` の継承導入 + `Any` 全撤去 + 判定不整合修正

1. `ztb/trading/strategies/action_signal_guide/pattern_recognition/candlestick_patterns.py` に `_CandlestickPatternBase` を導入し、共通処理（入力/トレンド検証、pattern factor 構築、MTF補正、`SignalResult` 生成）を基底化。  
2. `_ThreeCandleStarBase` / `_LongShadowReversalBase` / `_ThreeConsecutiveReversalBase` / `_EngulfingPatternBase` を追加し、Morning/Evening・Hammer/Hanging・Three Crows/Soldiers・Bull/Bear Engulfing の重複ロジックを継承側へ集約。  
3. `CandleCharacteristics` (`TypedDict`) を導入し、中間解析 payload のキー契約を明示。`Any` 注釈を全撤去し、当該ファイル `any_type_debt_tokens=0` を達成。  
4. `ThreeBlackCrows` / `ThreeWhiteSoldiers` の candle 走査順を時系列順（oldest -> newest）へ修正し、進行方向判定が逆順解釈になる不具合可能性を解消。  
5. 連続足判定に 0除算ガード（`prev_close=0`, `high==low`）を追加し、異常OHLC入力で例外に依存して `None` へ落ちる経路を明示ガードへ置換。  
6. 不使用の module-level 定数群と未使用 `base_strength` 算出を削除し、保守対象の責務を縮小（ファイル行数: 1,393 -> 1,007）。  

### Step32: 水平展開検証（`wave_counting`）+ 継承導入 + `Any` 全撤去

1. `ztb/trading/strategies/action_signal_guide/pattern_recognition/wave_counting.py` に `_WavePatternBase` を導入し、index解決・pivot抽出・confidence算出を共通化。  
2. `Impulse/Corrective/WaveExtension/WaveI/WaveV/WaveY/WaveP/WaveN/WaveS` の 9 recognizer を共通基底へ移行し、初期化・前処理重複を削減。  
3. `WaveStructure` (`TypedDict`) を追加し、`identify_wave_structure()` の返却契約を明示。  
4. 既存不具合を修正:  
   - `index=-1`（デフォルト）時に `if index < lookback_period` で早期 `None` になる経路を、共通 index 正規化で解消。  
   - lookback 切り出し後のローカル pivot 位置をグローバル index と比較していた不整合（completion 判定が成立しにくい）を、共通 global pivot 変換で解消。  
   - `wave_extension` metadata の `extension_ratio` で 0除算しうる経路を `EPSILON` ガードで解消。  
5. 当該ファイルの `Any` を全撤去し、`any_type_debt_tokens=0` を達成。  

### Step33: `streaming_processor` の `Any` 全撤去 + 実行時不具合修正

1. `ztb/trading/strategies/action_signal_guide/realtime_adaptation/streaming_processor.py` の `Any` を全撤去し、`ObjectMap` / `ObjectRecords` ベースへ統一（当該ファイル `any_type_debt_tokens=0`）。  
2. `BaseStreamingProcessor` に `IStreamingProcessor` の不足実装（`process_streaming_data` / `register_data_handler` / `get_processed_data`）を追加し、抽象メソッド未実装での具象生成不能リスクを解消。  
3. `AdvancedStreamingProcessor.get_processed_data()` が `super().get_processed_data()` を呼ぶ一方で基底未実装だった不整合を解消。  
4. `data_handlers` / `processed_data_cache` を基底で初期化し、高度機能経路で `AttributeError` になりうる既存不具合を修正。  
5. `config.max_cache_size` 参照（未定義属性）を廃止し、基底で `max_cache_size` を算出して一元管理。  
6. 保存キーを `processing_timestamp` ベースへ変更し、`timestamp` 不在で cache key が衝突しうる問題を解消。  
7. `feature_buffer` を `list.pop(0)` から `deque(maxlen=1000)` へ変更し、ホットパスでの O(n) 削除を回避。  
8. `_enhance_processed_data()` の `enhanced.get(..., 0).rolling(...)` による型不整合（`int` に rolling が無く例外）を修正し、数値列チェック付きの安全な Series 処理へ置換。  
9. 数値変換ヘルパ（`_as_float`）と payload 正規化（`_as_object_map`）を導入し、予期しない入力型での比較/演算例外を低減。  

### Step34: `fibonacci_patterns` の継承導入 + `Any` 全撤去 + 既存不整合修正

1. `ztb/trading/strategies/action_signal_guide/pattern_recognition/fibonacci_patterns.py` に `_FibonacciPatternBase` を導入し、index正規化・confidence算出・閾値判定・共通設定処理を集約。  
2. `FibonacciRetracement/Extension/Projection` の3 recognizer を基底継承へ移行し、初期化・factor組立・confidence cap 処理の重複を削減。  
3. `FibonacciRetracementMatch` (`TypedDict`) / `FibonacciLevelConfig` (`dataclass`) を導入し、返却 payload とレベル設定契約を明示。  
4. `index=-1`（デフォルト）で `index < max_swing_length` により即 `None` となる経路を、`validate_recognition_inputs` ベースの index 正規化へ置換。  
5. retracement cache key をデータ文脈込みへ変更し、異なる入力DataFrame間で `start_idx/end_idx` だけが一致した際の誤ヒットを抑制。  
6. retracement cache を上限付き（`_max_cache_size=2048`）へ変更し、長時間稼働時の無制限メモリ増加を抑制。  
7. extension/projection の tolerance 0 ケースで発生しうる 0除算をガード。  
8. 互換補完として `find_support_resistance_levels()` と `thresholds` を実装し、既存テスト/呼び出し側の期待属性欠落リスクを解消。  
9. 当該ファイルの `Any` を全撤去し、`any_type_debt_tokens=0` を達成。  

### Step35: `harmonic_patterns` の継承導入 + `Any` 全撤去 + 保守性/安定性改善

1. `ztb/trading/strategies/action_signal_guide/pattern_recognition/harmonic_patterns.py` に `_HarmonicPatternBase` を導入し、4 recognizer（`Gartley/Butterfly/Bat/Crab`）の重複していた探索・confidence算出・MTF補正・metadata生成を共通化。  
2. `HarmonicPatternMatch` (`TypedDict`) を導入し、Analyzer 返却payload（`completion_position` を含む）を明示化。  
3. `index=-1` 既定値で即 `None` へ落ちる経路を `validate_recognition_inputs` ベースの index 正規化へ置換し、デフォルト呼び出し時の取りこぼしを解消。  
4. `completion_index`（ラベル）と `index`（整数位置）の直接減算を廃止し、`completion_position` で比較するよう修正。DatetimeIndex 環境での型不整合例外リスクを解消。  
5. Gartley の到達不能な synthetic signal ブロックを撤去し、制御フローの可読性と保守性を改善。  
6. pivot 生成の重複コードを撤去し、synthetic pivot fallback は Gartley のみに限定（他パターンへの誤適用での偽陽性を抑制）。  
7. 認識結果 cache を4 recognizer 共通で上限付き運用へ統一し、長時間稼働時のメモリ増加を抑制。  
8. 当該ファイルの `Any` を全撤去し、`any_type_debt_tokens=0` を達成。  

### Step36: `pattern_optimizer` の `Any` 全撤去 + 学習/推論の安定化

1. `ztb/trading/strategies/action_signal_guide/ml_integration/pattern_optimizer.py` の `Any` 注釈を全撤去し、`ModelTrainingSuccess/Error` (`TypedDict`) と `RegressorModel` union を導入。  
2. `BasePatternOptimizer` に `optimize_pattern_combination` / `get_optimization_metrics` を実装し、`IPatternOptimizer` 抽象契約未充足での具象生成不能リスクを解消。  
3. 学習時に `self._feature_names` を保持し、推論時 `_prepare_features()` で列順序・欠損列補完（0.0）を統一。学習時と推論時の特徴量並び不整合による誤予測リスクを低減。  
4. `_prepare_data()` に Feature/Target 長さ整合チェックと最小サンプル検証を追加し、壊れた入力での暗黙失敗を早期検知。  
5. `TimeSeriesSplit(n_splits=5)` 固定を見直し、サンプル数に応じた動的 split へ変更。少量データでの CV 例外頻発を回避。  
6. 全モデル学習失敗時に `success=True` で返り得る経路を修正し、成功モデル 0 件なら `MLResult(success=False)` を返すように変更。  
7. `_ensemble_predictions()` を scalar 正規化ベースへ整理し、配列形状差異時の float 変換不整合リスクを低減。  
8. `AdvancedPatternOptimizer.performance_history` を上限付きで管理し、長時間稼働時の履歴無制限増加を抑制。  
9. 当該ファイルの `Any` を全撤去し、`any_type_debt_tokens=0` を達成。  

### Step37: `pattern_recognition/base` の `Any` 全撤去 + キャッシュ不整合修正

1. `ztb/trading/strategies/action_signal_guide/pattern_recognition/base.py` の `Any` 注釈を全撤去し、`LRUCache[T]`（Generic）と `MultiCandleCharacteristics`（`TypedDict`）を導入。  
2. `timed` decorator を `Callable[..., object]` ベースへ置換し、ラッパー型を `TypeVar` で保持。  
3. fallback `MultiTimeframeData` を `Dict[str, Dict[str, object]]` へ更新し、alias 側 `Any` 負債を解消。  
4. `recognize_with_cache()` を修正し、`index=-1` のまま cache key が衝突しやすい経路を `resolved_index` 正規化へ変更。  
5. `_signal_cache` の保持値を `(SignalResult, signal_index)` へ変更し、従来 `is_expired(index, index)` で実質無効だった期限判定を有効化。  
6. `_validate_multi_timeframe_data()` に key/payload 型チェックを追加し、誤構造入力の早期検知を強化。  
7. `SignalResult.timestamp` / `PatternRecognizer` / `CandlestickPatternRecognizer` / `MultiCandlePatternRecognizer` の設定・返却注釈を `object`/具体型へ整理し、横断的な保守性を改善。  
8. 当該ファイルの `Any` を全撤去し、`any_type_debt_tokens=0` を達成。  

### Step38: `signal_quality_filter` + `action_signal_guide` の重複削減/互換修正 + `Any` 全撤去

1. `ztb/trading/strategies/action_signal_guide/components/signal_quality_filter.py` の `Any` 注釈を全撤去し、`PatternQualityRecord` (`TypedDict`) と `Mapping[str, object]` ベースへ移行（当該ファイル `any_type_debt_tokens=0`）。  
2. `signal_quality_filter` 内で重複していた composite score 算出ロジックを `_calculate_composite_score()` に集約し、`_rank_by_quality_score` と `_update_quality_statistics` の重複を削減。  
3. 互換メソッド `filter_by_quality()` / `update_thresholds()` を追加し、`SignalGenerator` 側の旧API呼び出し不整合（メソッド未定義）を解消。  
4. `SignalQualityEvaluator` を追加し、`SignalGenerator.initialize_adaptive_components()` で参照されるが未実装だった依存欠落を解消。  
5. `SignalGenerator.initialize_adaptive_components()` を runtime import + config注入へ修正し、`SignalQualityFilter()` 無引数生成と `SignalQualityEvaluator` 未解決の NameError リスクを解消。  
6. `SignalGenerator.__init__` で `signal_quality_filter` / `signal_quality_evaluator` を明示初期化し、属性未初期化参照リスクを低減。  
7. `ztb/trading/strategies/action_signal_guide/action_signal_guide.py` の `Any` 注釈を全撤去し、`GuidanceInput` / `object` / 具体型へ統一（当該ファイル `any_type_debt_tokens=0`）。  
8. `update_guidance_mode(None)` の安全早期 return を追加し、互換入力での不要警告・不定挙動を防止。  
9. repo 全体 `any_type_debt_tokens` を `3,061 -> 3,024` へ削減。  

### Step39: `signal_generator` の重複定義解消 + 不具合修正 + `Any` 全撤去

1. `ztb/trading/strategies/action_signal_guide/components/signal_generator.py` の `Any` 注釈を全撤去し、当該ファイル `any_type_debt_tokens=0` を達成。  
2. 同名メソッドの二重定義（`generate_signal` / `apply_adaptive_filtering` / `_filter_by_guidance_level`）を解消し、後半実装による前半実装の上書きバグを解消。  
3. `generate_signal()` の早期 return 経路で未定義 `ActionSignal` を参照する不具合を修正（先頭で class 解決）。  
4. `__init__` で `performance_tracker` を `None` で再上書きしていた不具合を修正し、注入依存を保持。  
5. `_initialize_adaptive_weights()` の `self.recognizer_groups` 未定義参照を解消し、稼働 recognizer 群から重みを初期化する方式へ変更。  
6. `RegimeAdaptiveSignalProcessor` の未実装API呼び出し（`adapt_for_regime` / `filter_signals_for_regime`）を廃止し、実装済み `process_signals_for_regime()` ベースへ統一。  
7. recognizer 実行ロジックを `_process_recognizer` / `_build_action_signal` / `_record_generated_signal` に集約し、parallel/sequential 経路の重複を削減。  
8. MTF入力の互換処理 `_extract_timeframe_alignment()` を追加し、`timeframe_alignment` 直接指定と nested payload の両方を受理。  
9. repo 全体 `any_type_debt_tokens` を `3,024 -> 3,017` へ削減。  

### Step40: `market_regime` の互換崩れ修正 + 不具合修正 + `Any` 全撤去

1. `ztb/trading/strategies/action_signal_guide/components/market_regime.py` の `Any` 注釈を全撤去し、当該ファイル `any_type_debt_tokens=0` を達成。  
2. `MarketRegimeDetector.detect_regime()` のデフォルト返却を `MODERATE_VOLATILITY_RANGING` に統一し、旧 `RANGING` 依存での実行時 AttributeError リスクを解消。  
3. `detect_regime_from_data()` の戻り値を `str` へ是正し、`detect_regime(...).value` を返す互換経路へ修正。  
4. `RegimeAdaptiveSignalProcessor` で `signal_type` / `pattern_type` の不一致を吸収し、パターン適合判定が常時ミスマッチになる不具合を修正。  
5. `confidence_adjustment` 算出時のゼロ除算リスクを解消し、regime分析情報を `metadata["regime_analysis"]` に格納して dataclass 互換性を改善。  
6. canonical regime enum（`MODERATE_BULL_TREND` 等）へ寄せて設定表を整理し、旧名依存を低減。  
7. `ztb/analysis/regime/market_regime_types.py` に後方互換 alias（`RANGING` / `TRENDING_BULLISH` / `TRENDING_BEARISH` / `HIGH_VOLATILITY` / `LOW_VOLATILITY` / `SIDEWAYS`）を追加し、他モジュールへの水平互換を確保。  
8. 市場データ欠損・ゼロ価格時のガードを追加し、`_analyze_momentum` / `_analyze_support_resistance` 等の例外・不正値リスクを低減。  
9. repo 全体 `any_type_debt_tokens` を `3,017 -> 3,012` へ削減。  

### Step41: レジーム運用の実効性修正（判定/連携の不整合解消）+ `Any` 追加削減

1. `ztb/trading/strategies/action_signal_guide/components/market_regime.py` に legacy互換の価格ベース検出経路を追加し、`(current_price, step)` 呼び出しでも破綻しないよう修正。  
2. `MarketRegimeDetector` に `price_history` を持たせ、環境側 reward shaper の検出器契約（価格履歴参照）との互換性を改善。  
3. `_calculate_volatility()` の単位不整合（std*10 スケーリング）を解消し、`volatility_threshold=0.03` と整合する raw std ベースへ修正。  
4. relative regime 判定の percentile 計算を修正し、`reference_window` 未満データ時でも利用可能な履歴範囲で評価するよう改善。  
5. `RegimeAdaptiveSignalProcessor` で recognizer固有 `signal_type` をパターン family へ正規化し、regime適合判定が常時ミスマッチになる問題を修正。  
6. `ztb/trading/strategies/action_signal_guide/action_signal_guide.py` の `_detect_market_regime()` を enum返却へ統一し、1回ごとの detector再生成を廃止（状態を維持）。  
7. `ztb/trading/strategies/action_signal_guide/components/dynamic_adapter.py` を `Mapping[str, object]` ベースへ移行し、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
8. `DynamicAdapter` に regime coercion を追加し、文字列/enum 混在入力でも `AdaptivePatternSelector` が正しく regime 別閾値を参照できるよう修正。  
9. `ztb/trading/environment/components/threshold_manager.py` の regime label 変換を刷新し、`moderate_bull_trend` 等の新 regime でも `trending_bull/bear/ranging/volatile` へ正規化。  
10. `ztb/trading/environment/components/dynamic_reward_shaper.py` で enum regime を文字列へ正規化し、`ranging` / `high_volatility_ranging` を正しく報酬調整へ反映。  
11. `ztb/trading/strategies/action_signal_guide/components/signal_quality_filter.py` の regime判定を bucket 化し、`MODERATE_BULL_TREND` 等の新規 enum 名でも quality alignment が有効に働くよう修正。  
12. `ztb/analysis/regime/market_regime_types.py` の `RegimeDetectionResult.metadata` を `Dict[str, object]` 化し、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
13. repo 全体 `any_type_debt_tokens` を `3,012 -> 3,004` へ削減。  

### Step42: 生成/品質評価の性能最適化 + `threshold_manager` の `Any` 全撤去

1. `ztb/trading/strategies/action_signal_guide/components/signal_quality_filter.py` に `MarketQualityContext` を導入し、bar単位の trend/volatility/regime bucket を事前計算して各 signal 評価で再利用。  
2. `filter_signals()` の評価順を見直し、現バーの market factor を先に反映してから risk-adjusted score を算出するよう修正（前バー係数混入リスクを低減）。  
3. reliability 算出に per-bar cache（`_get_cached_reliability_score`）を追加し、同一 pattern の履歴集計重複を削減。  
4. mean-reversion 判定の文字列分岐を `_is_mean_reversion_pattern()` へ抽出し、重複条件式を集約。  
5. `SignalQualityEvaluator.evaluate_signal_quality()` に `precomputed_volatility` を追加し、`SignalGenerator` 側で volatility を1回だけ計算して使い回す構成へ変更。  
6. `ztb/trading/strategies/action_signal_guide/components/signal_generator.py` の parallel 実行を persistent `ThreadPoolExecutor` 再利用へ変更し、barごとのスレッドプール生成/破棄オーバーヘッドを削減。  
7. parallel future 回収を `as_completed` 化し、個別 future 失敗時も全体を継続できるよう例外処理を強化。  
8. `SignalGenerator.close()` / `_shutdown_parallel_executor()` を追加し、明示的なリソース解放経路を整備。  
9. `ztb/trading/strategies/action_signal_guide/components/dynamic_adapter.py` の `adaptation_history` を `deque(maxlen=1000)` 化し、長時間稼働時の履歴肥大化を抑制。  
10. `ztb/trading/environment/components/threshold_manager.py` の `Any` を全撤去し、`TypedDict`（`AdaptiveSignalThresholds` / cache entry）と `Mapping[str, object]` ベースへ移行（当該ファイル `any_type_debt_tokens=0`）。  
11. `threshold_manager` の `signal_history` を `deque(maxlen=performance_memory)` 化し、履歴トリムの重複処理を削除。  
12. `calculate_adaptive_signal_thresholds()` の cache key にデータ終端 index を含め、同一長データでの誤キャッシュヒットを低減。  
13. repo 全体 `any_type_debt_tokens` を `3,004 -> 3,000` へ削減。  

### Step43: `TrendPatternRecognizer` 導入による継承ベース重複削減 + `dow/granville` 高速化 + `Any` 全撤去

1. `ztb/trading/strategies/action_signal_guide/pattern_recognition/base.py` に `TrendPatternRecognizer` を追加し、`resolve_analysis_index` / `safe_ratio` / `calculate_normalized_slope` / `slope_direction` を共通化。  
2. `TrendPatternRecognizer` に回帰用重みキャッシュ（window長ごと）を導入し、固定窓（10/20/50）での反復 slope 計算コストを削減。  
3. `ztb/trading/strategies/action_signal_guide/pattern_recognition/dow_theory.py` を `TrendPatternRecognizer` 継承へ変更し、primary/secondary/short の重複 trend 解析を `_analyze_trend()` に統合。  
4. `dow_theory` の slope 計算を `np.polyfit` 反復から共通の軽量回帰ロジックへ置換し、認識処理のオーバーヘッドを削減。  
5. `dow_theory` の confidence 算出で `min(0.0001, ...)` により常時極小値へ潰れる不具合を修正し、下限付き上限制御（`max(0.0001, ...)` + cap）へ是正。  
6. `ztb/trading/strategies/action_signal_guide/pattern_recognition/granville_law.py` を同基底へ移行し、index解決・比率計算の重複処理を削減。  
7. `dow_theory.py` / `granville_law.py` の `Any` を全撤去し、`TypedDict` + `Mapping[str, object]` ベースへ移行（両ファイル `any_type_debt_tokens=0`）。  
8. repo 全体 `any_type_debt_tokens` を `3,000 -> 2,980` へ削減。  

### Step44: `gann_analysis` の継承整理・重複削減・計算軽量化 + `Any` 全撤去

1. `ztb/trading/strategies/action_signal_guide/pattern_recognition/gann_analysis.py` を `Any` 依存から `TypedDict` / `dataclass` / `Mapping[str, object]` ベースへ移行し、当該ファイル `any_type_debt_tokens=0` を達成。  
2. `GannPatternBase`（`CandlestickPatternRecognizer` 継承）を追加し、`GannAngleRecognizer` / `GannSquareRecognizer` / `GannTimeClusterRecognizer` で重複していた index 解決・lookback 切り出し・市場コンテキスト計算を共通化。  
3. `GannAnalyzer` に `calculate_gann_angle_prices_at_time()` と `calculate_gann_square_levels()` を追加し、認識時に不要な full-series 生成（配列構築・文字列キー分解）を回避。  
4. `GannAngleRecognizer` の pivot 時刻算出を bars-ago ベースに是正し、直近 pivot を選んだ際の off-by-one リスクを低減。  
5. `gann` 系 3 recognizer で `compute_sma` / volatility ratio / trend strength 計算の重複を `GannPatternBase._calculate_market_context()` に統合し、保守性と実行効率を改善。  
6. 互換性維持のため `GannAnalyzer.calculate_gann_angles()` / `calculate_gann_square()` は残しつつ、内部に軽量 API を追加する構成へ整理。  
7. repo 全体 `any_type_debt_tokens` を `2,980 -> 2,973` へ削減。  

### Step45: `validation/pattern_statistics` の `Any` 全撤去 + レポート型固定 + 履歴管理整理

1. `ztb/trading/strategies/action_signal_guide/components/validation.py` に `ValidationRuleResult` / `SanitizationResult` / `PerformanceRecord` などの `TypedDict` を導入し、当該ファイル `any_type_debt_tokens=0` を達成。  
2. `SignalValidator` の rule registry / history payload を型固定し、`ValidationResult.metadata` を `ValidationMetadata` へ明示化。  
3. `DataSanitizer._remove_outliers()` で「copy に書かず入力参照へ書いてしまう」不整合を修正し、実際に返却データへ外れ値置換が反映されるよう是正。  
4. `DataSanitizer._normalize_data()` で datetime index 変換先が元 `data` へ向いていた不整合を修正し、`normalized_data` への反映に統一。  
5. `PerformanceTracker` の cache key を tuple 化、標準偏差計算を `ddof=0` + `NaN` ガードへ整理し、少量サンプル時の `NaN` 混入リスクを低減。  
6. `ztb/trading/strategies/action_signal_guide/components/pattern_statistics.py` に `DetectionHistoryEntry` / `PatternCombinationStats` / `TemporalPatternStats` などを追加し、当該ファイル `any_type_debt_tokens=0` を達成。  
7. `pattern_statistics` の履歴圧縮ロジックを `_append_with_compaction()` に集約し、`strength/confidence/accuracy/temporal` の重複トリム処理を削減。  
8. `detection_history` を `deque(maxlen=...)` 化し、長期稼働時の履歴肥大を抑止。  
9. repo 全体 `any_type_debt_tokens` を `2,973 -> 2,953` へ削減。  

### Step46: 既存ヘルパ拡張 + 水平展開（`cache/sac/plugin`）+ `Any` 全撤去

1. `ztb/trading/strategies/action_signal_guide/components/history_helpers.py` を追加し、既存の履歴圧縮ヘルパを `append_with_compaction()` として共通化（`retain/high_water` ガード付き）。  
2. `pattern_statistics.py` はローカル `_append_with_compaction` を削除し、新 helper へ委譲する構成へ変更（既存機能を維持しつつ重複を削減）。  
3. `sac_integration.py` へ helper を水平展開し、`correlation_history` / `integration_history` / `performance_history` のトリム処理を共通化。  
4. `sac_integration.py` に `SACDecisionPayload` / `SACValidationResult` / `IntegratedDecision` などの `TypedDict` を導入し、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
5. `sac_integration.py` で action 正規化（enum/str 混在）と数値 coercion を追加し、`.upper()` 直接呼び出しや不正型混入での例外リスクを低減。  
6. `sac_integration.py` で `market_data` の列存在ガード（`close`）を追加し、データ欠損時の KeyError リスクを低減。  
7. `cache_manager.py` を generic cache (`TypeVar`) へ整理し、`signal_cache` の実データ契約（単体/複数 signal）を型注釈へ反映。`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
8. `plugin_manager.py` を `PluginMetadata` + `PluginType` で型固定し、metadata 生成を `_build_plugin_metadata()` へ集約。`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
9. `components` 配下 `any_type_debt_tokens` を `36 -> 10` へ削減し、repo 全体 `any_type_debt_tokens` を `2,953 -> 2,927` へ削減。  

### Step47: helper 横展開拡大（`performance/market_regime/signal_generator`）+ components `Any` 完全解消

1. `ztb/trading/strategies/action_signal_guide/components/advanced_signal_aggregator.py` に `SACAggregationContext` (`TypedDict`) を導入し、`sac_context` の `Any` を撤去（当該ファイル `any_type_debt_tokens=0`）。  
2. 同ファイルで SAC payload の action/reward coercion を追加し、型揺れ入力（str/float/int 混在）での集約失敗リスクを低減。  
3. `weighted_direction == 0` で sell 側へ倒れていた分岐を neutral (`direction=0`) に是正し、閾値近傍での誤方向シグナルを抑制。  
4. `ztb/trading/strategies/action_signal_guide/components/performance_tracker.py` を `dict[str, object]` / `TypedDict` ベースへ移行し、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
5. `performance_tracker` の `max_history_size` 未定義参照を解消（`self.max_history_size=1000` + high-water 管理）し、長時間稼働時の実行時エラー要因を除去。  
6. `performance_tracker` で `signal_generation/pattern/cache/memory/SAC correlation` の履歴トリムを `append_with_compaction()` に統一し、重複処理と `pop(0)` 系 O(n) コストを削減。  
7. `ztb/trading/strategies/action_signal_guide/components/adaptive_pattern_selector.py` の `config` / 統計返却を型固定し、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
8. `adaptive_pattern_selector` では performance 集計を単一ループ化し、`list(...)` + 複数 `np.mean` の重複計算を削減。  
9. 同ファイル `_adapt_thresholds()` の重複ループ（未使用平均算出）を除去し、保守性を改善。  
10. helper の追加横展開として `market_regime.py` と `signal_generator.py` でも `append_with_compaction()` を適用し、履歴上限管理の重複を削減。  
11. `components` 配下 `any_type_debt_tokens` を `10 -> 0` へ削減し、repo 全体 `any_type_debt_tokens` を `2,927 -> 2,917` へ削減。  

### Step48: helper 横展開（`analysis/ml_integration`）+ 重複経路整理 + `Any` 追加削減

1. `ztb/trading/strategies/action_signal_guide/analysis/signal_performance_analyzer.py` を `TypedDict`（`SignalQualityRecord` / `SACLearningLog` / `SACCorrelationRecord` など）ベースへ移行し、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
2. 同ファイルで `append_with_compaction()` を導入し、`signal_quality_history` / `signal_sac_correlations` / `signal_contribution_scores` の履歴管理を共通化。  
3. `analyze_sac_learning_correlation()` の return 後に残っていた未到達な重複ブロック（別形式の correlation result 生成）を削除し、分析経路を単一化。  
4. rolling window 算出を `_rolling_windows()` に抽出し、window 生成の重複・0/1窓起因の例外ノイズを低減。  
5. `ztb/trading/strategies/action_signal_guide/ml_integration/pattern_optimizer.py` の `_append_performance_snapshot()` を helper 化し、手動 `del` トリムを `append_with_compaction()` へ置換。  
6. `action_signal_guide` 配下 `any_type_debt_tokens` を `71 -> 58` へ削減し、repo 全体 `any_type_debt_tokens` を `2,917 -> 2,904` へ削減。  

### Step49: 既存型への収束（`oscillator/strategy_allocator`）+ 最適化重複の抽出

1. `ztb/trading/strategies/action_signal_guide/pattern_recognition/oscillator_patterns.py` を `PatternConfig` / `MultiTimeframeData` / `RegimeAdjustment` へ寄せ、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
2. `oscillator_patterns.py` に `_iter_multi_timeframe_frames()` / `_coerce_level()` を追加し、各認識器で重複していた MTF dataframe 走査と threshold coercion を共通化。  
3. `CCI/Stochastic/WilliamsR/MFI` の8箇所で同型だった `for timeframe in multi_timeframe_data` + `'data'` 存在判定を helper 経由へ置換し、重複分岐を削減。  
4. `ztb/trading/strategies/action_signal_guide/portfolio_optimization/strategy_allocator.py` を `PayloadMap` ベースへ移行し、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
5. `strategy_allocator.py` で `_optimize_weights()` を導入し、`risk_parity` / `maximum_sharpe` / `minimum_variance` の同型最適化セットアップ（境界・制約・初期値）を共通化。  
6. `strategy_allocator.py` で `PortfolioAllocation` 生成時の非互換フィールド（`expected_risk`, `metadata` 等）を是正し、空配分/リバランス経路の実行時 `TypeError` リスクを解消。  
7. `strategy_allocator.py` の配分履歴を `append_with_compaction()` で bounded 管理し、履歴肥大化を抑制。  
8. `action_signal_guide` 配下 `any_type_debt_tokens` を `58 -> 39` へ削減し、repo 全体 `any_type_debt_tokens` を `2,904 -> 2,885` へ削減。  

### Step50: 指標系の共通基底導入（`RSI/MACD/ATR`）+ 既存不具合修正 + `Any` 追加削減

1. `ztb/trading/strategies/action_signal_guide/pattern_recognition/base.py` に `IndicatorPatternRecognizer` / `IndicatorMarketContext` を追加し、指標系で重複していた `index` 解決・相場コンテキスト（volatility/trend）算出・MTF confidence 算出・regime cluster 抽出を継承で共通化。  
2. `PatternRecognizer` へ `resolve_analysis_index` / `safe_ratio` / `clamp` を昇格し、`TrendPatternRecognizer` と指標系の共通ユーティリティを基底へ集約。  
3. `rsi.py` を `IndicatorPatternRecognizer` 継承へ移行し、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
4. `RSI` 認識で `index=-1` 時に previous 値参照が崩れてクロス判定が不正化する既存不具合を修正（`resolved_index` ベースへ統一）。  
5. `macd.py` を `IndicatorPatternRecognizer` 継承へ移行し、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
6. `MACD` 認識で `index=-1` 時に previous histogram が `0` 固定化する不具合を修正し、クロス判定を正常化。  
7. `MACD` で regime 調整後の `histogram_threshold` が実際には未使用だった不整合を修正し、判定ロジックに反映。  
8. `MACDPatternRecognizer.calculate()` を追加し、line/signal/histogram の算出 API を補完。  
9. `atr.py` を `IndicatorPatternRecognizer` 継承へ移行し、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
10. `ATR` の `avg_atr=0/NaN`・`recent_price=0`・`recent_atr=0` 由来の 0 除算/不正値混入リスクを `safe_ratio` ベースでガード。  
11. `ATR` の非MTF/MTF 経路を委譲構造に整理し、ブレイクアウト・トレンド判定の重複ロジックを削減。  
12. `pattern_recognition` 配下 `any_type_debt_tokens` を `25 -> 13` へ削減し、repo 全体 `any_type_debt_tokens` を `2,885 -> 2,873` へ削減。  

### Step51: メモリリーク防止策（認識器ライフサイクルの明示解放 + キャッシュ上限管理）

1. `ztb/trading/strategies/action_signal_guide/pattern_recognition/base.py` の `PatternRecognizer` に `close()` / `clear_runtime_state()` / `__del__` を追加し、`_signal_cache` を明示解放できるように変更。  
2. `recognize_with_cache()` に定期クリーンアップ（`_maybe_cleanup_signal_cache`）を追加し、長時間稼働で期限切れエントリが残留し続ける経路を抑制。  
3. `TrendPatternRecognizer` の `_regression_cache` を無制限 `dict` から上限付き `OrderedDict`（LRU）へ変更し、可変 window 長が増える運用でのキャッシュ肥大化リスクを低減。  
4. `ztb/trading/strategies/action_signal_guide/pattern_recognition/harmonic_patterns.py` に `HarmonicAnalyzer.clear_cache()` を追加し、`_HarmonicPatternBase.clear_runtime_state()` から pivot/pattern cache を明示解放。  
5. `ztb/trading/strategies/action_signal_guide/pattern_recognition/fibonacci_patterns.py` に `FibonacciAnalyzer.clear_retracement_cache()` を追加し、共有 retracement cache を解放可能に。  
6. `ztb/trading/strategies/action_signal_guide/components/signal_generator.py` に `_close_recognizers()` を追加し、`initialize_recognizers()` の再初期化時と `close()` 時に全 recognizer の `close()` を呼んでキャッシュを解放。  
7. `SignalGenerator.close()` で recognizer 解放に加えて、`pattern_performance_history` / `adaptive_weights` / runtime context を clear するよう変更。  
8. `SignalGenerator.__del__()` は `close()` 委譲へ統一し、スレッド停止とキャッシュ解放の後始末経路を単一化。  
9. 今回は leak 防止が主目的のため `Any` 追加削減は無し（repo 全体 `any_type_debt_tokens` は `2,873` を維持）。  

### Step52: 特徴量計算コストの水平展開削減（`RSI/MACD/ATR/ADX/oscillator`）

1. `pattern_recognition/base.py` の `IndicatorPatternRecognizer` に `build_indicator_view()` を追加し、指標計算を「対象 index までの固定窓」へ統一。  
2. `rsi.py` / `macd.py` / `atr.py` を fixed-window 計算へ移行し、長系列での full-series 再計算を回避（O(N) -> O(W) 化）。  
3. 上記 3 recognizer は `index` より後方データを使わない構造へ変更し、履歴評価時の将来データ混入リスクを低減。  
4. `adx_patterns.py` を `IndicatorPatternRecognizer` 継承へ移行し、固定窓計算・index 正規化・`adx_change` 0除算ガードを適用。  
5. `oscillator_patterns.py`（`CCI/Stochastic/WilliamsR/MFI`）で `index` 正規化と fixed-window 計算を導入。  
6. `oscillator` 4 recognizer の MTF alignment で、base timeframe 指標の再計算（重複 `compute_*`）を廃止し、認識時に算出済みの current 値を再利用。  
7. `analysis_window` を各 oscillator recognizer に追加し、計算窓を config で調整可能にした（デフォルト 240）。  
8. `Any` 負債は増減なし（repo 全体 `any_type_debt_tokens` は `2,873` を維持）。  

### Step53: `pattern_recognition` 完全 `Any=0` 化 + レジーム/トレンド系の不具合修正 + 計算量削減

1. `ztb/trading/strategies/action_signal_guide/pattern_recognition/trend_analyzer.py` を `TrendPatternRecognizer` 継承へ移行し、`Any` を全撤去（当該ファイル `any_type_debt_tokens=0`）。  
2. `HierarchicalTrendAnalyzer.recognize()` の `index=-1` 既定時に常時 `None` を返しうる不具合を修正（index 正規化 + bounded window 化）。  
3. `trend_analyzer.py` の ADX 計算で scalar 化後に `.rolling()` を呼んで例外に落ちる既存不具合を修正し、DI/DX/ADX の series 計算へ是正。  
4. `trend_analyzer.py` の pivot 探索を二重ループから rolling ベースへ変更し、波動検出の計算コストを削減。  
5. `ztb/trading/strategies/action_signal_guide/pattern_recognition/heikin_ashi.py` を `dict[str, object]` / `TypedDict` ベースへ移行し、`Any` を全撤去。  
6. `heikin_ashi.py` は `index` 解決を共通化し、対象 index までのデータで判定する構造へ変更（将来データ参照を抑止）。  
7. `ztb/trading/strategies/action_signal_guide/pattern_recognition/volume_patterns.py` を `IndicatorPatternRecognizer` 継承へ移行し、固定窓計算・`safe_ratio` ガード・`Any` 全撤去を実施。  
8. `ztb/trading/strategies/action_signal_guide/pattern_recognition/ichimoku.py` を `IndicatorPatternRecognizer` 継承へ移行し、固定窓計算・重複 `compute_*` 呼び出し削減・debug print/traceback 除去を実施。  
9. `ztb/trading/strategies/action_signal_guide/pattern_recognition/bollinger_patterns.py` を型統一し、index 正規化 + bounded analysis window を導入して計算量を削減。  
10. `ztb/trading/strategies/action_signal_guide/recognizer_factory.py` の `Any` 注釈を全撤去し、factory map / create API を `dict[str, object] | None` に統一。  
11. `pattern_recognition` 配下 `any_type_debt_tokens` を `13 -> 0` に削減。repo 全体 `any_type_debt_tokens` を `2,873 -> 2,858` に削減。  
12. 回帰確認: `test_bollinger_adx_recognizers` (17 pass), `test_new_recognizers` (12 pass), `test_hierarchical_trend_analyzer` (3 pass, 新規), `test_ichimoku_recognizer` (2 pass), `test_enhanced_recognizers` (1 pass)。  

### Step54: 残タスク水平展開（`multi_timeframe_analyzer` + base/oscillator 重複削減）

1. `pattern_recognition/base.py` に `PatternRecognizer.iter_multi_timeframe_frames()` を追加し、MTF payload から dataframe を抽出する重複ロジックを共通化。  
2. `base.py` の `_analyze_multi_timeframe_alignment()` で `index<=0` 時に前足参照が末尾へ回り込む境界不具合を修正（neutral return へ変更）。  
3. 同メソッドの trend 判定を 3値化（up/down/flat）し、flat な時間足を無理に逆張り扱いする誤差を抑制。  
4. `base.py` の `_adjust_thresholds_for_regime()` を共通ヘルパベースへ置換し、`for tf in multi_timeframe_data` の重複分岐を削減。  
5. 同メソッドの trend 強度計算を `np.polyfit` から first-last 比率へ変更し、軽量化。  
6. `oscillator_patterns.py` からローカル `_iter_multi_timeframe_frames()` を削除し、基底共通ヘルパへ統一（重複削減）。  
7. `multi_timeframe_analyzer.py` に回帰傾きキャッシュ（LRU）を導入し、`_calculate_trend_strength()` の反復 `np.polyfit` を削減。  
8. `multi_timeframe_analyzer.py` の primary timeframe 参照を安全化し、`primary` が存在しても `data` 欠落時に `KeyError`/`TypeError` へ落ちる経路を解消。  
9. `_calculate_level_consensus()` の多数決 tie を neutral 扱いへ修正し、同数拮抗時に bearish へ偏る既存挙動を解消。  
10. `_is_consolidation()` の `avg_price=0` 0除算ガードを追加。  
11. 新規テスト `test_pattern_multi_timeframe_analyzer.py` を追加（共通ヘルパ抽出、tie consensus、primary欠落、0価格ガード）。  
12. `Any` 負債は維持（`pattern_recognition any_type_debt_tokens=0`, repo 全体 `any_type_debt_tokens=2,858`）。  
13. 回帰確認: `test_pattern_multi_timeframe_analyzer` (4 pass), `test_bollinger_adx_recognizers + test_new_recognizers + test_hierarchical_trend_analyzer` (32 pass)。  

### Step55: `action_signal_guide` 全域 `Any=0` 化 + config型安全化 + Fibonacci互換修正

1. `ztb/trading/strategies/action_signal_guide/types.py` を `ConfigSection` / `ObjectMap` ベースへ移行し、`Any` alias を全撤去。  
2. `types.py` の `MultiTimeframeData` を `dict[str, dict[str, object]]` に統一し、各認識器の既存 payload 契約と整合。  
3. `ztb/trading/strategies/action_signal_guide/config/asg_portfolio_config.py` の `Any` を全撤去し、`StressTestScenario` / `AllocationConstraintsPayload` / `RiskLimitsPayload` / `OptimizationSchedulePayload` (`TypedDict`) を導入。  
4. `ztb/trading/strategies/action_signal_guide/config/asg_adaptation_config.py` の `Any` を全撤去し、trigger/processing/schedule の戻り値を `TypedDict` 化。  
5. `ztb/trading/strategies/action_signal_guide/config/asg_ml_config.py` の `hyperparameters` を `dict[str, ConfigValue]` に型固定し、`Any` を全撤去。  
6. `ztb/trading/strategies/action_signal_guide/pattern_recognition/fibonacci_patterns.py` で MTF dataframe 抽出を `PatternRecognizer.iter_multi_timeframe_frames()` へ統合（重複削減）。  
7. `FibonacciAnalyzer.calculate_deviation_from_ideal()` に後方互換経路（`levels: dict[float, float]` 入力）を追加し、既存テスト互換を回復。  
8. `action_signal_guide` 配下 `any_type_debt_tokens` を `12 -> 0` に削減し、repo 全体 `any_type_debt_tokens` を `2,858 -> 2,846` へ削減。  
9. 回帰確認: `test_fibonacci_recognizer` + `test_pattern_recognition -k Fibonacci` で `14 passed`。  

### Step56: `ensemble_signal_generator` の `Any=0` 化 + 重複削減 + 不具合/性能改善

1. `ztb/trading/signal/ensemble_signal_generator.py` の `Any` を全撤去し、`MarketData` / `SignalReliability` (`TypedDict`) と `SignalScorer` (`Protocol`) で入出力契約を型固定。  
2. `generate_ensemble_signal()` と `get_signal_reliability()` で重複していた scorer 全走査を `_collect_scores_and_confidences()` に集約。信頼性計算時の二重計算を解消。  
3. 重み計算を `_resolve_weights()` / `_calculate_weighted_score()` / `_calculate_final_confidence()` に分離し、責務を明確化。`normalize_weights()` を再利用して重複正規化実装を削減。  
4. `PatternRecognitionScorer` のトレンド傾きを `np.polyfit` から centered 回帰（cache 付き）へ置換し、反復呼び出し時の割当と計算コストを削減。  
5. 反転判定の参照窓を「直近5本（現足除外）」へ修正し、従来条件が成立しにくかったロジック不整合を解消。  
6. `VolumeProfileScorer.get_confidence()` に平均出来高ゼロ/NaN/inf ガードを追加し、不正 confidence 値混入を抑止。  
7. 既存テスト不具合として `tests/unit/trading/signal/scorers/test_signal_scorers.py` の未定義変数（`no_volume_df`）を修正。  
8. 回帰確認: `test_ensemble_signal_generator` + `test_signal_scorers` で `46 passed`。  
9. 在庫更新: `ztb/trading/signal/ensemble_signal_generator.py` 単体 `any_type_debt_tokens=0`、repo 全体 `2,846 -> 2,817`。  

### Step57: `quality/indicators` 基底の継承整理 + `Any=0` 化 + 適応ロジック修正

1. `ztb/trading/signal/quality/indicators/base.py` の `Any` を全撤去し、`IndicatorConfig` / `IndicatorResult` を `object` ベースへ統一。  
2. `BaseTechnicalIndicator` に `temporary_config()` / `on_config_updated()` を追加し、継承先が config 一時上書きロジックを共通利用できるよう整理。  
3. `BaseTechnicalIndicator.calculate()` は cached result をコピー返却に変更し、呼び出し側更新でキャッシュ内容が破壊されるリスクを低減。  
4. `AdaptiveIndicator` の重複していた適応計算経路を `_calculate_with_regime()` に統合し、`calculate()` / `calculate_adaptive()` の実装差分を解消。  
5. `AdaptiveIndicator` は base indicator の結果 dict を直接更新しないよう修正し、base 側キャッシュ汚染（regime metadata 混入）不具合を解消。  
6. `AdaptiveIndicator` は `temporary_config` 非対応の mock 指標でも動く fallback を追加し、既存テスト互換を維持。  
7. `ztb/trading/signal/quality/indicators/rsi.py` / `macd.py` へ `on_config_updated()` を水平展開し、adaptive config 変更時に `periods` / `fast_period` / `slow_period` / `signal_period` が即時反映されるよう修正。  
8. `quality/indicators` 配下 `any_type_debt_tokens` を `23 -> 0` に削減。repo 全体 `any_type_debt_tokens` は `2,817 -> 2,786`。  
9. 回帰確認: `test_signal_indicators` + `test_modular_indicators` で `60 passed`。  

### Step58: `end_to_end_validation` / `e2e_test_framework` の広域浅層改善（型安全 + 不具合修正）

1. `ztb/trading/end_to_end_validation.py` の `Any` を全撤去し、`ObjectMap` / `FloatMap` / `StringMap` を導入して payload 型を統一。  
2. `ComponentIntegrationTester` に `_get_component_manager()` を追加し、`component_manager` 未初期化時に `AttributeError` へ落ちる経路を各統合テストでガード。  
3. `end_to_end_validation.py` の nested dict 参照を `_as_object_map()` / `_to_float()` 経由へ統一し、`KeyError` / 型揺れ起因の例外リスクを低減。  
4. `end_to_end_validation.py` の performance 応答時間計測を `time.perf_counter()` 化し、短時間測定の精度を改善。  
5. `ztb/trading/e2e_test_framework.py` の `Any` を全撤去し、`ExpectedPredicate` / `ExpectedValue` を導入して expected_results の契約を明確化。  
6. `_validate_test_results()` は callable expected（`lambda x: ...`）を正しく評価するよう修正。従来は predicate が実質未検証だった不具合を解消。  
7. 同メソッドで `bool` 判定を数値判定より先に実施するよう修正し、`bool` が `int` として扱われる誤判定を防止。  
8. `e2e_test_framework.py` の `_test_signal_processing_rate()` で未定義 `send_signals` 呼び出し不具合を修正（async helper を復元）。  
9. `_test_memory_pressure()` に `initial_memory=0` ガードを追加し、0除算リスクを解消。  
10. 在庫更新: `end_to_end_validation.py` / `e2e_test_framework.py` はともに `any_type_debt_tokens=0`。repo 全体は `2,786 -> 2,747`、`ztb/trading` は `554 -> 515`。  
11. 回帰確認: `py_compile`（2ファイル）通過。  

### Step59: `integrated_backtest_runner` の集計最適化（計算量削減 + 不具合修正 + `Any=0`）

1. `ztb/trading/backtest/integrated_backtest_runner.py` の `Any` を全撤去し、`ObjectMap` / `IterationList` / `TradeList` へ統一（当該ファイル `any_type_debt_tokens=0`）。  
2. 内部クラス `FunctionStrategyAdapter` を module-level `_FunctionStrategyAdapter` へ昇格し、毎回生成されるクラス定義コストを削減。  
3. 戦略出力の互換吸収を強化し、`signal` payload だけでなく `action` 文字列系出力も受理。`dict` 入力失敗時は DataFrame 入力へ後方互換フォールバック。  
4. `_run_enhanced_backtest()` で `initial_capital` / `commission`（decimal or bps）を `BacktestEngine` へ反映するよう修正。従来の「引数が実質未反映」不整合を解消。  
5. リスク管理統合で ATR を取引ごと再計算していた経路を修正し、イテレーションごと1回の事前計算へ変更（`precomputed_atr` 注入）。  
6. `_aggregate_results()` を `numpy` 配列ベースへ最適化し、`std`/`mean` 再計算と `np.std(total_returns)` の重複呼び出しを解消。  
7. `_validate_statistically()` は単一ループ化し、`portfolio_values -> returns` の重複計算（二重ループ）を解消。  
8. `_calculate_returns_from_portfolio_values()` を `np.diff` + `np.divide(where=...)` でベクトル化し、ゼロ除算/NaN/inf を安全に無害化。  
9. `_aggregate_results()` の空 `portfolio_values` 時 `[-1]` 参照クラッシュと、`n_iterations=0` 時の 0除算リスクをガード。  
10. 設定で無効化されている場合は `risk_analysis` / `statistical_validation` の重い集計をスキップし、不要計算を抑制。  
11. 回帰確認: `py_compile`（対象ファイル）通過。`pytest` はこの環境で未導入のため未実施。  
12. 在庫更新: repo 全体 `any_type_debt_tokens` は `2,747 -> 2,728`、`ztb/trading/backtest/integrated_backtest_runner.py` は `19 -> 0`。  

### Step60: `data_validation` / `utils.config` の型固定 + 重複削減（`Any=0`）

1. `ztb/data/data_validation.py` の `Any` を全撤去し、`SchemaValidationResult` / `RuleEvaluationResult` / `IntegrityCheckResult` などの `TypedDict` で結果payload契約を固定化（当該ファイル `any_type_debt_tokens=0`）。  
2. 同ファイルで `_append_rule_messages()` / `_safe_ratio()` を導入し、ルール結果のメッセージ振り分け・除算ガードの重複処理を共通化。  
3. `column_uniqueness` の分母 0（全欠損列）と統計検証の `expected_mean/std=0` で 0除算しうる経路を修正。  
4. `DataIntegrityChecker._check_data_types()` の列メトリクス算出が「先行列のエラーに引きずられて後続列も 0 になる」不整合を修正し、列単位評価へ変更。  
5. `ztb/utils/config.py` の `Any` を全撤去し、`ObjectMap` / `object` ベースへ型統一（当該ファイル `any_type_debt_tokens=0`）。  
6. `utils.config` の環境変数変換ロジックを `_convert_env_value()` に集約し、`validate_config()` / `get_validated_config()` の重複分岐を解消。  
7. `get_config_value()` の list/dict JSON 解析重複を `_parse_json_value()` / `_coerce_json_container()` に集約し、保守性を向上。  
8. `TypedConfig.__init__()` の validator 実行を例外安全化し、型不整合入力時の曖昧な失敗経路を明示 `ValueError` へ統一。  
9. 回帰確認: `py_compile`（2ファイル）通過。`pytest` はこの環境で未導入のため未実施。  
10. 在庫更新: `data_validation.py` / `utils/config.py` はともに `any_type_debt_tokens=0`。repo 全体は `2,728 -> 2,687`（-41）。  

### Step61: JSON 読み書き重複の共通化 + MTF 候補生成バグ修正（`Any=0`）

1. `ztb/io/json_io.py` に `read_json_object()` / `read_json_array()` を追加し、JSONオブジェクト/配列前提の呼び出しで型契約を明示。  
2. `ztb/io/__init__.py` から新 helper を re-export し、既存 import 経路で水平展開しやすい形へ整理。  
3. `ztb/utils/safety.py` の `safe_open_json()` を `read_json_object()` 経由へ統一し、JSON 読み込み実装重複を削減。  
4. 同ファイルの `safe_config_get*` / `safe_get_nested_value` / `safe_list_get` を `object` ベースへ型固定し、`Any` を全撤去（`any_type_debt_tokens=0`）。  
5. `ztb/training/reward_function_optimizer/candidate_evaluator.py` の `json.loads(Path(...).read_text())` 重複を helper 化し、`CandidateEvaluationResult` を導入。  
6. `candidate_evaluator` は candidate config を先読みして `training.model_name` を一度だけ抽出する流れに変更し、再読込重複と失敗時分岐の複雑性を削減。  
7. `ztb/training/reward_function_optimizer/mtf_optimizer.py` を `read_json_object` / `write_json` へ統一し、JSON I/O 重複を解消。  
8. `mtf_optimizer.propose_candidates()` の shallow copy を `deepcopy` に変更。これにより、候補生成ごとに base 設定が汚染される不具合（`model_name` 連結肥大化/weights 連鎖変形）を修正。  
9. `ztb/training/reward_function_optimizer/mtf_scheduler.py` / `config_manager.py` / `reward_function_optimizer.py` へ同 helper を水平展開し、open/load/dump 実装を共通化。  
10. 回帰確認: `py_compile`（8ファイル）通過。`pytest` はこの環境で未導入のため未実施。  
11. 在庫更新: 対象 6ファイル（`json_io.py`, `safety.py`, `candidate_evaluator.py`, `mtf_optimizer.py`, `mtf_scheduler.py`, `config_manager.py`）は `any_type_debt_tokens=0`。repo 全体は `2,687 -> 2,664`（-23）。  

### Step62: `production` state 永続化の横断統合 + JSON 重複削減

1. `ztb/trading/production/state_persistence.py` を追加し、`write_state_payload()` / `read_state_payload()` を新設。  
2. `state_persistence` は `write_json()` / `read_json_object()` を内部利用し、`save_state/load_state` のファイルI/O重複を集約。  
3. 以下13モジュールで `os.makedirs + open + json.dump/load` を helper 呼び出しへ置換し、実装重複を削減。  
   `traffic_distributor.py`, `system_switcher.py`, `virtual_portfolio_manager.py`, `performance_monitor.py`, `health_checker.py`, `real_time_metrics.py`, `alert_system.py`, `emergency_stop.py`, `recovery_system.py`, `result_comparator.py`, `risk_based_allocator.py`, `rollback_manager.py`, `paper_trading_manager.py`  
4. `real_time_metrics.py` の JSON エクスポートも `write_json()` へ統一し、`json.dump` 直接呼び出しを削除。  
5. 横断調査: `trading/production` の `save_state/load_state` は 12コンポーネントで同型I/O重複を確認し、今回全て適用。  
6. 追加調査: `save_state/load_state` の残存候補は `ztb/trading/signal/entry_system.py` と `ztb/training/callbacks/monitoring/metrics_collector.py`。次フェーズ対象として記録。  
7. 回帰確認: `py_compile`（`state_persistence.py` + production 13ファイル）通過。  
8. 在庫更新: 今回は重複削減が主目的であり `Any` 量は据え置き。repo 全体 `any_type_debt_tokens=2,664`（変化なし）。  

### Step63: `signal/training` への state helper 水平展開 + metrics 不具合修正

1. 汎用 helper `ztb/io/state_persistence.py` を追加し、`write_state_payload()` / `read_state_payload()` を `ztb.io` から再利用可能に整理。  
2. 既存 `ztb/trading/production/state_persistence.py` は互換ラッパー化し、production 側 import 互換を保ったまま汎用 helper へ委譲。  
3. `ztb/trading/signal/entry_system.py` の `save_state/load_state` を helper 統合し、`open + json.dump/load` を除去。  
4. 同ファイルの正規化ロジック重複を `_normalize_action()` に抽出し、`process_signal()` と `update_outcome()` の重複分岐を一本化。  
5. `update_outcome()` は `threshold` にデフォルト `0.2` を追加し、旧呼び出し（4引数）との互換を維持。  
6. `ztb/training/callbacks/monitoring/metrics_collector.py` の `_export_json()` / `load_state()` を helper 統合し、`_serialize_metrics_payload()` / `_restore_metrics_payload()` に責務分離。  
7. `register_metric()` で `max_series_size` を実際の deque 上限へ反映し、設定と実体の不整合によるメモリ増加余地を解消。  
8. `get_latest_metrics()` キャッシュの無効化漏れ（`add_metric_value` / `_cleanup_old_data` / `load_state`）を修正し、更新後に古い値を返し続ける不具合を解消。  
9. `metrics_collector.get_performance_stats()` の `value_pool.pool_size` 参照不整合を `max_pool_size` へ修正。  
10. `metrics_collector` の危険な pooled object 再利用経路を撤去し、時系列データ参照破壊の潜在不具合を防止。  
11. `WeakRefRegistry` に `registry` property を追加し、統計取得時の属性不一致による例外を回避。  
12. 型更新: `ztb/trading/signal/types.py` の `GateResult` に `normalized_action: NotRequired[float]` を追加して契約を明示。  
13. 回帰確認: `py_compile`（7ファイル）通過。`pytest` は本環境で未導入のため未実施。  
14. 在庫更新: repo 全体 `any_type_debt_tokens=2,664 -> 2,620`（-44）。`ztb/io/state_persistence.py` / `ztb/trading/production/state_persistence.py` / `ztb/trading/signal/types.py` は `any_type_debt_tokens=0`。  

### Step64: `io` 横展開の追加圧縮（training/trading/ops/features）

1. `ztb/training/components/regime_adaptive_trainer.py` の JSON state I/O を `read_state_payload` / `write_state_payload` へ統合。  
2. 同ファイルで `max_history` / `adaptation_frequency` / `performance_tracking_window` の数値変換を共通化し、無効設定値で例外化しうる経路を修正。  
3. `regime_performance` 復元時に payload バリデーションを追加し、壊れた履歴データで全復元失敗する不具合リスクを低減。  
4. `ztb/trading/cost/venue_transaction_cost_manager.py` の JSON I/O を `read_json_object` / `write_json` へ統一。  
5. 同ファイルで venue 名を正規化（lowercase）して保存/検索契約を統一し、設定ファイル内の大文字混在で `get_cost_config()` が取りこぼす不整合を修正。  
6. 設定ロードは1件不正で全体失敗しないようレコード単位にバリデーションし、無効レコードはスキップする設計へ変更。  
7. `ztb/ops/health/performance_monitor.py` の履歴保存/読込を `write_json` / `read_json_array` へ統一。  
8. 同ファイルで履歴エントリ parser を追加し、timestamp/数値型の壊れた行をスキップする復元経路へ改善（履歴1件不正で全件失敗しない）。  
9. `ztb/features/feature_set_config.py` の config load/save を `read_json_object` / `write_json` へ統一し、`open + json.load/dump` 重複を削減。  
10. 回帰確認: `py_compile`（4ファイル）通過。`feature_set_config` は importlib 経由 smoke で save/load 往復を確認。  
11. 在庫更新: repo 全体 `any_type_debt_tokens=2,620 -> 2,610`（-10）。`regime_adaptive_trainer.py` / `venue_transaction_cost_manager.py` / `ops/health/performance_monitor.py` / `feature_set_config.py` は `any_type_debt_tokens=0`。  

### Step65: JSON/state helper の追加横展開（ops/features/utils/analysis/experiments）+ `Any` 追加削減

1. `ztb/ops/costs/budget_rollup.py` の JSON I/O を `read_json_object` / `write_text` へ統一し、`_iter_run_dirs` / `_load_run_json_file` / `_to_float` を追加して run 走査と数値変換の重複を削減。  
2. 同ファイルに `RunSummary` (`TypedDict`) を導入し、`aggregate_by_date` の payload 契約を明示。`cost_estimate` 欠損時の安全フォールバックを強化。  
3. `ztb/features/generators/multi_timeframe/config.py` の config load/save を `read_json_object` / `write_json` へ統一。  
4. 同ファイルで `_as_object_map` / `_as_string_list` / `_get_section` を導入し、section 取得の重複ロジックを共通化。`base_timeframe` 無効値は `"5min"` へ明示フォールバック。  
5. `ztb/utils/run_manifest.py` の manifest load/save を `read_json_object` / `write_json` へ統一し、`Dict[str, Any]` を `dict[str, object]` ベースへ移行。  
6. 同ファイルで `_as_object_map` / `_as_string_list` を追加し、`validate_manifest` / `compare_manifests` / `preflight_dataset_check` の nested dict 参照を安全化。  
7. `ztb/analysis/common/data_loaders.py` を `read_json_object` ベースへ寄せ、`Any` 注釈を全撤去（`dict[str, object]` に統一）。  
8. `ztb/experiments/run_sac_experiments.py` の設定/結果 I/O を `read_json_object` / `write_json` へ統一し、`HyperParams` / `ExperimentConfig` / `CommonConfig` / `ExperimentResult` (`TypedDict`) で契約を固定。  
9. 同スクリプトで config parser を追加し、無効な hyperparam / common config を早期検知するよう改善。summary 作成時の数値 coercion を共通化し、型揺れ入力での実行時例外リスクを低減。  
10. 回帰確認: `py_compile`（5ファイル + Step65対象）通過。  
11. 在庫更新: repo 全体 `any_type_debt_tokens=2,610 -> 2,571`（-39）。`budget_rollup.py` / `multi_timeframe/config.py` / `run_manifest.py` / `analysis/common/data_loaders.py` / `experiments/run_sac_experiments.py` は `any_type_debt_tokens=0`。  

### Step66: `multi_task/meta` への継承導入 + 重複削減 + 不具合修正（`Any=0`）

1. `ztb/training/callbacks/multi_task/multi_task_callbacks.py` に `_BaseFrequencyCallback`（`NoOpMemoryOptimizedCallback` 継承）を導入し、`TaskBalancing` / `SharedRepresentation` / `TaskInterference` の共通処理（`compute_frequency` gating + logger + no-op lifecycle）を基底化。  
2. 同ファイルで `super().__init__` 未実行だった `SharedRepresentationCallback` を修正し、`compute_frequency` / `representation_layers` 初期化欠落による実行時エラーを解消。  
3. `TaskInterferenceCallback` の未初期化属性（`task_interference_scores`, `interference_events`）を追加し、`on_epoch_end` での `AttributeError` リスクを解消。  
4. `TaskInterferenceCallback` に logs `None` ガードと frequency gating を追加し、空 payload での例外・不要計算を抑制。  
5. `multi_task` 内の no-op lifecycle メソッド重複を削除し、継承で吸収。  
6. `ztb/training/callbacks/meta/meta_callbacks.py` に `_BaseMetaCallback`（`NoOpMemoryOptimizedCallback` 継承）を導入し、`MAML` / `FewShot` / `MetaAdaptation` の共通 frequency 処理を集約。  
7. `MetaAdaptationCallback` の `super().__init__` と `compute_frequency` / `adaptation_steps` / `stability_threshold` 初期化欠落を修正し、設定未反映不具合を解消。  
8. `meta` 内で重複していた no-op lifecycle 実装を削除し、基底継承で統一。  
9. 両ファイルに `_as_float` / `_append_bounded` 等の共通 helper を導入し、履歴更新と型変換の重複ロジックを削減。  
10. 回帰確認: `py_compile`（2ファイル）通過。`pytest` は本環境で `numpy` 非導入のため未実施。  
11. 在庫更新: repo 全体 `any_type_debt_tokens=2,571 -> 2,537`（-34）。`multi_task_callbacks.py` / `meta_callbacks.py` は `any_type_debt_tokens=0`。  

### Step67: `distributed/performance` の継承導入 + 並行実行安定化 + `Any` 追加削減

1. `ztb/training/callbacks/distributed/threading_mixin.py` を追加し、`BackgroundThreadController`（背景スレッド start/join 共通基底）を導入。  
2. `DistributedCoordinator` / `WorkerPool` / `DistributedTrainingManager` に同基底を適用し、重複していた thread lifecycle 処理を継承で集約。  
3. `WorkerPool` は task ごとの無制限 thread 生成を廃止し、`ThreadPoolExecutor(max_workers=num_workers)` へ変更（スレッド増殖抑制）。  
4. `WorkerPool` の「round-robin 実装だが先頭 worker 固定」の不整合を修正し、実際にローテーションする選択へ変更。  
5. `WorkerPool` 内 callback closure の late-binding（`task_info`/`worker` 取り違え）リスクを `_on_task_done` 経由で解消。  
6. `result_queue` を bounded 化し、満杯時の古い結果破棄 (`_enqueue_result`) を追加して長時間運用時のメモリ膨張を抑制。  
7. `DistributedWorker.send_task()` を task lock で直列化し、`heartbeat/sync_ack` 混入時に task 結果待機を継続する方式へ修正（誤 `None` 返却リスク低減）。  
8. 同 worker で parent 側 status/stats 更新を lock 保護し、pool 側の可用判定・統計が実態から乖離しにくいよう改善。  
9. `DistributedTrainingManager` の同期 thread が `initialize()` 時点で即終了する不具合を修正し、`start_distributed_training()` で起動する設計へ変更。  
10. `distributed/integration.py` の `memory_monitor.emergency_cleanup()` 不整合を修正（`force_cleanup()` へ統一）。  
11. `ztb/training/callbacks/performance/memory_optimizer.py` を再構成し、`_ThreadSafeStatsBase` 継承導入、`memory_pressure` 追加、`emergency_cleanup` 互換 alias を導入。  
12. `Any` 削減: `worker.py` / `integration.py` / `memory_optimizer.py` / `threading_mixin.py` は `any_type_debt_tokens=0`。  
13. 回帰確認: `py_compile`（5ファイル）通過。`pytest` は本環境で未導入のため未実施。  
14. 在庫更新: repo 全体 `any_type_debt_tokens=2,537 -> 2,502`（-35）。`ztb/training` は `630 -> 595`。  

### Step68: callback 共通ヘルパ横展開 + coordinator 不具合修正 + I/O 微最適化

1. `ztb/training/callbacks/shared/utils/value_utils.py` を追加し、`as_optional_float` / `as_optional_array` / `append_bounded` を共通化。  
2. `supervised/sac/transfer/unsupervised/meta/multi_task` の6 callback ファイルで、重複していた `_as_float` / `_as_array(_to_array)` / `_append_bounded` の本体実装を共通 helper に統合。  
3. `ztb/training/callbacks/distributed/coordinator.py` に worker 状態アクセス helper（status/metrics/heartbeat）を導入し、dict/dataclass 分岐の重複を集約。  
4. 同 coordinator で `register_worker` の payload 正規化を追加し、`worker_id` 不正値受理や型揺れに起因する運用時不整合を低減。  
5. `coordinator._handle_error` / `_heartbeat_loop` を helper 経由へ統一し、worker 表現差異時に `status/last_heartbeat` 参照が崩れる潜在バグを解消。  
6. `ztb/training/unified_optimizer.py` の `safe_json_dump` 周辺で重複していた `open(..., 'w')` 包装を削除し、不要 I/O を削減。  
7. 回帰確認: `py_compile`（9ファイル）通過。`pytest` は本環境で未導入のため未実施。  
8. 在庫確認: Step68 対象 callback ファイル（7ファイル）は `any_type_debt_tokens=0` を維持。  
9. repo 全体在庫は `2,516`（前回 `2,502` から +14）。同時進行の別差分により `scanned_files: 1,280 -> 1,286` へ変動しており、今回対象ファイル起因の増加は確認されず。  

### Step69: JSON I/O 統合（experiments/results/optimization）+ 型安全化 + 潜在不具合修正

1. `ztb/training/run_optimization.py` の設定/結果I/Oを `read_json_object` / `write_json` へ統一し、`open + json.load/dump` の重複を削減。  
2. 同ファイルで `_write_temp_config` / `_run_unified_trainer` を追加し、一時 JSON 設定ファイル生成 + subprocess 実行の重複実装を共通化。  
3. `run_extended_backtest()` 内の `DataLoader` import インデント崩れ（実行時構文不整合リスク）を修正。  
4. `optimize_hyperparameters()` は `self.config.copy()` 由来の浅いコピー汚染を `deepcopy` 化で解消し、trial 間で設定が汚染される潜在不具合を排除。  
5. 同ファイルで `np.random.default_rng()` を導入し、乱数生成呼び出しを集約（可読性/軽量化を改善）。  
6. `ztb/utils/results_utils.py` の JSON 読み込みを `read_json_object` に統一し、`TrainingResultsPayload` / `BacktestMetricsPayload`（`TypedDict`）を導入して schema を固定。  
7. 同ファイルで `safe_json_dump` の失敗戻り値を検査し、保存失敗を成功扱いして処理継続する不具合余地を `OSError` 送出で解消。  
8. `ztb/experiments/base.py` の結果集約/保存を `read_json_object` / `write_json` へ統一し、`json.load/dump` の分散実装を整理。  
9. 同ファイルで `run_metadata` 文字列化を `_serialize_run_metadata()` に抽出して重複を削減し、`Any` 注釈を `object`/`ObjectMap` に置換。  
10. `checkpoint` 関連メソッド（step/get_checkpoint_data/load/checkpoint_save/load）の契約を `Any` から `object` ベースへ寄せ、型安全性を改善。  
11. 回帰確認: `py_compile`（`ztb/experiments/base.py` / `ztb/utils/results_utils.py` / `ztb/training/run_optimization.py`）通過。  
12. 在庫更新: repo 全体 `any_type_debt_tokens=2,516 -> 2,494`（-22）。`ztb/experiments/base.py` / `ztb/utils/results_utils.py` / `ztb/training/run_optimization.py` は `any_type_debt_tokens=0`。  

### Step70: 課題探索バッチ（広く浅く）- 不具合/重複/性能の優先度整理

1. `ztb/training/utils/sac_utils.py` を `py_compile` で検査し、構文エラー（未閉鎖 docstring）で import 不可を確認。  
2. 同ファイルで `self.project_root` の未初期化参照と `config_dir` / `data_dir` 名称不整合を確認し、実行時 `NameError` 発生リスクを特定。  
3. `ztb/experiments/job_manager.py` の並列実行経路を確認し、`ProcessPoolExecutor` で bound method + 任意 `train_function` を渡す設計が pickling 失敗しやすい構造であることを確認。  
4. 同ファイルで `future.result(timeout=...)` の timeout 時に `cancel()` が呼ばれず worker が継続実行しうるため、`_register_async_failure` との状態競合（timeout 扱いと後続 completed 書き込みの競合）リスクを特定。  
5. `ztb/utils/run_metadata.py` は package hash 取得時に `site-packages` を再帰走査し `.py/.pyc/.pyo` 全読込を行うため、起動時メタデータ収集の高コスト要因であることを確認。  
6. 同ファイルの git 情報取得は `subprocess.run` を5回直列実行しており、同種コマンドの重複呼び出しを統合できる余地を確認。  
7. `ztb/training/core/config_builder.py` は `UnifiedConfig = Dict[str, Any]` を含む `Any` 流出点で、`get_config_value` の戻り値 `Any` が下流型を弱める主因であることを確認。  
8. `any_inventory` の上位 debt を再確認し、次の高効果対象を `sac_algorithm.py(19)` / `reward_function_optimizer.py(19)` / `checkpoint_manager.py(18)` に設定。  
9. 回帰確認: `py_compile`（`ztb/training/utils`, `ztb/experiments`, `ztb/training/config` の33ファイル）を実施し、失敗は `sac_utils.py` のみ。  
10. 在庫確認: repo 全体 `any_type_debt_tokens=2,494`（Step69から変化なし）。本Stepは探索のみでコード変更なし。  

### Step71: 既存 `safety` helper への統合（util 抽出の水平展開）

1. `ztb/utils/safety.py` の `ensure_dict` / `safe_to_float` を canonical helper とし、重複していた局所変換ロジックの統合先を明確化。  
2. `ztb/training/run_optimization.py` の `_as_object_map` / `_as_float` 実装本体を `ensure_dict` / `safe_to_float` 委譲へ置換。  
3. `ztb/experiments/job_manager.py` の `_as_object_map` / `_as_float` 実装本体を同 helper 委譲へ置換。  
4. `ztb/experiments/run_sac_experiments.py` の `_as_object_map` / `_as_float` 実装本体を同 helper 委譲へ置換。  
5. `ztb/utils/run_manifest.py` の `_as_object_map` 実装本体を `ensure_dict` 委譲へ置換。  
6. これにより `dict/float` coercion の実装差異を減らし、今後の挙動修正時に `safety` 側だけ更新すれば水平展開できる構造へ整理。  
7. 回帰確認: `py_compile`（`run_optimization.py` / `job_manager.py` / `run_sac_experiments.py` / `run_manifest.py`）通過。  
8. 在庫確認: 対象4ファイルはいずれも `any_type_debt_tokens=0` を維持。repo 全体は `2,501`（`scanned_files: 1,289`）で、同時進行差分による母数変動を確認。  

---

## 5. 進捗サマリー

| Step | 対象 | any_type_debt_tokens |
|---|---|---:|
| Step0 | repo全体 baseline | 4,436 |
| Step2時点 | repo全体 | 4,414 |
| Step3時点 | repo全体 | 4,404 |
| Step4時点 | repo全体 | 4,397 |
| Step5時点 | repo全体 | 4,356 |
| Step8時点 | repo全体 | 4,281 |
| Step9時点 | repo全体 | 4,235 |
| Step10時点 | repo全体 | 4,181 |
| Step11時点 | repo全体 | 4,023 |
| Step12時点 | repo全体 | 3,988 |
| Step13時点 | repo全体 | 3,941 |
| Step14時点 | repo全体 | 3,823 |
| Step15時点 | repo全体 | 3,803 |
| Step16時点 | repo全体 | 3,771 |
| Step17時点 | repo全体 | 3,745 |
| Step18時点 | repo全体 | 3,745 |
| Step19時点 | repo全体 | 3,574 |
| Step20時点 | repo全体 | 3,496 |
| Step21時点 | repo全体 | 3,470 |
| Step22時点 | repo全体 | 3,446 |
| Step23時点 | repo全体 | 3,422 |
| Step24時点 | repo全体 | 3,395 |
| Step25時点 | repo全体 | 3,370 |
| Step26時点 | repo全体 | 3,350 |
| Step27時点 | repo全体 | 3,327 |
| Step28時点 | repo全体 | 3,327 |
| Step29時点 | repo全体 | 3,304 |
| Step30時点 | repo全体 | 3,281 |
| Step31時点 | repo全体 | 3,258 |
| Step32時点 | repo全体 | 3,240 |
| Step33時点 | repo全体 | 3,218 |
| Step34時点 | repo全体 | 3,209 |
| Step35時点 | repo全体 | 3,098 |
| Step36時点 | repo全体 | 3,077 |
| Step37時点 | repo全体 | 3,061 |
| Step38時点 | repo全体 | 3,024 |
| Step39時点 | repo全体 | 3,017 |
| Step40時点 | repo全体 | 3,012 |
| Step41時点 | repo全体 | 3,004 |
| Step42時点 | repo全体 | 3,000 |
| Step43時点 | repo全体 | 2,980 |
| Step44時点 | repo全体 | 2,973 |
| Step45時点 | repo全体 | 2,953 |
| Step46時点 | repo全体 | 2,927 |
| Step47時点 | repo全体 | 2,917 |
| Step48時点 | repo全体 | 2,904 |
| Step49時点 | repo全体 | 2,885 |
| Step50時点 | repo全体 | 2,873 |
| Step51時点 | repo全体 | 2,873 |
| Step52時点 | repo全体 | 2,873 |
| Step53時点 | repo全体 | 2,858 |
| Step54時点 | repo全体 | 2,858 |
| Step55時点 | repo全体 | 2,846 |
| Step56時点 | repo全体 | 2,817 |
| Step57時点 | repo全体 | 2,786 |
| Step58時点 | repo全体 | 2,747 |
| Step59時点 | repo全体 | 2,728 |
| Step60時点 | repo全体 | 2,687 |
| Step61時点 | repo全体 | 2,664 |
| Step62時点 | repo全体 | 2,664 |
| Step63時点 | repo全体 | 2,620 |
| Step64時点 | repo全体 | 2,610 |
| Step65時点 | repo全体 | 2,571 |
| Step66時点 | repo全体 | 2,537 |
| Step67時点 | repo全体 | 2,502 |
| Step68時点 | repo全体 | 2,516 |
| Step69時点 | repo全体 | 2,494 |
| Step70時点 | repo全体 | 2,494 |
| Step71時点 | repo全体 | 2,501 |
| Step72時点 | repo全体 | 2,505 |
| Step73時点 | repo全体 | 2,505 |
| Step74時点 | repo全体 | 2,488 |
| Step75時点 | repo全体 | 2,465 |
| Step4時点 | `scripts/v460` | **0** |
| Step5時点 | `ztb/evaluation/unified_evaluation.py` | **0** |
| Step8時点 | `ztb/metrics/metrics.py` | **0** |
| Step9時点 | `ztb/trading/comprehensive_backtest.py` | **0** |
| Step10時点 | `ztb/trading/real_data_validation.py` | **0** |
| Step11時点 | `ztb/training/unified_optimizer.py` | **0** |
| Step11時点 | `ztb/analysis/common/types.py` | **0** |
| Step12時点 | `ztb/types/common.py` | **0** |
| Step13時点 | `ztb/training/unified_trainer/reporting.py` | **0** |
| Step14時点 | `ztb/optimization/model_compression.py` | **0** |
| Step14時点 | `ztb/analysis/core/analyzer.py` | **0** |
| Step14時点 | `ztb/utils/checkpoint.py` | **0** |
| Step15時点 | `ztb/training/callbacks/shared/base/learning_callback.py` | **0** |
| Step16時点 | `ztb/training/callbacks/supervised/supervised_callbacks.py` | **0** |
| Step17時点 | `ztb/training/callbacks/reinforcement/sac/sac_callbacks.py` | **0** |
| Step19時点 | `ztb/analysis/v4xx_unified_analyzer.py` | **0** |
| Step19時点 | `ztb/utils/type_guards.py` | **0** |
| Step19時点 | `ztb/analysis/regime/v444_regime_analyzer.py` | **0** |
| Step19時点 | `ztb/trading/real_data_validator.py` | **0** |
| Step20時点 | `ztb/config/loaders/priority_loader.py` | **0** |
| Step20時点 | `ztb/trading/strategies/action_signal_guide/interfaces/adaptation_interfaces.py` | **0** |
| Step20時点 | `ztb/trading/strategies/action_signal_guide/interfaces/ml_interfaces.py` | **0** |
| Step20時点 | `ztb/trading/strategies/action_signal_guide/interfaces/portfolio_interfaces.py` | **0** |
| Step21時点 | `ztb/experiments/job_manager.py` | **0** |
| Step22時点 | `ztb/trading/performance_optimizer.py` | **0** |
| Step23時点 | `ztb/trading/end_to_end_validator.py` | **0** |
| Step24時点 | `ztb/training/unified_trainer/trainer.py` | **0** |
| Step25時点 | `ztb/training/unified_trainer/algorithms/sac_trainer.py` | **0** |
| Step25時点 | `ztb/training/system_optimizer.py` | **0** |
| Step26時点 | `ztb/training/unified_trainer/ensemble_system.py` | **0** |
| Step27時点 | `ztb/analysis/promotion.py` | **0** |
| Step28時点 | `ztb/analysis/promotion.py` | **0** |
| Step29時点 | `ztb/trading/environment/heavy_env/core.py` | **0** |
| Step30時点 | `ztb/utils/env_metrics.py` | **0** |
| Step31時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/candlestick_patterns.py` | **0** |
| Step32時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/wave_counting.py` | **0** |
| Step33時点 | `ztb/trading/strategies/action_signal_guide/realtime_adaptation/streaming_processor.py` | **0** |
| Step34時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/fibonacci_patterns.py` | **0** |
| Step35時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/harmonic_patterns.py` | **0** |
| Step36時点 | `ztb/trading/strategies/action_signal_guide/ml_integration/pattern_optimizer.py` | **0** |
| Step37時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/base.py` | **0** |
| Step38時点 | `ztb/trading/strategies/action_signal_guide/components/signal_quality_filter.py` | **0** |
| Step38時点 | `ztb/trading/strategies/action_signal_guide/action_signal_guide.py` | **0** |
| Step39時点 | `ztb/trading/strategies/action_signal_guide/components/signal_generator.py` | **0** |
| Step40時点 | `ztb/trading/strategies/action_signal_guide/components/market_regime.py` | **0** |
| Step41時点 | `ztb/trading/strategies/action_signal_guide/components/dynamic_adapter.py` | **0** |
| Step41時点 | `ztb/analysis/regime/market_regime_types.py` | **0** |
| Step42時点 | `ztb/trading/environment/components/threshold_manager.py` | **0** |
| Step43時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/dow_theory.py` | **0** |
| Step43時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/granville_law.py` | **0** |
| Step44時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/gann_analysis.py` | **0** |
| Step45時点 | `ztb/trading/strategies/action_signal_guide/components/validation.py` | **0** |
| Step45時点 | `ztb/trading/strategies/action_signal_guide/components/pattern_statistics.py` | **0** |
| Step46時点 | `ztb/trading/strategies/action_signal_guide/components/sac_integration.py` | **0** |
| Step46時点 | `ztb/trading/strategies/action_signal_guide/components/plugin_manager.py` | **0** |
| Step46時点 | `ztb/trading/strategies/action_signal_guide/components/cache_manager.py` | **0** |
| Step47時点 | `ztb/trading/strategies/action_signal_guide/components/advanced_signal_aggregator.py` | **0** |
| Step47時点 | `ztb/trading/strategies/action_signal_guide/components/performance_tracker.py` | **0** |
| Step47時点 | `ztb/trading/strategies/action_signal_guide/components/adaptive_pattern_selector.py` | **0** |
| Step48時点 | `ztb/trading/strategies/action_signal_guide/analysis/signal_performance_analyzer.py` | **0** |
| Step49時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/oscillator_patterns.py` | **0** |
| Step49時点 | `ztb/trading/strategies/action_signal_guide/portfolio_optimization/strategy_allocator.py` | **0** |
| Step50時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/rsi.py` | **0** |
| Step50時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/macd.py` | **0** |
| Step50時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/atr.py` | **0** |
| Step53時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/trend_analyzer.py` | **0** |
| Step53時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/heikin_ashi.py` | **0** |
| Step53時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/volume_patterns.py` | **0** |
| Step53時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/ichimoku.py` | **0** |
| Step53時点 | `ztb/trading/strategies/action_signal_guide/pattern_recognition/bollinger_patterns.py` | **0** |
| Step53時点 | `ztb/trading/strategies/action_signal_guide/recognizer_factory.py` | **0** |
| Step55時点 | `ztb/trading/strategies/action_signal_guide/types.py` | **0** |
| Step55時点 | `ztb/trading/strategies/action_signal_guide/config/asg_portfolio_config.py` | **0** |
| Step55時点 | `ztb/trading/strategies/action_signal_guide/config/asg_adaptation_config.py` | **0** |
| Step55時点 | `ztb/trading/strategies/action_signal_guide/config/asg_ml_config.py` | **0** |
| Step56時点 | `ztb/trading/signal/ensemble_signal_generator.py` | **0** |
| Step57時点 | `ztb/trading/signal/quality/indicators/base.py` | **0** |
| Step57時点 | `ztb/trading/signal/quality/indicators/rsi.py` | **0** |
| Step57時点 | `ztb/trading/signal/quality/indicators/macd.py` | **0** |
| Step58時点 | `ztb/trading/end_to_end_validation.py` | **0** |
| Step58時点 | `ztb/trading/e2e_test_framework.py` | **0** |
| Step59時点 | `ztb/trading/backtest/integrated_backtest_runner.py` | **0** |
| Step60時点 | `ztb/data/data_validation.py` | **0** |
| Step60時点 | `ztb/utils/config.py` | **0** |
| Step61時点 | `ztb/io/json_io.py` | **0** |
| Step61時点 | `ztb/utils/safety.py` | **0** |
| Step61時点 | `ztb/training/reward_function_optimizer/candidate_evaluator.py` | **0** |
| Step61時点 | `ztb/training/reward_function_optimizer/mtf_optimizer.py` | **0** |
| Step61時点 | `ztb/training/reward_function_optimizer/mtf_scheduler.py` | **0** |
| Step61時点 | `ztb/training/reward_function_optimizer/config_manager.py` | **0** |
| Step62時点 | `ztb/trading/production/state_persistence.py` | **0** |
| Step63時点 | `ztb/io/state_persistence.py` | **0** |
| Step63時点 | `ztb/trading/signal/types.py` | **0** |
| Step64時点 | `ztb/training/components/regime_adaptive_trainer.py` | **0** |
| Step64時点 | `ztb/trading/cost/venue_transaction_cost_manager.py` | **0** |
| Step64時点 | `ztb/ops/health/performance_monitor.py` | **0** |
| Step64時点 | `ztb/features/feature_set_config.py` | **0** |
| Step65時点 | `ztb/ops/costs/budget_rollup.py` | **0** |
| Step65時点 | `ztb/features/generators/multi_timeframe/config.py` | **0** |
| Step65時点 | `ztb/utils/run_manifest.py` | **0** |
| Step65時点 | `ztb/analysis/common/data_loaders.py` | **0** |
| Step65時点 | `ztb/experiments/run_sac_experiments.py` | **0** |
| Step66時点 | `ztb/training/callbacks/multi_task/multi_task_callbacks.py` | **0** |
| Step66時点 | `ztb/training/callbacks/meta/meta_callbacks.py` | **0** |
| Step67時点 | `ztb/training/callbacks/distributed/worker.py` | **0** |
| Step67時点 | `ztb/training/callbacks/distributed/integration.py` | **0** |
| Step67時点 | `ztb/training/callbacks/performance/memory_optimizer.py` | **0** |
| Step67時点 | `ztb/training/callbacks/distributed/threading_mixin.py` | **0** |
| Step68時点 | `ztb/training/callbacks/shared/utils/value_utils.py` | **0** |
| Step68時点 | `ztb/training/callbacks/supervised/supervised_callbacks.py` | **0** |
| Step68時点 | `ztb/training/callbacks/reinforcement/sac/sac_callbacks.py` | **0** |
| Step68時点 | `ztb/training/callbacks/transfer/transfer_callbacks.py` | **0** |
| Step68時点 | `ztb/training/callbacks/unsupervised/unsupervised_callbacks.py` | **0** |
| Step68時点 | `ztb/training/callbacks/meta/meta_callbacks.py` | **0** |
| Step68時点 | `ztb/training/callbacks/multi_task/multi_task_callbacks.py` | **0** |
| Step69時点 | `ztb/experiments/base.py` | **0** |
| Step69時点 | `ztb/utils/results_utils.py` | **0** |
| Step69時点 | `ztb/training/run_optimization.py` | **0** |
| Step71時点 | `ztb/experiments/job_manager.py` | **0** |
| Step71時点 | `ztb/experiments/run_sac_experiments.py` | **0** |
| Step71時点 | `ztb/utils/run_manifest.py` | **0** |
| Step72時点 | `ztb/training/utils/sac_utils.py` | **0** |
| Step72時点 | `ztb/utils/run_metadata.py` | **0** |
| Step72時点 | `ztb/utils/git_utils.py` | **0** |
| Step72時点 | `ztb/utils/run_manifest.py` | **0** |
| Step73時点 | `ztb/experiments/job_manager.py` | **0** |
| Step73時点 | `tests/unit/experiments/test_job_manager.py` | **0** |
| Step74時点 | `ztb/trading/strategies/action_signal_guide/realtime_adaptation/streaming_processor.py` | **0** |
| Step74時点 | `ztb/training/core/config_builder.py` | **0** |
| Step74時点 | `ztb/training/checkpoint/checkpoint_manager.py` | **0** |
| Step75時点 | `scripts/v460/ml/retrain_scheduler.py` | **0** |

---

## Step72 追補: SAC utility 復旧 + metadata 収集軽量化 + git helper 統合 (2026-02-22)

### 1) `ztb/training/utils/sac_utils.py` の機能復旧と安全化

- 構文/初期化不整合を解消し、CLI サブコマンド (`check-config`, `validate-data`, `quality-checks`, `clean`, `fix-common`) を安定化。
- `clean` に `max_scan_seconds` 上限制御を追加し、大規模リポジトリでも実行時間を制御可能化。
- `fix-common` に `max_files` 上限制御を追加し、全走査の過負荷を抑止。
- `check-config` に `max_details` を追加し、巨大 JSON 出力による I/O コストと可読性低下を抑制。

### 2) `ztb/utils/run_metadata.py` の性能改善・型契約整理

- package hash を **opt-in** (`--include-package-hashes`) 化し、通常実行のメタデータ収集時間を削減。
- package hash は distribution location 全再帰を廃止し、対象 package path + file stat ベースへ変更（高コスト経路を除去）。
- `save/load` を `write_json` / `read_json_object` へ統一し、JSON I/O 契約を明確化。
- direct script 実行時の import 問題を回避する `sys.path` fallback を追加。

### 3) git subprocess 重複統合

- `ztb/utils/git_utils.py` を新設し、git 情報取得処理を共通化（LFS 依存環境でも失敗しにくい実装）。
- `ztb/utils/run_manifest.py` の `get_git_sha` / `get_git_dirty_status` を同 helper 委譲へ置換。
- `run_metadata` も同 helper を利用し、git 情報取得の分散実装を縮退。

### 4) 検証

- `py_compile`:
  - `ztb/training/utils/sac_utils.py`
  - `ztb/utils/run_metadata.py`
  - `ztb/utils/run_manifest.py`
  - `ztb/utils/git_utils.py`
- 実行確認:
  - `python3 ztb/training/utils/sac_utils.py --help`
  - `python3 ztb/training/utils/sac_utils.py check-config --max-details 20`
  - `python3 ztb/training/utils/sac_utils.py fix-common --max-files 50`
  - `python3 ztb/utils/run_metadata.py --output /tmp/run_metadata_test.json`
  - `python3 ztb/utils/run_metadata.py --output /tmp/run_metadata_hash_test.json --include-package-hashes --package-hash-file-limit 10`
- `any_inventory`（Step72）:
  - repo 全体 `any_type_debt_tokens=2,505`（`scanned_files=1,292`）
  - 変更対象4ファイルは `any_type_debt_tokens=0` を維持

## Step73 追補: job_manager 競合対策（timeout/cancel）+ 並列実行安定化 (2026-02-22)

### 1) 競合原因の整理

- 旧実装は worker 側 (`execute_job`) が output/manifest/state を直接更新していたため、
  親側で timeout 判定した後に worker 完了結果で上書きされる競合が発生し得た。
- `ProcessPoolExecutor + bound method` 依存により、環境依存の pickling 失敗リスクも残存。

### 2) 修正内容

- worker 実行を副作用なしの `_execute_training_job()` に分離し、**永続化は親プロセスの `_finalize_job()` に一本化**。
- `run_all_jobs()` は `parallel_backend` を導入し、既定を `thread` へ変更（pickling 依存を低減）。
- parallel scheduler を `wait(FIRST_COMPLETED)` + timeout 監視ループへ再構成し、
  timeout 到達 job は親側で `timeout` 確定・永続化し、以降の遅延完了結果は無視。
- `executor.shutdown(wait=False, cancel_futures=True)` で timeout job 待機による scheduler 停滞を回避。
- polling 間隔は `timeout_seconds` に応じた動的値へ調整（小さい timeout でも即応）。
- `job_manager` の code hash 取得は `git_utils.get_git_sha` に統合し、git subprocess 重複を削減。

### 3) 検証

- `py_compile`:
  - `ztb/experiments/job_manager.py`
  - `tests/unit/experiments/test_job_manager.py`
- 追加テスト:
  - `tests/unit/experiments/test_job_manager.py`
    - default thread backend で local callable が実行可能なこと
    - timeout 後に遅延完了が発生しても `status=timeout` が上書きされないこと
- `pytest` は実行環境に未導入のため未実施（`pytest: command not found`）。
- `any_inventory`（Step73）:
  - repo 全体 `any_type_debt_tokens=2,505`（`scanned_files=1,294`）
  - `job_manager.py` / 追加テストとも `Any=0`

## Step74 追補: streaming helper 統合 + config/checkpoint 型固定 + replay metadata 正常化 (2026-02-22)

### 1) helper 横展開（重複削減）

- `ztb/trading/strategies/action_signal_guide/realtime_adaptation/streaming_processor.py` の
  `_as_object_map/_as_float` を `safety.ensure_dict/safe_to_float` 委譲へ統一。
- `ztb/analysis/v4xx_unified_analyzer.py` / `ztb/analysis/promotion.py` も同方針で実施予定だが、
  現在は git-lfs pointer 管理ファイルのため差分が全体置換化する状態を確認し、
  先に他ファイルの安全な改善を優先。

### 2) `config_builder` の型固定

- `ztb/training/core/config_builder.py` から `Any` を撤去し、
  `ConfigMap` / generic default (`TypeVar`) を使った `get_config_value()` 契約へ変更。
- `get_config_value()` で section 取得時に `ensure_dict()` 正規化を挟み、
  非dict値混入時の `safe_config_get` 呼び出し不整合リスクを解消。
- `UnifiedConfig = dict[str, object]` へ更新し、`Any` 流出起点を縮退。

### 3) `checkpoint_manager` の payload/metadata 契約整理

- `ztb/training/checkpoint/checkpoint_manager.py` に
  `CheckpointPayload` / `CheckpointMetadata` / `RNGStatePayload` /
  `CheckpointValidationResult` を導入し、`Any` ベース注釈を撤去。
- `BaseAlgorithm` の runtime fallback を `Protocol` 化し、`BaseAlgorithm = Any` alias を除去。
- `_build_payload()` を型付き payload 構築へ更新し、`policy_state` 取得失敗時の防御を追加。

### 4) 追加の機能改善

- `ztb/trading/live/simulation/paper_trader.py` の replay metadata を
  dummy JSON から `RunMetadata.capture_all_metadata()` + `save_to_file()` に置換し、
  監査/再現用メタデータの実効性を回復。

### 5) 検証

- `py_compile`:
  - `ztb/trading/strategies/action_signal_guide/realtime_adaptation/streaming_processor.py`
  - `ztb/training/core/config_builder.py`
  - `ztb/training/checkpoint/checkpoint_manager.py`
  - `ztb/trading/live/simulation/paper_trader.py`
- `any_inventory`（Step74）:
  - repo 全体 `any_type_debt_tokens=2,488`（`scanned_files=1,300`）
  - `streaming_processor.py` / `config_builder.py` / `checkpoint_manager.py` は `Any=0`

## Step75 追補: retrain_scheduler Any 撤去 + WF評価高速化 + LFSガード明示化 (2026-02-22)

### 1) `retrain_scheduler.py` の Any 削減・型契約整理

- `scripts/v460/ml/retrain_scheduler.py` の `Any` を全撤去し、`ConfigMap` / `object` ベースへ移行。
- config 取得の主要分岐で `safe_to_int` / `safe_to_float` / `safe_to_bool` を適用し、
  YAML 値型ゆらぎ時の実行時不整合リスクを低減。
- `_safe_import_ztb_module()` の spec/loader 検証を追加し、import spec 失敗時の曖昧な例外を解消。

### 2) 重複削減・性能改善

- WF評価の PnL 集計を `_extract_numeric_column()` / `_compute_skip_metrics()` に統合し、
  single/multi window の重複ロジックを削減。
- `multi-window` 側で PnL 列をウィンドウ外で一括抽出して再利用し、
  反復 `DataFrame.loc` コストを削減。
- `SimpleImputer`/`StandardScaler` 後の中間 `DataFrame` 再構築を削減し、配列ベースで学習。
- `X_val` が小さい/空のウィンドウで不要 transform を回避し、early-stopping 前処理の失敗余地を低減。

### 3) JSONL I/O 統合と運用安定化

- `_append_jsonl_record()` を導入し、history 追記（scheduler/side/once）を単一実装へ統合。
- 追記処理の重複を減らし、将来的な監査フォーマット変更点を1箇所に集約可能化。

### 4) LFS汚染回避の調整

- `.gitattributes` 末尾に `*.py` / `*.pyi` の明示 override を追加し、
  source code が LFS 管理へ再侵食するリスクを低減。

### 5) 検証

- `py_compile`:
  - `scripts/v460/ml/retrain_scheduler.py`
- `any_inventory`:
  - `--roots scripts/v460/ml`: `scripts/v460/ml/retrain_scheduler.py` は `any_type_debt_tokens=0`
  - repo 全体: `any_type_debt_tokens=2,465`（`scanned_files=1,302`）

## Step76 追補: collect 完了 + pytest 設定復旧 + 互換 shim Any 全撤去 (2026-02-23)

### 1) テスト安定化の未完了項目を解消（153# 追補）

- `pytest --co -q` を再実行し、`5402 tests collected / collect errors=0` を確認。
- warning 根因だった pytest 設定未読込を修正:
  - `pytest.ini` セクションを `[tool:pytest]` から `[pytest]` に修正。
  - `pytest.ini` の `-n auto` を除去し、`pytest-xdist` 未導入環境でも実行可能化。
  - `pyproject.toml` 側にも `unit/integration/slow/performance` marker を明示追加。
- 再計測: `tests/unit/v460` 主要5ファイルで `170 passed, 0 warnings`（`-W all`）。

### 2) 互換 shim 群の型安全化（Any 後退を回収）

- 追加互換モジュール群の `Any` を `object` / `Mapping` ベースへ統一:
  - `stable_baselines3/*`, `sb3_contrib/*`, `prometheus_client/__init__.py`
  - `ztb/utils/v4xx_config_converter.py`
  - `ztb/evaluation/logging.py`
  - `ztb/evaluation/promotion.py`
- `stable_baselines3` shim では `BaseAlgorithm` 継承を導入し、
  学習/保存/推論のダミー実装重複を削減（継承ベースの重複排除）。

### 3) 検証

- `py_compile`:
  - `stable_baselines3/__init__.py`
  - `stable_baselines3/common/base_class.py`
  - `stable_baselines3/common/callbacks.py`
  - `stable_baselines3/common/monitor.py`
  - `stable_baselines3/common/evaluation.py`
  - `stable_baselines3/common/vec_env.py`
  - `stable_baselines3/common/type_aliases.py`
  - `stable_baselines3/common/torch_layers.py`
  - `sb3_contrib/common/wrappers.py`
  - `prometheus_client/__init__.py`
  - `ztb/utils/v4xx_config_converter.py`
  - `ztb/evaluation/logging.py`
  - `ztb/evaluation/promotion.py`
- `any_inventory`:
  - `--roots stable_baselines3 sb3_contrib prometheus_client ztb/evaluation/logging.py ztb/utils/v4xx_config_converter.py ztb/evaluation/promotion.py`
  - 結果: `scanned_files=22`, `any_type_debt_tokens=0`

## Step77 追補: regime A/B ハーネスの型安全化 + 入力正規化 (2026-02-23)

### 1) `compare_regime_ab.py` の `Any` 全撤去

- `scripts/v460/analysis/compare_regime_ab.py` の `Any` 注釈を `object` ベースへ置換。
- `FillRecord = dict[str, object]` を導入し、`_simulate()` / `_save_summary()` で payload 契約を明示化。

### 2) 不具合余地の低減

- `post_fill_30s_pnl` を `float | None` へ正規化して `SimRecord.pnl_30s` へ格納するよう修正。
  - 非数値混入時にそのまま伝播して後段集計で型不整合になるリスクを抑制。
- `recorded_pnl` 生成時に `regime` を `str` 限定で扱うようにし、異常値混入時の辞書キー汚染を防止。

### 3) 検証

- `py_compile`:
  - `scripts/v460/analysis/compare_regime_ab.py`
- テスト:
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_152_parallel_tasks.py::TestCompareRegimeAB -q --override-ini="addopts=" -W all`
  - 結果: `5 passed`
- `any_inventory`:
  - `--roots scripts/v460/analysis/compare_regime_ab.py`
  - 結果: `any_type_debt_tokens=0`

## Step78 追補: テスト/Git 重量対策（ローカル高速化）(2026-02-23)

### 1) pytest の重い既定動作を軽量化

- `pytest.ini` の既定 `addopts` から次を削減:
  - `--verbose`（大量出力による I/O 負荷）
  - `--cov-report=html:htmlcov`（毎回 HTML 生成）
- `--cov` / `--cov-fail-under` は維持し、品質ゲート要件は保持。

### 2) 高速テストランナー追加

- 追加: `scripts/testing/run_fast_pytest.py`
  - `--override-ini=addopts=` で重い既定 addopts をバイパス。
  - `scope` (`v460` / `unit` / `all`) と `collect-only` をサポート。
  - `xdist` 導入環境では自動で `-n auto` を付与。
  - `collect-only` は `-qq` でノード一覧の出力量を圧縮。

### 3) Git の重量対策（設定 + helper）

- 追加: `scripts/maintenance/optimize_git_local.py`
  - local git config に高速化設定を適用:
    - `feature.manyFiles=true`
    - `core.untrackedCache=true`
    - `core.preloadIndex=true`
    - `index.threads=0`
    - `status.aheadBehind=false`
- `ztb/utils/git_utils.py` を高速化:
  - `git --no-optional-locks` + `GIT_OPTIONAL_LOCKS=0`
  - status 系 API の既定を **tracked-only** (`--untracked-files=no`) へ変更
  - status 行数を上限付きで返すように変更（巨大差分時のコスト抑制）
- `run_metadata.py` / `run_manifest.py` は tracked-only dirty 判定へ追従。

### 4) 検証

- `py_compile`:
  - `ztb/utils/git_utils.py`
  - `ztb/utils/run_metadata.py`
  - `ztb/utils/run_manifest.py`
  - `scripts/testing/run_fast_pytest.py`
  - `scripts/maintenance/optimize_git_local.py`
- テスト:
  - `.venv/Scripts/python.exe scripts/testing/run_fast_pytest.py tests/unit/v460/test_152_parallel_tasks.py`
  - 結果: `12 passed`
  - `.venv/Scripts/python.exe scripts/testing/run_fast_pytest.py --scope all --collect-only`
  - 結果: `5394 tests collected`
- `any_inventory`:
  - `--roots ztb/utils/git_utils.py ztb/utils/run_metadata.py ztb/utils/run_manifest.py scripts/testing/run_fast_pytest.py scripts/maintenance/optimize_git_local.py`
  - 結果: `any_type_debt_tokens=0`

## Step79 追補: metrics再現スクリプトの単一パス集計化 + 型安全化 (2026-02-23)

### 1) `reproduce_152_metrics.py` の `Any` 全撤去

- 対象: `scripts/v460/analysis/reproduce_152_metrics.py`
- `FillRecord` / `MetricsMap` alias を導入し、`Any` を `object` ベースへ置換。
- `_load_records()` は読み込んだ JSONL を dict のみに正規化して扱うよう変更。
- 既存実装を再利用:
  - `ztb.utils.safety.ensure_dict` / `safe_to_float` / `safe_to_int`
  - `ztb.io.json_io.write_json`
  に寄せ、同等ヘルパーのローカル重複を削減。

### 2) パフォーマンス・重複削減

- `_compute_metrics()` を多重ループ構造から **単一パス集計** へ再構成。
  - `regime_distribution` / `lot_distribution` / `run_ids`
  - `regime_pnl_30s` / `side_regime_pnl`
  - `hour_pnl` / `as_probability_distribution`
  を1回の走査で蓄積し、後段で summary 生成に統一。
- `filled` / `with_qty` の中間配列依存を削減し、メモリ使用を抑制。

### 3) 不具合余地の低減

- 数値項目は `_to_float()` に統一し、文字列・不正値混入時の `ValueError` 伝播を抑止。
- `_print_report()` を防御的な型変換（`_to_dict`, `_as_int`, `_as_float_or_zero`）で強化し、
  破損データ混入時の表示処理クラッシュを回避。

### 4) 検証

- `py_compile`:
  - `scripts/v460/analysis/reproduce_152_metrics.py`
- テスト:
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_152_parallel_tasks.py -q --override-ini="addopts="`
  - 結果: `12 passed`
- `any_inventory`:
  - `--roots scripts/v460/analysis/reproduce_152_metrics.py`
  - 結果: `any_type_debt_tokens=0`

## Step80: `compare_regime_ab.py` の既存実装再利用 + 型安全化 (2026-02-23)

### 1) 既存実装の再利用

- 対象: `scripts/v460/analysis/compare_regime_ab.py`
- 数値変換を `ztb.utils.safety.safe_to_float` ベースへ統一。
  - ローカル helper `_to_float_or_none()` を追加し、`bool`/非有限値を除外して `None` に正規化。
- summary 出力を `Path.write_text(json.dumps(...))` から `ztb.io.json_io.write_json` に置換。

### 2) 重複削減・不具合余地の解消

- `float(...)` 直接変換を排除し、時刻/価格/PNL の変換失敗で落ちる経路を防御化。
- `_evaluate_gates()` の `type: ignore[arg-type]` を除去し、`None` ガードで型整合を明示。
- 未使用 import (`RegimeResult`) を削除。
- `main()` 側の実質未使用な `recorded_pnl` 集計ループを削除（1 パス分の無駄走査を解消）。
- `_evaluate_gates()` は既存テスト互換のため第2引数を optional で維持。

### 3) パフォーマンス・保守性

- 変換処理を helper に集約し、前処理とシミュレーションの重複ロジックを縮退。
- JSON 出力経路を共通 I/O helper に統一し、他スクリプトとの実装差分を削減。

### 4) 検証

- `py_compile`:
  - `scripts/v460/analysis/compare_regime_ab.py`
- テスト:
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_152_parallel_tasks.py -q --override-ini="addopts="`
  - 結果: `12 passed`
- `any_inventory`:
  - `.venv/Scripts/python.exe scripts/quality/any_inventory.py --roots scripts/v460/analysis/compare_regime_ab.py`
  - 結果: `any_type_debt_tokens=0`

## Step81: Git フック修復の再現化 + `compare_regime_ab` 安定性改善 (2026-02-23)

### 1) Git 修復（再現可能化）

- 背景:
  - `.git/hooks/pre-commit` が CRLF と shell 非互換構文を含み、
    `fatal: cannot exec '.git/hooks/pre-commit': No such file or directory`
    を誘発する状態を確認。
- 対応:
  - `scripts/maintenance/optimize_git_local.py` に
    `--repair-pre-commit-hook` を追加。
  - 同オプションで `.git/hooks/pre-commit` を LF の portable launcher に再生成し、
    実行環境が合致しない場合は非ブロッキングで skip する挙動を標準化。

### 2) 機能改善（不具合予防 + 軽量化）

- 対象: `scripts/v460/analysis/compare_regime_ab.py`
- `0除算` 予防:
  - `_safe_pct()` を導入し、`_print_report()` の
    `total==0` / `filled==0` ケースでも例外なく出力継続。
- 前処理の軽量化:
  - 前処理で正規化した `timestamp/order_price` を
    `ParsedFillRecord` として保持し、後段ループで再変換を廃止。
  - 変換重複と dict 参照回数を削減。

### 3) テスト強化

- 対象: `tests/unit/v460/test_152_parallel_tasks.py`
- 追加:
  - `test_print_report_handles_zero_records`
  - `test_print_report_handles_zero_filled`
- 目的:
  - レコード0件・filled0件の境界条件で回帰しないことを固定化。

### 4) 検証

- `py_compile`:
  - `scripts/v460/analysis/compare_regime_ab.py`
  - `scripts/maintenance/optimize_git_local.py`
  - `tests/unit/v460/test_152_parallel_tasks.py`
- テスト:
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_152_parallel_tasks.py -q --override-ini="addopts="`
  - 結果: `14 passed`
- `any_inventory`:
  - `.venv/Scripts/python.exe scripts/quality/any_inventory.py --roots scripts/v460/analysis/compare_regime_ab.py scripts/maintenance/optimize_git_local.py`
  - 結果: `any_type_debt_tokens=0`

## Step82: 損益計算の共通化・効率化 + 継承整理 (2026-02-23)

### 1) 既存損益実装の再利用を強化

- 対象:
  - `ztb/metrics/fill_quality.py`
  - `scripts/v460/lib/results_analyzer.py`
- `fill_quality` に `PnlAccumulator` を追加し、有限値のみを集計する共通 PnL 集計器を導入。
- `results_analyzer.compute_event_contribution()` は
  `PnlAccumulator` を利用する形に統一し、局所 `sum/len` 実装を削減。

### 2) 計算効率改善（損益計算のホットパス）

- `compute_round_trip_metrics()` の FIFO キューを
  `list.pop(0)` から `collections.deque.popleft()` へ変更。
  - 先頭取り出しが O(n) → O(1) となり、大量 fill 時の計算負荷を削減。
- `compute_event_contribution()` を単一パス集計へ再構成。
  - 旧: filled/FFD/VG/SG で複数回フィルタ + 中間配列生成。
  - 新: 1 回走査でイベント別 accumulator に投入し、SG のみ閾値計算用に最小データ保持。

### 3) 継承整理（重複削減）

- `GroupedMetricsBase` を導入し、共通フィールド
  (`count`, `filled`, `pnl_mean_bps`, `as_ratio`) を基底化。
- `RegimeMetrics` / `HourlyMetrics` は `GroupedMetricsBase` を継承する形へ再整理。
- `compute_regime_metrics()` / `compute_hourly_metrics()` は
  `_summarize_filled_records()` 共通ヘルパーを利用し、重複ロジックを削減。

### 4) 不具合余地の低減

- `results_analyzer` の `type: ignore[arg-type/operator]` を除去し、
  `None` / 非有限値ガードで安全に集計。
- `round_trip` 計算で `fill_price` の `None` を明示ガードし、
  Optional 値演算の潜在不具合を抑止。

### 5) テスト

- 追加:
  - `tests/unit/v460/test_results_analyzer.py`
    - `test_compute_event_contribution_basic`
    - `test_compute_event_contribution_ignores_non_finite_values`
- 回帰:
  - `tests/unit/v460/test_fill_quality.py::Test051RoundTripMetrics`
  - `tests/unit/v460/test_fill_quality.py::Test051RegimeMetrics`
  - `tests/unit/v460/test_fill_quality.py::Test051HourlyMetrics`

### 6) 検証

- `py_compile`:
  - `ztb/metrics/fill_quality.py`
  - `scripts/v460/lib/results_analyzer.py`
  - `tests/unit/v460/test_results_analyzer.py`
- テスト:
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_results_analyzer.py tests/unit/v460/test_fill_quality.py::Test051RoundTripMetrics tests/unit/v460/test_fill_quality.py::Test051RegimeMetrics tests/unit/v460/test_fill_quality.py::Test051HourlyMetrics -q --override-ini="addopts="`
  - 結果: `11 passed`
- `any_inventory`:
  - `.venv/Scripts/python.exe scripts/quality/any_inventory.py --roots ztb/metrics/fill_quality.py scripts/v460/lib/results_analyzer.py tests/unit/v460/test_results_analyzer.py`
  - 結果: `any_type_debt_tokens=0`

## Step83: Oracle系損益計算の単一パス化 + 追加計算削減 (2026-02-23)

### 1) 既存損益実装の再利用

- 対象:
  - `scripts/v460/analysis/oracle_baseline.py`
  - `scripts/v460/lib/results_analyzer.py`
- `oracle_baseline` で `ztb.metrics.fill_quality.PnlAccumulator` を再利用。
  - Oracle集計を `PnlAccumulator` ベースに統一し、手書き `sum/len` を削減。

### 2) 計算効率改善

- `oracle_baseline`:
  - `_OracleAggregate` / `_aggregate_oracle()` / `_metrics_from_aggregate()` を導入。
  - `compute_oracle_metrics()` を単一パス集計に変更（`type: ignore` 除去）。
  - `run_oracle_baseline()` の lotシナリオは、同一データの再走査をやめて
    **全体集計を再利用**して算出。
  - side/regime 用のグループ化も1回の走査で構築。
- `results_analyzer`:
  - `compute_multi_track_analysis()` の trailing 算出を
    全件ソート (`sorted(..., reverse=True)[:N]`) から
    `heapq.nlargest(N, ...)` に変更。
    - `O(n log n)` → `O(n log N)`（N は trailing window）。

### 3) 不具合余地の低減

- `oracle_baseline` で `_to_finite_float()` を導入し、
  非数値/非有限値混入時に安全に除外。
- Oracle集計の `filled + pnl30 有効` 条件を明示化し、
  60s/120s も同一基準の安全変換で集計。

### 4) 検証

- `py_compile`:
  - `scripts/v460/analysis/oracle_baseline.py`
  - `scripts/v460/lib/results_analyzer.py`
  - `ztb/metrics/fill_quality.py`
- テスト:
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_retrain_hot_reload.py -k oracle -q --override-ini="addopts="`
  - 結果: `5 passed, 64 deselected`
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_results_analyzer.py tests/unit/v460/test_fill_quality.py::Test051RoundTripMetrics tests/unit/v460/test_fill_quality.py::Test051RegimeMetrics tests/unit/v460/test_fill_quality.py::Test051HourlyMetrics -q --override-ini="addopts="`
  - 結果: `11 passed`
- `any_inventory`:
  - `.venv/Scripts/python.exe scripts/quality/any_inventory.py --roots scripts/v460/analysis/oracle_baseline.py scripts/v460/lib/results_analyzer.py ztb/metrics/fill_quality.py`
  - 結果: `any_type_debt_tokens=0`

## Step84: `analyze_fill_records` の多重走査削減 (2026-02-23)

### 1) 目的

- 対象: `scripts/v460/analysis/analyze_fill_records.py`
- 既存実装は同一 `all_records` を用途ごとに繰り返し走査しており、
  データ量増加時に処理時間・中間配列コストが増大しやすい状態だった。

### 2) 改善内容

- 全面を **単一パス集計** ベースに再構成。
  - filled/skip
  - 30s/60s/120s PnL
  - side別 fill/AS/PnL
  - skip reason
  - regime別件数/PnL
  - UTC hour別 PnL/AS
  - latest run 指標
  - queue wait 指標
  を1回走査で蓄積。
- `_PnlStats` / `_RunStats` の小型集計構造を導入し、
  `sum/len` の重複ロジックを排除。
- `_to_finite_float()` と `_pct()` を追加し、
  非数値混入・ゼロ除算に対する安全性を強化。
- データ未存在時（fill_recordsファイルなし）も早期returnで明示出力。

### 3) 検証

- `py_compile`:
  - `scripts/v460/analysis/analyze_fill_records.py`
  - `scripts/v460/analysis/oracle_baseline.py`
  - `scripts/v460/lib/results_analyzer.py`
- テスト:
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_retrain_hot_reload.py -k oracle -q --override-ini="addopts="`
  - 結果: `5 passed, 64 deselected`
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_results_analyzer.py -q --override-ini="addopts="`
  - 結果: `2 passed`
- `any_inventory`:
  - `.venv/Scripts/python.exe scripts/quality/any_inventory.py --roots scripts/v460/analysis/analyze_fill_records.py scripts/v460/analysis/oracle_baseline.py scripts/v460/lib/results_analyzer.py`
  - 結果: `any_type_debt_tokens=0`

## Step85: `hindsight_filter` の filter 簡素化 + 契約化 + 補間ホットパス改善 (2026-02-23)

### 1) 対象

- `scripts/v460/analysis/hindsight_filter.py`

### 2) Any削減（型安全向上）

- `dict[str, Any]` / `dict[str, dict[str, Any]]` を排除。
- `RawRecord`（`Mapping[str, object]`）と `TypedDict` 契約を導入:
  - `AggregatePnlSummary`（基底）
  - `WaitBandSummary` / `RegimeSideSummary`（基底継承）
  - `SideReversalSummary`
  - `HourlySummary`
  - `SkipGate*` 系契約
  - `InterpolatedStats`
- `main()` 戻り値を `dict[str, object]` に変更。

### 3) filter/重複整理（保守性改善）

- cancel reason 分類を `_category_from_result()` + 定数テーブル化し、
  if/elif 連鎖を簡素化。
- 集計ロジックを共通化:
  - `_PnlAggregateBase`
  - `_SignedPnlAggregate`
  - `_SideReversalAggregate`
  - `_HourlyAggregate`
- `_analyze_side_reversal` / `_analyze_hourly` / `_analyze_wait_bands` /
  `_analyze_regime_side` / `_analyze_skip_gate_calibration` を
  単一走査寄りの実装へ整理（中間 list の重複生成を削減）。
- レポート表示の未使用プレースホルダコード（`pass` を含むブロック）を削除。

### 4) 不具合余地の低減

- 数値入力を `_to_float()` で統一変換し、
  文字列/非有限値混入時の計算不具合を抑止。
- 補間計算で同一 timestamp 混在時のゼロ除算リスクをガード
  （`interval <= 0` を除外）。
- `timeline` が空の場合は明示エラーで early-exit。

### 5) パフォーマンス改善

- `_analyze_records()` 内で `timeline` の timestamp 配列を前計算し、
  `_interpolate_price()` 呼び出しごとの再生成を除去。
- `_build_price_timeline()` で同一 timestamp を圧縮し、
  補間点数と検索ノイズを削減。

### 6) 検証

- `py_compile`:
  - `.venv/Scripts/python.exe -m py_compile scripts/v460/analysis/hindsight_filter.py`
- テスト:
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_155_hindsight_review.py -q --override-ini="addopts="`
  - 結果: `18 passed`
- `any_inventory`:
  - `.venv/Scripts/python.exe scripts/quality/any_inventory.py --roots scripts/v460/analysis/hindsight_filter.py`
  - 結果: `any_type_debt_tokens=0`

## Step86: v460 soft-offset 系の重複削減 + misconfig/ゼロ除算防止 (2026-02-28)

### 1) 対象

- `scripts/v460/lib/skip_gate_evaluator.py`
- `scripts/v460/lib/fill_cycle_executor.py`
- `tests/unit/v460/test_196_velocity_proportional_trending_soft.py`

### 2) 不具合封じ

- `SkipGateEvaluator._compute_velocity_offset_multiplier()` を追加し、
  velocity proportional 計算を一元化。
- `threshold_bps=0` のときは proportional 計算を使わず
  固定倍率へフォールバックし、ゼロ除算を防止。
- `base/max multiplier < 1.0` の誤設定時も
  1.0 未満に落とさず clamp し、soft mode が攻撃的価格になる事故を防止。

### 3) 重複削減 / 保守性改善

- `FillCycleExecutorMixin._apply_offset_multiplier()` を追加し、
  193/195/196 系の offset multiplier 適用パターンを共通化。
- `velocity_offset` と `trending_offset` の
  価格更新ロジック重複を削減。
- 追加で `ev_offset` も同ヘルパへ寄せ、
  方向反転（aggressive/conservative）の分岐を 1 箇所に統合。
- 1.0 以下の倍率は no-op 扱いに統一し、
  「soft mode なのに offset を狭める」逆方向挙動を抑止。

### 4) テスト強化

- `test_196_velocity_proportional_trending_soft.py` を
  source 文字列依存から一部脱却し、実際の helper 挙動を直接検証。
- 追加観点:
  - proportional boost の実計算
  - `threshold=0` fallback
  - sub-1 multiplier clamp
  - executor helper の no-op 保証

### 5) 検証

- `py_compile`:
  - `.venv/Scripts/python.exe -m py_compile scripts/v460/lib/skip_gate_evaluator.py scripts/v460/lib/fill_cycle_executor.py tests/unit/v460/test_196_velocity_proportional_trending_soft.py`
- テスト:
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_193_ev_offset.py tests/unit/v460/test_176_trending_offset_asymmetry.py tests/unit/v460/test_196_velocity_proportional_trending_soft.py tests/unit/v460/test_113_resilience.py -q --override-ini="addopts="`
  - 結果: `124 passed`
- `any_inventory`:
  - `.venv/Scripts/python.exe scripts/quality/any_inventory.py --roots scripts/v460/lib/skip_gate_evaluator.py scripts/v460/lib/fill_cycle_executor.py tests/unit/v460/test_196_velocity_proportional_trending_soft.py`
  - 結果: `any_type_debt_tokens=0`

## Step87: `maker_price` の offset倍率適用一元化 + FFD整合修正 (2026-02-28)

### 1) 対象

- `scripts/v460/lib/maker_price.py`
- `tests/unit/v460/test_168_low_vol_offset_boost.py`
- `tests/unit/v460/test_175_code_review_sweep2.py`

### 2) 重複削減 / 横展開

- `MakerPriceCalculator._scale_offset_ratio()` を追加し、
  offset ratio への倍率適用を共通化。
- 以下の手書き `*= ...` / `min(...)` / `max(...)` パターンを helper 経由へ整理:
  - trending boost / discount
  - high_vol
  - ranging
  - low_vol
  - unknown buy guard
  - spread adaptive (narrow / wide)
  - volatility guard
  - imbalance AS risk
  - inventory skewing
  - FFD boost

### 3) 不具合封じ

- 0 以下の multiplier が来ても no-op とし、ratio を壊さないようにした。
- FFD boost 後の `effective_offset_ratio` と実際の `offset` が
  clamp 時に食い違う問題を修正。
  - 旧: raw `boost_mult` で `offset` を先に拡大し、ratio だけ clamp
  - 新: clamp 後の ratio から `offset` を再計算し、価格と返却値を整合
- helper は「実際の適用倍率」を返すため、clamp 後の実効値ログにも流用可能。

### 4) テスト強化

- `test_168_low_vol_offset_boost.py`
  - helper の clamp/no-op テストを追加
  - FFD clamp 後に価格と ratio が一致する機能テストを追加
- `test_175_code_review_sweep2.py`
  - 旧 `min(...)` 文字列依存をやめ、helper 経由の clamp 実装を検証

### 5) 検証

- `py_compile`:
  - `.venv/Scripts/python.exe -m py_compile scripts/v460/lib/maker_price.py tests/unit/v460/test_168_low_vol_offset_boost.py tests/unit/v460/test_175_code_review_sweep2.py`
- テスト:
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_168_low_vol_offset_boost.py -q --override-ini="addopts="`
  - 結果: `13 passed`
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_175_code_review_sweep2.py::TestFFDBoostClamp -q --override-ini="addopts="`
  - 結果: `1 passed`
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_143_regime_utilization.py -q --override-ini="addopts="`
  - 結果: `58 passed`
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_fill_quality.py -k "volatility_guard" -q --override-ini="addopts="`
  - 結果: `3 passed, 186 deselected`
- `any_inventory`:
  - `.venv/Scripts/python.exe scripts/quality/any_inventory.py --roots scripts/v460/lib/maker_price.py tests/unit/v460/test_168_low_vol_offset_boost.py tests/unit/v460/test_175_code_review_sweep2.py`
  - 結果: `any_type_debt_tokens=0`

## Step88: `maker_price` の spread guard 共通化 (2026-02-28)

### 1) 対象

- `scripts/v460/lib/maker_price.py`
- `tests/unit/v460/test_168_low_vol_offset_boost.py`

### 2) 重複削減

- `MakerPriceCalculator._finalize_price_with_spread_guard()` を追加し、
  `compute()` 末尾の buy/sell 別 spread guard 分岐を一本化。
- cross 時の fallback 価格決定と
  `effective_offset_ratio=0.0` 化を helper に集約。

### 3) テスト強化

- spread guard helper の直接テストを追加:
  - buy cross → `best_bid` fallback
  - sell cross → `best_ask` fallback

### 4) 検証

- `py_compile`:
  - `.venv/Scripts/python.exe -m py_compile scripts/v460/lib/maker_price.py tests/unit/v460/test_168_low_vol_offset_boost.py`
- テスト:
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_168_low_vol_offset_boost.py -q --override-ini="addopts="`
  - 結果: `15 passed`
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_143_regime_utilization.py -q --override-ini="addopts="`
  - 結果: `58 passed`
  - `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_fill_quality.py -k "volatility_guard" -q --override-ini="addopts="`
  - 結果: `3 passed, 186 deselected`
- `any_inventory`:
  - `.venv/Scripts/python.exe scripts/quality/any_inventory.py --roots scripts/v460/lib/maker_price.py tests/unit/v460/test_168_low_vol_offset_boost.py`
  - 結果: `any_type_debt_tokens=0`

## Step89: `config_hot_reload` の対象漏れ補完 + `fast_fill_defense` の side解決共通化 (2026-02-28)

### 1) 対象

- `scripts/v460/lib/config_hot_reload.py`
- `scripts/v460/lib/fill_config.py`
- `scripts/v460/lib/fast_fill_defense.py`
- `tests/unit/v460/test_169_config_hot_reload.py`
- `tests/unit/v460/test_100_fast_fill_defense.py`
- `tests/unit/v460/test_196_velocity_proportional_trending_soft.py`

### 2) 不具合封じ / 重複削減

- `config_hot_reload` の `_HOT_RELOADABLE_FIELDS` に、
  後発追加された soft-guard 系の対象漏れを補完:
  - EV soft offset (`193#`)
  - velocity soft offset (`195/196#`)
  - trending sell soft offset (`196#`)
  - `balance_forced_apply_trending_offset` (`197#`)
- `FastFillDefense._resolve_side_value()` を追加し、
  threshold / boost / base offset の side別 fallback を一元化。
- `FastFillDefense._compute_capped_multiplier()` で、
  誤設定時でも multiplier が `1.0` 未満へ落ちないよう clamp。
- `min_offset_ratio<=0` でも 0 除算しないよう安全床を導入。
- `FillTestConfig.trending_sell_offset_boost_factor` の dataclass 既定値を
  live YAML (`2.0`) と一致させ、デフォルト構築時だけ `3.0` になる不整合を解消。
- `FillTestConfig.velocity_offset_boost_factor` は
  live YAML (`1.5`) と整合していることをテストで固定。
- `FillTestConfig.balance_forced_apply_trending_offset` を
  live YAML (`True`) と一致させ、デフォルト構築時との乖離を解消。

### 3) テスト強化

- `test_169_config_hot_reload.py`
  - 後発 soft-guard 項目が hot-reload 対象から漏れていないことを固定。
- `test_100_fast_fill_defense.py`
  - `boost<1.0` の誤設定で防御が弱まらないことを追加検証。
  - `min_offset_ratio<=0` でも cap 計算が安全に成立することを追加検証。
- `test_196_velocity_proportional_trending_soft.py`
  - `balance_forced` は live YAML 既定で offset が乗ることを固定。
  - `balance_forced_apply_trending_offset=False` のときだけ bypass になることを追加検証。
- `temp_yaml` fixture の戻り値を `Iterator[Path]` にして `Any` を削減。

## Step90: `stopgap_health` の型固定 + JSONL fallback 防御強化 (2026-02-28)

### 1) 対象

- `scripts/v460/lib/stopgap_health.py`
- `tests/unit/v460/test_stopgap_health.py`

### 2) Any削減 / 不具合封じ / 重複削減

- `stopgap_health` のレコード入力型を `JSONObject` ベースに固定し、
  `StopgapMetrics` / `StopgapCriteria` / 各 report row を `TypedDict` 化。
- `DailyHealthReport` の list payload を `TypedDict` 契約へ寄せ、
  `Any` 注釈を削減。
- JSONL fallback 読み込み時に object 以外 (`[]`, 数値, 文字列) を無視し、
  後続の `.get()` で壊れる不正行を遮断。
- `_collect_finite_values()` を追加し、
  PnL 抽出処理の重複を `2-A` / `2-D` / model_used 集計で共通化。
- `unknown_regime_count` を report serialize 時にも出力し、
  既に集計している指標が落ちていた不整合を解消。

### 3) テスト強化

- fallback 読み込みで非 object 行を無視することを追加検証。
- `unknown_regime_count` が report に含まれることを追加検証。

## Step91: `ab_judgment` の payload 契約型固定 (2026-02-28)

### 1) 対象

- `scripts/v460/lib/ab_judgment.py`
- `tests/unit/v460/test_160_ab_judgment.py`

### 2) Any削減 / 保守性改善

- `ab_judgment` の入力レコード型を `JSONObject` ベースに統一し、
  `FillRecord` alias を導入。
- 判定メトリクス戻り値を `JudgmentMetrics`、日次内訳を `DailyBreakdownRow`
  (`TypedDict`) に固定し、`Any` 注釈を削減。
- `ABJudgmentCriteria.from_dict()` / `TrendingEvalCriteria.from_dict()` の
  入力も `JSONObject` に寄せ、YAML payload 契約を明確化。
- テスト側の helper も `JSONObject` に寄せ、
  `tmp_path: Any` を `Path` に置換。

### 3) 検証

- `tests/unit/v460/test_160_ab_judgment.py`: `65 passed`
- `any_inventory`:
  - `scripts/v460/lib/ab_judgment.py`
  - `tests/unit/v460/test_160_ab_judgment.py`
  - 結果: `any_type_debt_tokens=0`

## Step92: `metrics_utils` 契約化 + `side_regime_dashboard` fallback 防御 (2026-02-28)

### 1) 対象

- `scripts/v460/lib/metrics_utils.py`
- `scripts/v460/analysis/side_regime_dashboard.py`
- `tests/unit/v460/test_160_ab_judgment.py`

### 2) Any削減 / 重複削減 / 不具合封じ

- `metrics_utils` に `MetricRecord` / `BaseMetrics` / `ExtendedMetrics`
  (`TypedDict`) を導入し、共通メトリクス契約を明示化。
- `compute_base_metrics()` / `compute_extended_metrics()` の
  入出力注釈から `Any` を除去。
- `_collect_finite_values()` を `metrics_utils` 側へ集約し、
  PnL / drift 抽出の重複を削減。
- `side_regime_dashboard` は `MetricRecord` を使うように変更し、
  `compute_extended_metrics()` 呼び出しの `type: ignore[arg-type]` を除去。
- `side_regime_dashboard` の JSONL fallback 読み込みでも
  object 以外の JSON 行を無視し、後続の `.get()` 前提崩れを防止。

### 3) テスト強化

- `side_regime_dashboard._load_all_records()` の fallback が
  非 object 行を無視することを追加検証。

## Step93: `train_alt_horizon` の payload 型固定 + skip集計重複削減 (2026-02-28)

### 1) 対象

- `scripts/v460/ml/train_alt_horizon.py`

### 2) Any削減 / 重複削減 / 軽微な効率化

- `AltSpec` / `DataStats` / `PredStats` / `TrainReport` / `ErrorReport`
  を `TypedDict` 化し、訓練 payload の `Any` 注釈を除去。
- `evaluate_skip_quality()` の返却を `EvalResults` 契約に固定。
- `evaluate_skip_quality()` で
  `np.percentile(preds, skip_pct)` と `preds >= threshold` を
  horizon ごとに再計算していたため、skip率ごとに一度だけ前計算するよう整理。
- `skip{pct}_n_keep` の重複代入も 1 回に集約。

## Step94: JSONL reader の BOM耐性追加 + ローカルfallback削減 (2026-02-28)

### 1) 対象

- `ztb/io/jsonl.py`
- `scripts/v460/lib/stopgap_health.py`
- `scripts/v460/analysis/side_regime_dashboard.py`
- `tests/unit/v460/test_stopgap_health.py`
- `tests/unit/v460/test_160_ab_judgment.py`

### 2) 実運用改善 / 重複削減

- `ztb.io.jsonl.iter_jsonl_objects()` の先頭行で UTF-8 BOM を除去し、
  BOM 付き JSONL を追加 fallback なしで読めるよう改善。
- これにより `read_jsonl_objects()` 利用箇所全体で BOM 耐性が有効化。
- `stopgap_health` / `side_regime_dashboard` の
  `utf-8-sig` 再読込 fallback を削除し、読み込み経路を共通 helper に一本化。
- 非 object 行のスキップも共通 helper に一本化され、ローカル分岐を削減。

### 3) テスト更新

- 両 loader テストを、fallback 強制ではなく
  `BOM + 非 object 行` の実入力に対する実挙動確認へ更新。

## Step95 実施内容（scripts/v460 実行パス優先: order/fill/dashboard 型安全化）

### 1) 対象ファイル

- `scripts/v460/lib/order_monitor.py`
- `scripts/v460/lib/fill_cycle_executor.py`
- `scripts/v460/lib/fill_record_helpers.py`
- `scripts/v460/analysis/side_regime_dashboard.py`

### 2) 実運用改善 / 重複削減

- `order_monitor` の regime timeout / regime reprice 解決で、
  `None` を辞書 `.get()` に渡さない分岐へ整理し、
  `type: ignore[arg-type]` を除去。
- `fill_cycle_executor` で `adapter.place_order()` 戻り値に
  `OrderLike` 実行時ガードを追加し、
  非互換オブジェクト混入時は早期に `TypeError` で検出するよう改善。
- `fill_record_helpers` の `_make_skip_record()` は
  `extra` を dataclass 既知フィールドだけへ限定して適用する形に変更。
- これにより unknown key / base field 重複 key は debug ログへ退避し、
  `FillRecord` 構築時の動的 `**extra` 依存を排除。
- `side_regime_dashboard` の trending 日次集計では
  `timestamp` を `safe_to_finite()` 経由で正規化し、
  `0` も有効 timestamp として扱えるよう改善。

### 3) 検証

- `py_compile`
  - `scripts/v460/lib/order_monitor.py`
  - `scripts/v460/lib/fill_cycle_executor.py`
  - `scripts/v460/lib/fill_record_helpers.py`
  - `scripts/v460/analysis/side_regime_dashboard.py`
- `pytest`
  - `tests/unit/v460/test_143_regime_utilization.py`
  - `tests/unit/v460/test_094_stale_order.py`
  - `tests/unit/v460/test_179_regime_policy_cycle_strategy.py`
  - `tests/unit/v460/test_145_structural_fixes.py`
  - `tests/unit/v460/test_159_side_regime_dashboard.py`
  - `tests/unit/v460/test_160_ab_judgment.py`
  - 結果: `307 passed`
- `any_inventory`
  - 対象 4 ファイルとも `any_type_debt_tokens=0`

## Step96 実施内容（学習評価ロジックの共通化 + feature enricher 軽量化）

### 1) 対象ファイル

- `scripts/v460/ml/skip_eval_utils.py`
- `scripts/v460/ml/train_sg_v3.py`
- `scripts/v460/ml/deploy_sg_v3.py`
- `scripts/v460/ml/deploy_sg_v4.py`
- `scripts/v460/ml/train_alt_horizon.py`
- `scripts/v460/ml/retrain_scheduler.py`
- `scripts/v460/ml/feature_enricher.py`

### 2) 実運用改善 / 重複削減

- `skip_eval_utils` を新設し、
  skip percentile ごとの `baseline / kept / improvement / keep_mask` 計算を共通化。
- これにより `train_sg_v3` / `deploy_sg_v3` / `deploy_sg_v4` /
  `train_alt_horizon` / `retrain_scheduler` の重複ロジックを削減。
- `safe_finite_mean()` を共通化し、
  `np.nanmean()` が空集合や全欠損で `nan` / warning を返す経路を抑止。
- スコアが同値に偏るケースで keep 集合が空になっても、
  baseline をそのまま返して改善値を `0` に固定するようにし、
  学習レポートが `nan` 汚染される不具合余地を低減。
- `retrain_scheduler` の fold-level kept/all PnL 抽出も
  共通 `keep_mask` を使う形に整理し、選別契約を一本化。
- `feature_enricher.enrich_fill_records()` は
  `iterrows()` を廃止して `timestamp` 配列の直接走査へ変更。
- これにより学習前処理ホットパスでの row 単位 `Series` 生成を削減し、
  メモリ断片化と Python オブジェクト生成コストを圧縮。

### 3) 検証

- `py_compile`
  - `scripts/v460/ml/skip_eval_utils.py`
  - `scripts/v460/ml/train_sg_v3.py`
  - `scripts/v460/ml/deploy_sg_v3.py`
  - `scripts/v460/ml/deploy_sg_v4.py`
  - `scripts/v460/ml/train_alt_horizon.py`
  - `scripts/v460/ml/retrain_scheduler.py`
  - `scripts/v460/ml/feature_enricher.py`
- `pytest`
  - `tests/unit/v460/test_train_sg_v3.py`
  - `tests/unit/v460/test_retrain_hot_reload.py -k "evaluate_wf_multi_returns_fold_data or single_window_returns_fold_data"`
  - `tests/unit/v460/test_189_alt_horizon_macro_integration.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - 結果: `110 passed`
- `any_inventory`
  - 対象 7 ファイルとも `any_type_debt_tokens=0`

## Step97 実施内容（feature_enricher: trades / orderbook 窓参照の配列化）

### 1) 対象ファイル

- `scripts/v460/ml/feature_enricher.py`

### 2) 実運用改善 / 重複削減

- `TradeFeatureContext` を導入し、
  `trades` の `timestamp / price / amount / buy_volume 累積和` を
  1 回だけ前計算する形に変更。
- `enrich_fill_records()` では
  30s / 60s / 300s の各窓を DataFrame slice + `sum()` で都度計算せず、
  `searchsorted + 累積和差分` で解くよう改善。
- これにより 1 fill あたりの `trades` 側処理は
  行スライス生成と `side.str.lower()` 再評価を避け、
  Python / pandas オブジェクト生成量を大幅に削減。
- `OrderbookFeatureContext` も導入し、
  `spread_bps / depth_imbalance / bid_vol_5 / ask_vol_5 / mid_price` を配列化。
- `_find_nearest_ob()` と `_compute_return_momentum()` は
  `ob_df.iloc[...]` 依存を減らし、配列直接参照へ変更。
- `enrich_fill_records()` 全体で、
  `trades` / `orderbook` ともに per-record の DataFrame row アクセスをほぼ排除。

### 3) 検証

- `py_compile`
  - `scripts/v460/ml/feature_enricher.py`
- `pytest`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - `tests/unit/v460/test_ob_recorder.py -k "feature_enricher or matched or load_raw_orderbook or _find_nearest_ob"`
  - 結果: `65 passed`
- `any_inventory`
  - `scripts/v460/ml/feature_enricher.py`: `any_type_debt_tokens=0`

## Step98 実施内容（時刻特徴量共通化 + side×hour フィルタのベクトル化）

### 1) 対象ファイル

- `scripts/v460/ml/frame_utils.py`
- `scripts/v460/ml/data_loader.py`
- `scripts/v460/ml/feature_enricher.py`
- `scripts/v460/ml/run_073_strategy_analysis.py`
- `scripts/v460/ml/run_075_verification.py`

### 2) 実運用改善 / 重複削減

- `frame_utils` を新設し、
  `compute_local_hour_cyclic()` / `compute_utc_hour()` を共通化。
- `data_loader` / `feature_enricher` の
  `timestamp.apply(datetime.fromtimestamp(...))` をベクトル化 helper に置換。
- これにより hour cyclic 特徴量生成の重複を解消し、
  学習データ準備時の Python レベル callback 実行を削減。
- `frame_utils.collect_bad_side_hours()` を追加し、
  `side×hour` ごとの悪化スロット抽出を `groupby` 集約へ置換。
- `frame_utils.exclude_side_hour_combos()` を追加し、
  `apply(axis=1)` ベースの tuple 判定をベクトル化マスクへ置換。
- `run_073_strategy_analysis` / `run_075_verification` の
  time filter 系戦略は上記 helper に統一し、
  fold ごとの test filtering コストを削減。
- `iterrows()` による表示ループも `itertuples()` へ置換し、
  行オブジェクト生成を軽量化。
- `feature_enricher` の特徴量組み立てでは
  同一 index の `.loc[...]` を繰り返さず、
  slice 済み DataFrame を再利用する形へ整理。

### 3) 検証

- `py_compile`
  - `scripts/v460/ml/frame_utils.py`
  - `scripts/v460/ml/data_loader.py`
  - `scripts/v460/ml/feature_enricher.py`
  - `scripts/v460/ml/run_073_strategy_analysis.py`
  - `scripts/v460/ml/run_075_verification.py`
- `pytest`
  - `tests/unit/v460/test_ml_pipeline.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - 結果: `80 passed`
- import smoke
  - `scripts.v460.ml.run_073_strategy_analysis`
  - `scripts.v460.ml.run_075_verification`
  - `scripts.v460.ml.data_loader`
  - `scripts.v460.ml.feature_enricher`
  - `scripts.v460.ml.frame_utils`
- `any_inventory`
  - 対象 5 ファイルとも `any_type_debt_tokens=0`

## Step99 実施内容（070/073/075 の時間帯集計整理 + Monte Carlo block bootstrap 軽量化）

### 1) 対象ファイル

- `scripts/v460/ml/frame_utils.py`
- `scripts/v460/ml/run_070_model_search.py`
- `scripts/v460/ml/run_070_deep_analysis.py`
- `scripts/v460/ml/run_070_final_analysis.py`
- `scripts/v460/ml/run_073_strategy_sweep.py`
- `scripts/v460/ml/run_075_verification.py`

### 2) 実運用改善 / 重複削減

- `frame_utils` に
  `_select_side_hour_combos()` / `collect_good_side_hours()` /
  `_match_side_hour_combos()` を追加し、
  side×hour 集計後の抽出と include/exclude 判定を共通化。
- `collect_bad_side_hours()` は `iterrows()` を廃止し、
  集約済み `count` / `mean` 配列から直接組み合わせを抽出する形へ変更。
- `run_073_strategy_sweep` の
  `s11_best_hours_only()` は `collect_good_side_hours()` へ統一し、
  local `groupby + iterrows()` 実装を削除。
- `run_070_model_search` / `run_070_deep_analysis` / `run_070_final_analysis` は
  `compute_utc_hour()` を使う形へ寄せ、
  `timestamp.apply(datetime.fromtimestamp(...))` の重複を削減。
- `run_070_final_analysis` では
  chained indexing を `train_window` 経由に整理し、
  学習半区間の sell 抽出を読みやすく保守しやすい形へ修正。
- `run_070_deep_analysis.analyze_round_trip_detail()` は
  `side` / `price` / `timestamp` / `entry_hour` を配列化し、
  `filled.iloc[i]` / `filled.iloc[j]` の row object 生成をやめて
  連続同sideスキャンを軽量化。
- `run_075_verification.section_5_monte_carlo_50k()` は
  既に修正済みの true block bootstrap を前提に、
  block の `size` / `sum` を前計算し、
  各反復でサンプル配列を毎回構築せず「合計値だけ」を計算する形へ変更。
- これにより Monte Carlo 部分の大きな一時配列確保を減らし、
  1000 回反復時の Python/NumPy オブジェクト生成コストを圧縮。

### 3) 検証

- `py_compile`
  - `scripts/v460/ml/frame_utils.py`
  - `scripts/v460/ml/run_073_strategy_sweep.py`
  - `scripts/v460/ml/run_070_model_search.py`
  - `scripts/v460/ml/run_070_deep_analysis.py`
  - `scripts/v460/ml/run_070_final_analysis.py`
  - `scripts/v460/ml/run_075_verification.py`
- import smoke
  - `scripts.v460.ml.run_070_model_search`
  - `scripts.v460.ml.run_070_deep_analysis`
  - `scripts.v460.ml.run_070_final_analysis`
  - `scripts.v460.ml.run_073_strategy_sweep`
  - `scripts.v460.ml.run_075_verification`
  - block helper smoke: `_prepare_block_stats()` / `_sample_block_bootstrap_sum()`
- `pytest`
  - `tests/unit/v460/test_ml_pipeline.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - 結果: `80 passed`
- `any_inventory`
  - 対象 6 ファイルとも `any_type_debt_tokens=0`

## Step100 実施内容（`run_075_verification` の pool 集計一時配列削減）

### 1) 対象ファイル

- `scripts/v460/ml/run_075_verification.py`

### 2) 実運用改善 / 重複削減

- `section_5_monte_carlo_50k()` で
  `pool_before` / `after_buy` / `after_sell` / `pool_after` の
  一時 NumPy 配列を先に組み立てる実装を廃止。
- 既に構築している日次 block と
  `_prepare_block_stats()` の `size` / `sum` 前計算値から、
  pool 件数と平均 PnL を直接算出する形へ変更。
- これにより bootstrap 本体とは別に保持していた
  「比較用の重複配列」をなくし、
  Monte Carlo 前段のメモリ消費と無駄な連結処理を削減。

### 3) 検証

- `py_compile`
  - `scripts/v460/ml/run_075_verification.py`
- import smoke
  - `scripts.v460.ml.run_075_verification`
- `any_inventory`
  - `scripts/v460/ml/run_075_verification.py`: `any_type_debt_tokens=0`

## Step101 実施内容（`skip_gate` hot path の単一走査化 + malformed input 防御）

### 1) 対象ファイル

- `scripts/v460/ml/skip_gate.py`
- `scripts/v460/lib/skip_gate_evaluator.py`

### 2) 実運用改善 / 重複削減

- `skip_gate` に
  `_coerce_finite_float()` / `_append_bounded_history()` /
  `_move_toward_target()` を追加し、
  数値変換・履歴 trimming・段階閾値移動の重複ロジックを共通化。
- `SkipGate.__init__()` で `feature_cols -> index` の map を前計算し、
  `evaluate()` は `_build_feature_vector()` 経由で
  `features.items()` を 1 回だけ走査する形へ変更。
- これにより、毎回 `feature_cols` 全件をなめて
  `if col in features` を繰り返す実装を削減。
- `evaluate()` の single-row 入力は
  `x.reshape(1, -1)` をそのまま `DataFrame` 化し、
  余計な list 包装を減らした。
- `build_features_from_market_state()` は
  `_summarize_recent_trades()` を使う単一走査へ変更し、
  trade window の filter / buy volume / sell volume / price velocity を
  別々に計算していた処理を統合。
- 同時に、`recent_trades` の
  不正な `ts` / `amount` / `price` / `side` で落ちないようにし、
  malformed 行はスキップまたは 0 扱いで継続する防御を追加。
- price velocity は「入力順」ではなく
  window 内の最古/最新 `ts` を優先して計算する形に変更し、
  未整列 trade 入力で誤った符号や値になり得る不具合余地を削減。
- `warm_start_skip_gate_thresholds()` は
  `file_records + prob_records` の前方連結を廃止し、
  newest-first で必要件数だけ保持する形へ変更。
  これにより、起動時 warm start の不要な list コピーを削減。
- `skip_gate_evaluator` では
  `build_features_from_market_state` / `SkipDecision` の
  per-call ローカル import を外し、常用パスの小さなオーバーヘッドを削減。

### 3) 検証

- `py_compile`
  - `scripts/v460/ml/skip_gate.py`
  - `scripts/v460/lib/skip_gate_evaluator.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
- `pytest`
  - `tests/unit/v460/test_enricher_skip_gate.py`
    - `test_evaluate_returns_decision`
    - `test_skip_rate_limit`
    - `test_with_recent_trades`
    - `test_recent_trades_handles_unsorted_and_malformed_rows`
    - `test_trade_window_sec_filters_trades`
    - `test_calibrate_after_warmup_adjusts`
    - `test_warm_start_restores_history`
    - `test_warm_start_prefers_most_recent_records`
  - `tests/unit/v460/test_193_ev_offset.py`
  - 結果: `25 passed`
- `any_inventory`
  - `scripts/v460/ml/skip_gate.py`: `any_type_debt_tokens=0`
  - `scripts/v460/lib/skip_gate_evaluator.py`: `any_type_debt_tokens=0`

## Step102 実施内容（`skip_gate_evaluator` の skip record 統一 + trade payload 正規化）

### 1) 対象ファイル

- `scripts/v460/lib/skip_gate_evaluator.py`
- `tests/unit/v460/test_skip_gate_v3.py`

### 2) 実運用改善 / 重複削減

- `skip_gate_evaluator` に
  `_get_trade_field()` / `_normalize_recent_trades()` を追加し、
  adapter から返る recent trade が `dict` / object 混在でも
  `skip_gate` へ渡す payload を共通形式に正規化するよう変更。
- これにより、従来の `getattr(...)` 前提では拾えていなかった
  dict 形式の `timestamp` / `amount` / `quantity` / `side` も扱えるようにした。
- `_make_skip_fill_record()` を追加し、
  unknown-regime rule skip / velocity rule skip / 通常 model skip の
  `FillRecord(...)` 生成を一本化。
- 3 箇所で重複していた
  `cycle_id` / `order_price` / `spread_at_order` / skip metadata の束を
  1 箇所へ集約し、保守時の変更漏れを減らした。
- `evaluate()` では `market_ts` を 1 回だけ取得し、
  feature 構築・trade fallback timestamp・early return record の
  timestamp を揃える形へ変更。
- UTC hour 算出は `datetime.now(timezone.utc)` をやめて
  `time.gmtime(market_ts).tm_hour` に変更し、
  常用パスの object 生成を減らした。

### 3) 検証

- `py_compile`
  - `scripts/v460/lib/skip_gate_evaluator.py`
  - `tests/unit/v460/test_skip_gate_v3.py`
- `pytest`
  - `tests/unit/v460/test_skip_gate_v3.py`
  - `tests/unit/v460/test_195_velocity_b1_soft.py`
  - `tests/unit/v460/test_141_side_specific_models.py`
  - `-k "skip_unknown or normalize_recent_trades or model_used_tag or side_only_missing_side_returns_reason or velocity"`
  - 結果: `36 passed`
- `any_inventory`
  - `scripts/v460/lib/skip_gate_evaluator.py`: `any_type_debt_tokens=0`
  - `tests/unit/v460/test_skip_gate_v3.py`: `any_type_debt_tokens=0`

## Step103 実施内容（`order_monitor` の reprice skip 判定統一 + `SkipGateResult` 代入集約）

### 1) 対象ファイル

- `scripts/v460/lib/order_monitor.py`
- `scripts/v460/lib/skip_gate_evaluator.py`
- `tests/unit/v460/test_143_regime_utilization.py`
- `tests/unit/v460/test_skip_gate_v3.py`

### 2) 実運用改善 / 重複削減

- `order_monitor` に
  `_resolve_regime_name()` を追加し、
  timeout/reprice 用のレジーム名取得を共通化。
- `order_monitor` に
  `_should_block_reprice_with_skip_gate()` を追加し、
  stale reprice 前の SkipGate 判定
  (`build_features_from_market_state` + threshold_offset + ログ) を
  1 箇所へ集約。
- これにより、`monitor()` 内の局所ロジックから
  ローカル import / feature 構築 / decision ログの重複を削減し、
  reprice 判定の見通しを改善。
- reprice check 時刻を `reprice_check_ts` として 1 回だけ取得し、
  SkipGate feature の時刻と `last_reprice_time` を揃える形へ統一。
- `skip_gate_evaluator` には
  `_assign_result_fields()` / `_apply_decision_to_result()` を追加し、
  `SkipGateResult` への代入束
  (`skipped/score/reason/model_used/as_prob/threshold_used/hour_offset`) を
  共通化。
- unknown rule skip / no_model_for_side / velocity rule skip / 最終 model decision
  の result 反映が同じ経路に寄り、変更時の項目漏れを減らした。
- `test_143_regime_utilization.py` の既存 `Any` 注釈を `MagicMock` に置換し、
  触った周辺テストも `any_type_debt_tokens=0` に維持。

### 3) 検証

- `py_compile`
  - `scripts/v460/lib/skip_gate_evaluator.py`
  - `scripts/v460/lib/order_monitor.py`
  - `tests/unit/v460/test_skip_gate_v3.py`
  - `tests/unit/v460/test_143_regime_utilization.py`
- `pytest`
  - `tests/unit/v460/test_skip_gate_v3.py`
  - `tests/unit/v460/test_195_velocity_b1_soft.py`
  - `tests/unit/v460/test_141_side_specific_models.py`
  - `tests/unit/v460/test_143_regime_utilization.py`
  - `-k "skip_unknown or normalize_recent_trades or model_used_tag or side_only_missing_side_returns_reason or velocity or OrderMonitorHelpers or reprice_offset_increases_limit or negative_offset_clamps_to_zero"`
  - 結果: `40 passed`
- `any_inventory`
  - `scripts/v460/lib/skip_gate_evaluator.py`: `any_type_debt_tokens=0`
  - `scripts/v460/lib/order_monitor.py`: `any_type_debt_tokens=0`
  - `tests/unit/v460/test_skip_gate_v3.py`: `any_type_debt_tokens=0`
  - `tests/unit/v460/test_143_regime_utilization.py`: `any_type_debt_tokens=0`

### Step104: executor failure skip record を共通 helper に統一

1. 対応概要
- `scripts/v460/lib/fill_record_helpers.py`
  - `_make_skip_record()` に `timestamp` 引数を追加し、明示時刻を保持できるようにした。
- `scripts/v460/lib/fill_cycle_executor.py`
  - 全注文試行失敗時の ad-hoc `FillRecord(...)` を `_make_skip_record()` 経由へ置換した。
- `tests/unit/v460/test_145_structural_fixes.py`
  - `_make_skip_record(timestamp=...)` の保持を検証する回帰テストを追加した。
  - 併せて touched helper の `Any` 注釈を削除した。

2. 目的
- executor 側の failure/cancel record 生成を、既存の skip record 契約へ寄せて変更漏れを防ぐ。
- 送信開始時刻 `t_submit` をそのまま記録できるようにし、失敗記録の時刻整合を上げる。

3. 検証
- `py_compile`
  - `scripts/v460/lib/fill_record_helpers.py`
  - `scripts/v460/lib/fill_cycle_executor.py`
  - `tests/unit/v460/test_145_structural_fixes.py`
- `pytest`
  - `tests/unit/v460/test_145_structural_fixes.py`
  - `-k "TestMakeSkipRecord or test_order_attempts_fail_returns_record"`
  - 結果: `7 passed`
- `any_inventory`
  - `scripts/v460/lib/fill_record_helpers.py`: `any_type_debt_tokens=0`
  - `scripts/v460/lib/fill_cycle_executor.py`: `any_type_debt_tokens=0`
  - `tests/unit/v460/test_145_structural_fixes.py`: `any_type_debt_tokens=0`

### Step105: skip FillRecord builder を evaluator と共通化

1. 対応概要
- `scripts/v460/lib/fill_record_helpers.py`
  - `build_skip_fill_record()` を追加し、skip/監査系 `FillRecord` の共通 builder を module-level に抽出した。
  - 追加フィールド反映も `_apply_skip_record_extras()` に分離した。
- `scripts/v460/lib/skip_gate_evaluator.py`
  - `_make_skip_fill_record()` を `build_skip_fill_record()` 経由へ切り替えた。

2. 目的
- `skip_gate_evaluator` と runner 側で二重化していた `FillRecord` 初期化契約を一本化する。
- skip 系フィールド追加時の変更漏れを減らし、`duplicate/unknown key` の防御も同じルールへ揃える。

3. 検証
- `py_compile`
  - `scripts/v460/lib/fill_record_helpers.py`
  - `scripts/v460/lib/skip_gate_evaluator.py`
- `pytest`
  - `tests/unit/v460/test_skip_gate_v3.py`
  - 結果: `15 passed`
- `any_inventory`
  - `scripts/v460/lib/fill_record_helpers.py`: `any_type_debt_tokens=0`
  - `scripts/v460/lib/skip_gate_evaluator.py`: `any_type_debt_tokens=0`

### Step106: loop 側 skip record 呼び出しを wrapper 化

1. 対応概要
- `scripts/v460/lib/fill_loop_orchestrator.py`
  - `_make_loop_skip_record()` を追加し、`run_continuous` 内の loop-level skip record を一本化した。
  - `regime=self._current_regime_value()` の重複指定を除去した。
  - `preflight_pause` は record timestamp と `cycle_id` の時刻成分に同じ値を使うよう揃えた。

2. 目的
- ループ側の skip record が常に同じ契約で現在レジームを保持するようにし、呼び出し側の重複と漏れを減らす。
- `preflight_pause` の記録時刻と ID の不一致余地をなくし、監査しやすくする。

3. 検証
- `py_compile`
  - `scripts/v460/lib/fill_loop_orchestrator.py`
- `pytest`
  - `tests/unit/v460/test_143_regime_utilization.py`
  - `tests/unit/v460/test_145_structural_fixes.py`
  - 結果: `114 passed`
- `any_inventory`
  - `scripts/v460/lib/fill_loop_orchestrator.py`: `any_type_debt_tokens=0`

### Step107: cycle 側 skip record 呼び出しも wrapper 化

1. 対応概要
- `scripts/v460/lib/fill_cycle_executor.py`
  - `_make_cycle_skip_record()` を追加し、`run_single_cycle` 内の skip record 生成を一本化した。
  - circuit breaker / orderbook error / narrow spread pause / order attempt failure の 4 経路を統一した。

2. 目的
- サイクル側の skip 記録も loop 側と同じく `regime` の伝搬を wrapper に閉じ込め、呼び出し側の重複指定をなくす。
- cycle-level の cancel/skip reason 追加時に変更箇所を狭める。

3. 検証
- `py_compile`
  - `scripts/v460/lib/fill_cycle_executor.py`
- `pytest`
  - `tests/unit/v460/test_145_structural_fixes.py`
  - `結果: 54 passed`
- `any_inventory`
  - `scripts/v460/lib/fill_cycle_executor.py`: `any_type_debt_tokens=0`

### Step108: skip FillRecord builder を ztb 側へ昇格

1. 対応概要
- `ztb/metrics/fill_quality.py`
  - `build_skip_fill_record()` を追加し、`FillRecord` 定義元へ skip/監査系 builder を昇格した。
  - known field のみ反映し、unknown field は無視する契約を ztb 側へ集約した。
- `scripts/v460/lib/fill_record_helpers.py`
  - local builder 実装を削除し、`ztb.metrics.fill_quality.build_skip_fill_record` を利用する形へ整理した。
- `tests/unit/v460/test_fill_quality.py`
  - ztb 側 builder が known extra のみ反映し、unknown extra を捨てることを追加検証した。

2. 目的
- `FillRecord` の生成責務を `FillRecord` 定義元へ寄せ、v460 固有 helper から ztb 共通資産として再利用しやすくする。
- 今回切り出した builder を、他系統（v459/将来 v4xx）でもそのまま使える配置にする。

3. 検証
- `py_compile`
  - `ztb/metrics/fill_quality.py`
  - `scripts/v460/lib/fill_record_helpers.py`
- `pytest`
  - `tests/unit/v460/test_fill_quality.py`
  - `tests/unit/v460/test_145_structural_fixes.py`
  - 結果: `244 passed`
- `any_inventory`
  - `ztb/metrics/fill_quality.py`: `any_type_debt_tokens=0`
  - `scripts/v460/lib/fill_record_helpers.py`: `any_type_debt_tokens=0`

### Step109: generic FillRecord builder と成功系 record の統合

1. 対応概要
- `ztb/metrics/fill_quality.py`
  - `_sanitize_fill_record_fields()` を追加し、`known field のみ通す` 契約を共通化した。
  - `FillRecord.from_dict()` を同 helper 経由に変更した。
  - `build_fill_record()` を追加し、generic な `FillRecord` builder を導入した。
  - `build_skip_fill_record()` も `build_fill_record()` 経由へ変更した。
  - skip builder の protected field に `filled` を追加し、`filled=True` 上書きで skip semantics が崩れる余地を塞いだ。
- `scripts/v460/lib/fill_cycle_executor.py`
  - `_build_fill_record()` を `build_fill_record()` 経由へ変更した。
  - cancel reason 解決を `_resolve_fill_cancel_reason()` に抽出した。
  - `spread_bps` 計算を `_compute_fill_spread_bps()` に抽出した。
- `tests/unit/v460/test_fill_quality.py`
  - generic builder の unknown field 無視を追加検証した。
  - skip builder が `filled=True` を受けても skip semantics を維持することを追加検証した。
- `tests/unit/v460/test_145_structural_fixes.py`
  - executor の `_build_fill_record()` が共通 builder を使う構造テストを追加した。

2. 目的
- `FillRecord.from_dict()` / skip builder / 成功系 builder の「既知フィールドだけを通す」契約を 1 箇所に揃える。
- 成功系 record も `FillRecord` 定義元の builder を通すことで、v460 側の直書き初期化を減らし、横展開しやすくする。

3. 検証
- `py_compile`
  - `ztb/metrics/fill_quality.py`
  - `scripts/v460/lib/fill_record_helpers.py`
  - `scripts/v460/lib/fill_cycle_executor.py`
- `pytest`
  - `tests/unit/v460/test_fill_quality.py`
  - `tests/unit/v460/test_145_structural_fixes.py`
  - `tests/unit/v460/test_skip_gate_v3.py`
  - `結果: 261 passed`
- `any_inventory`
  - `ztb/metrics/fill_quality.py`: `any_type_debt_tokens=0`
  - `scripts/v460/lib/fill_record_helpers.py`: `any_type_debt_tokens=0`
  - `scripts/v460/lib/fill_cycle_executor.py`: `any_type_debt_tokens=0`

### Step110: FillRecord JSONL I/O のメモリ使用と重複排除補助を整理

1. 対応概要
- `ztb/metrics/fill_quality.py`
  - `save_fill_records()` の一時 `lines` 配列を廃止し、temp file へ逐次書き込みに変更した。
  - temp file から本体ファイルへの append も `read()` 一括読み込みではなく `shutil.copyfileobj()` に変更した。
  - `load_fill_records_glob()` の重複排除ループを `_extend_unique_fill_records()` に抽出した。
- `tests/unit/v460/test_fill_quality.py`
  - emergency dump 側の duplicate が cross-file で重複追加されないことを追加検証した。

2. 目的
- 大きい batch 保存時に JSONL 全行文字列を一度に保持しないようにし、ピークメモリを下げる。
- `fill_records_*` / `emergency_*` の重複排除規約を 1 箇所に寄せて、挙動差の混入を防ぐ。

3. 検証
- `py_compile`
  - `ztb/metrics/fill_quality.py`
- `pytest`
  - `tests/unit/v460/test_fill_quality.py`
  - 結果: `192 passed`
- `any_inventory`
  - `ztb/metrics/fill_quality.py`: `any_type_debt_tokens=0`

### Step111: save_fill_records を Iterable 契約へ拡張

1. 対応概要
- `ztb/metrics/fill_quality.py`
  - `save_fill_records()` の入力型を `list[FillRecord]` から `Iterable[FillRecord]` に拡張した。
  - 逐次書き込みの件数を loop 内で数えるようにし、`len(records)` 依存を除去した。

2. 目的
- ストリーム書き込み実装と型契約を一致させ、呼び出し側に list への事前具象化を強制しない。
- 大きい batch 保存時に、ログ件数取得のためだけに `Sized` を要求しない。

3. 検証
- `py_compile`
  - `ztb/metrics/fill_quality.py`
- `pytest`
  - `tests/unit/v460/test_fill_quality.py -k "TestFillRecordIO"`
  - 結果: `5 passed`
- `any_inventory`
  - `ztb/metrics/fill_quality.py`: `any_type_debt_tokens=0`

### Step112: fill_record_helpers の runtime import を縮小

1. 対応概要
- `scripts/v460/lib/fill_record_helpers.py`
  - `FillRecord` の runtime import を外し、`TYPE_CHECKING` 側へ移動した。
  - runtime では実際に使う `build_skip_fill_record` / `load_fill_records_glob` のみを import する形に整理した。

2. 目的
- type annotation 用の import を runtime 依存から切り離し、モジュール初期化時の結合を少し下げる。
- helper モジュールの責務を `skip record` / resume 用の実利用シンボルに絞る。

3. 検証
- `py_compile`
  - `scripts/v460/lib/fill_record_helpers.py`
- `pytest`
  - `tests/unit/v460/test_145_structural_fixes.py -k "TestMakeSkipRecord"`
  - 結果: `7 passed`
- `any_inventory`
  - `scripts/v460/lib/fill_record_helpers.py`: `any_type_debt_tokens=0`

### Step113: FillRecord 読み込みも iterator 化して list 中継を削減

1. 対応概要
- `ztb/metrics/fill_quality.py`
  - `iter_fill_records()` を追加し、JSONL 読み込みを逐次 yield できるようにした。
  - `load_fill_records()` は `list(iter_fill_records(...))` の薄いラッパに整理した。
  - `load_fill_records_glob()` は `load_fill_records()` の中間 list を作らず、`iter_fill_records()` を直接マージする形に変更した。
  - `_extend_unique_fill_records()` の入力も `Iterable[FillRecord]` に拡張した。
- `tests/unit/v460/test_fill_quality.py`
  - `iter_fill_records()` の roundtrip テストを追加した。

2. 目的
- 大きい JSONL 群を横断する際に、各ファイルを一度 list 化してから再マージする無駄をなくす。
- API は既存互換 (`load_fill_records`) を維持したまま、内部だけ逐次処理へ寄せる。

3. 検証
- `py_compile`
  - `ztb/metrics/fill_quality.py`
- `pytest`
  - `tests/unit/v460/test_fill_quality.py -k "TestFillRecordIO"`
  - 結果: `6 passed`
- `any_inventory`
  - `ztb/metrics/fill_quality.py`: `any_type_debt_tokens=0`

### Step114: skip_gate_evaluator の builder 依存を定義元へ直結

1. 対応概要
- `scripts/v460/lib/skip_gate_evaluator.py`
  - `build_skip_fill_record` の import 元を `scripts.v460.lib.fill_record_helpers` から `ztb.metrics.fill_quality` へ変更した。

2. 目的
- re-export 的な import 経路をやめ、`FillRecord` builder の定義元へ直接依存させる。
- `skip_gate_evaluator` と `fill_record_helpers` の不要な runtime 結合を減らす。

3. 検証
- `py_compile`
  - `scripts/v460/lib/skip_gate_evaluator.py`
- `pytest`
  - `tests/unit/v460/test_skip_gate_v3.py`
  - 結果: `15 passed`
- `any_inventory`
  - `scripts/v460/lib/skip_gate_evaluator.py`: `any_type_debt_tokens=0`

### Step115: FillRecord glob 対象ファイル列挙を共通化

1. 対応概要
- `ztb/metrics/fill_quality.py`
  - `_iter_fill_record_files()` を追加し、`fill_records_*.jsonl` と `emergency/emergency_*.jsonl` の列挙規約を共通化した。
  - `load_fill_records_glob()` の 2 重ループを 1 ループに整理した。

2. 目的
- 対象ファイルの走査順序と対象パターンを 1 箇所に閉じ込め、将来の拡張時の変更点を減らす。
- glob 側の重複した走査コードを削って見通しを上げる。

3. 検証
- `py_compile`
  - `ztb/metrics/fill_quality.py`
- `pytest`
  - `tests/unit/v460/test_fill_quality.py -k "TestFillRecordIO"`
  - 結果: `6 passed`
- `any_inventory`
  - `ztb/metrics/fill_quality.py`: `any_type_debt_tokens=0`

### Step116: FillRecord glob 読み込み自体も iterator 化

1. 対応概要
- `ztb/metrics/fill_quality.py`
  - `iter_fill_records_glob()` を追加し、cross-file 重複排除つきの逐次読み込み API を導入した。
  - `load_fill_records_glob()` は `list(iter_fill_records_glob(...))` の薄いラッパに整理した。
  - 旧 `_extend_unique_fill_records()` は不要になったため削除した。
- `tests/unit/v460/test_fill_quality.py`
  - `iter_fill_records_glob()` の roundtrip テストを追加した。

2. 目的
- 単一ファイルだけでなく glob 読み込み全体でも streaming API を持たせ、呼び出し側が必要なら list 化を避けられるようにする。
- list API (`load_fill_records_glob`) はそのまま維持しつつ、内部実装の基準を iterator 側へ寄せる。

3. 検証
- `py_compile`
  - `ztb/metrics/fill_quality.py`
- `pytest`
  - `tests/unit/v460/test_fill_quality.py -k "TestFillRecordIO"`
  - 結果: `7 passed`
- `any_inventory`
  - `ztb/metrics/fill_quality.py`: `any_type_debt_tokens=0`

### Step117: clean/quarantine 分離も iterable 対応へ横展開

1. 対応概要
- `ztb/metrics/fill_quality.py`
  - `partition_clean_records()` を追加し、`Iterable[FillRecord]` を直接 clean/quarantine に分離できるようにした。
  - `filter_clean_records()` は list 互換 API として `partition_clean_records()` を呼ぶ薄いラッパに整理した。
- `scripts/v460/lib/results_analyzer.py`
  - `run_results_only()` を `iter_fill_records_glob()` + `partition_clean_records()` ベースへ変更し、中間 `all_records` list を除去した。
- `scripts/v460/lib/adaptation_engine.py`
  - `_load_clean_records()` を `iter_fill_records_glob()` + `partition_clean_records()` ベースへ変更し、中間 `all_records` を廃止した。
- `scripts/v460/analysis/oracle_baseline.py`
  - `run_oracle_baseline()` を同様に streaming 読み込みへ変更した。
- `tests/unit/v460/test_fill_quality.py`
  - `partition_clean_records()` が generator 入力を受けられることを追加検証した。
- `tests/unit/v460/test_169_ranging_buy_skip_and_metrics.py`
  - `results_analyzer` の patch 対象を新 helper ベースへ更新した。

2. 目的
- 読み込みだけでなく quarantine 分離も iterator ベースに寄せ、`all_records -> clean` の二重保持を減らす。
- `results_analyzer` / `adaptation_engine` / `oracle_baseline` の実運用寄り経路で、ピークメモリを下げる。

3. 検証
- `py_compile`
  - `ztb/metrics/fill_quality.py`
  - `scripts/v460/lib/results_analyzer.py`
  - `scripts/v460/lib/adaptation_engine.py`
  - `scripts/v460/analysis/oracle_baseline.py`
- `pytest`
  - `tests/unit/v460/test_fill_quality.py -k "TestFilterCleanRecordsExpanded or TestFillRecordIO"`
  - `tests/unit/v460/test_169_ranging_buy_skip_and_metrics.py`
  - 結果: `19 passed`
- `any_inventory`
  - `ztb/metrics/fill_quality.py`: `any_type_debt_tokens=0`
  - `scripts/v460/lib/results_analyzer.py`: `any_type_debt_tokens=0`
  - `scripts/v460/lib/adaptation_engine.py`: `any_type_debt_tokens=0`
  - `scripts/v460/analysis/oracle_baseline.py`: `any_type_debt_tokens=0`

### Step118: monitor / verification も streaming 読み込みへ寄せる

1. 対応概要
- `scripts/v460/monitor_fill_test.py`
  - `run_monitor()` を `iter_fill_records_glob()` + `partition_clean_records()` ベースへ変更し、中間 `records` list を除去した。
- `scripts/v460/ml/run_075_verification.py`
  - `load_clean_filled()` を iterator ベースへ変更した。
  - `run_id` / `git_sha` フィルタも generator で前段フィルタする形に変更した。
  - フィルタ後件数の表示は `clean + quarantine` 確定後に出す形へ変更した。
- `tests/unit/v460/test_fill_quality.py`
  - `run_monitor()` の構造テストを新 helper ベースへ更新した。

2. 目的
- モニタリングや 075 検証でも、全レコードの中間 list を不要にし、読み込み時のメモリピークを下げる。
- `run_id` / `git_sha` フィルタも streaming パスに揃える。

3. 検証
- `py_compile`
  - `scripts/v460/monitor_fill_test.py`
  - `scripts/v460/ml/run_075_verification.py`
- `pytest`
  - `tests/unit/v460/test_fill_quality.py -k "Test051MonitorExtensions"`
  - 結果: `3 passed`
- import smoke
  - `scripts.v460.ml.run_075_verification`
- `any_inventory`
  - `scripts/v460/monitor_fill_test.py`: `any_type_debt_tokens=0`
  - `scripts/v460/ml/run_075_verification.py`: `any_type_debt_tokens=0`

### Step119: resume 時の trailing skip カウントを helper 化

1. 対応概要
- `scripts/v460/lib/fill_record_helpers.py`
  - `_count_trailing_cancel_reason()` を追加し、末尾連続 `cancel_reason` 件数の計算を共通化した。
  - `resume_from_existing()` の `trending_sell_skip` / `balance_forced_skip` 復元を helper 経由へ変更した。
- `tests/unit/v460/test_145_structural_fixes.py`
  - helper の直接テストを追加した。

2. 目的
- `resume_from_existing()` 内の重複ループを削り、末尾連続カウント規約を 1 箇所に閉じ込める。
- 今後同種の trailing skip 復元を追加する場合の横展開をしやすくする。

3. 検証
- `py_compile`
  - `scripts/v460/lib/fill_record_helpers.py`
- `pytest`
  - `tests/unit/v460/test_145_structural_fixes.py -k "TestMakeSkipRecord"`
  - 結果: `8 passed`
- `any_inventory`
  - `scripts/v460/lib/fill_record_helpers.py`: `any_type_debt_tokens=0`

### Step120: 075 verification の load_clean_filled runtime bug 修正

1. 対応概要
- `scripts/v460/ml/run_075_verification.py`
  - `stats["total_records"]` が削除済み `all_records` を参照していた不具合を修正し、`len(clean) + len(quarantine)` を使うようにした。
  - `to_df()` の dataclass 手展開をやめ、`FillRecord.to_dict()` ベースに統一した。

2. 目的
- iterator 化後に残っていた `NameError` 余地を除去し、`load_clean_filled()` を実行可能状態に戻す。
- `FillRecord` → `DataFrame` 変換も既存の serialize 契約へ揃える。

3. 検証
- `py_compile`
  - `scripts/v460/ml/run_075_verification.py`
- smoke
  - `load_clean_filled()` を monkeypatch 付きで呼び出し、`stats["total_records"]` を確認
- `any_inventory`
  - `scripts/v460/ml/run_075_verification.py`: `any_type_debt_tokens=0`

### Step121: adaptation_engine の comment drift を解消

1. 対応概要
- `scripts/v460/lib/adaptation_engine.py`
  - モジュール docstring の `load_fill_records_glob` 前提表現を、現在の streaming + TTL cache 実装に合わせて更新した。

2. 目的
- 旧実装を前提にした説明を除去し、実装理解時の誤読を防ぐ。

3. 検証
- `py_compile`
  - `scripts/v460/lib/adaptation_engine.py`

### Step122: PnL bps→JPY 変換を共通 helper 化

1. 対応概要
- `ztb/metrics/fill_quality.py`
  - `compute_record_pnl_jpy()` を追加し、`FillRecord` の 30s PnL を JPY 概算へ変換する helper を導入した。
- `scripts/v460/lib/fill_loop_orchestrator.py`
  - レジューム時とサイクル進行中の `cumulative_pnl_jpy` 更新を helper 経由へ統一した。
- `scripts/v460/monitor_fill_test.py`
  - `_check_cumulative_loss()` とレポート内の累積PnL概算を helper 経由へ統一した。
- `tests/unit/v460/test_fill_quality.py`
  - helper の直接テストと `monitor_fill_test` の source-level 使用確認を追加した。

2. 目的
- `post_fill_30s_pnl * 1e-4 * fill_price * order_quantity` の重複実装を削る。
- JPY 換算ルールを 1 箇所に集約し、将来の計算条件変更を横展開しやすくする。

3. 検証
- `py_compile`
  - `ztb/metrics/fill_quality.py`
  - `scripts/v460/lib/fill_loop_orchestrator.py`
  - `scripts/v460/monitor_fill_test.py`
- `pytest`
  - `tests/unit/v460/test_fill_quality.py -k "TestFillRecord or Test051MonitorExtensions"`
  - 結果: `25 passed`
- `any_inventory`
  - `ztb/metrics/fill_quality.py`: `any_type_debt_tokens=0`
  - `scripts/v460/lib/fill_loop_orchestrator.py`: `any_type_debt_tokens=0`
  - `scripts/v460/monitor_fill_test.py`: `any_type_debt_tokens=0`

### Step123: fill_cycle_executor の成功系 payload を 3 分割

1. 対応概要
- `scripts/v460/lib/fill_cycle_executor.py`
  - `_build_fill_measurement_fields()` を追加し、約定/計測系フィールドを抽出した。
  - `_build_fill_market_fields()` を追加し、市場観測/skip_gate/実行メタ系フィールドを抽出した。
  - `_build_fill_strategy_fields()` を追加し、EV/gated_regime/macro 系フィールドを抽出した。
  - `_build_fill_record()` は base payload + 3 helper の合成に整理した。
- `tests/unit/v460/test_145_structural_fixes.py`
  - `_build_fill_record()` が 3 helper を経由する構造テストへ更新した。

2. 目的
- 成功系 `FillRecord` 組み立ての責務を分解し、変更影響範囲を狭める。
- 分割した payload helper を、将来の他 record builder へ横展開しやすい形にする。

3. 検証
- `py_compile`
  - `scripts/v460/lib/fill_cycle_executor.py`
- `pytest`
  - `tests/unit/v460/test_145_structural_fixes.py -k "TestFillRecordBuilderIntegration"`
  - 結果: `1 passed`
- `any_inventory`
  - `scripts/v460/lib/fill_cycle_executor.py`: `any_type_debt_tokens=0`

### Step124: 残る load_fill_records_glob 呼び出しも iterator-first に揃える

1. 対応概要
- `scripts/v460/lib/fill_record_helpers.py`
  - `resume_from_existing()` の読み込みを `list(iter_fill_records_glob(...))` に変更した。
- `scripts/v460/lib/fill_loop_orchestrator.py`
  - `run_continuous()` 終了時の全件再読込も `list(iter_fill_records_glob(...))` に変更した。
- `tests/unit/v460/test_145_structural_fixes.py`
  - `resume_from_existing()` / `run_continuous()` が `iter_fill_records_glob()` を使う構造テストを追加した。

2. 目的
- 残っていた `load_fill_records_glob()` 呼び出しを、iterator-first で統一する。
- 呼び出し側で「list を欲しいから明示的に list 化している」ことをはっきりさせる。

3. 検証
- `py_compile`
  - `scripts/v460/lib/fill_record_helpers.py`
  - `scripts/v460/lib/fill_loop_orchestrator.py`
- `pytest`
  - `tests/unit/v460/test_145_structural_fixes.py -k "TestFillRecordBuilderIntegration"`
  - 結果: `2 passed`
- `any_inventory`
  - `scripts/v460/lib/fill_record_helpers.py`: `any_type_debt_tokens=0`
  - `scripts/v460/lib/fill_loop_orchestrator.py`: `any_type_debt_tokens=0`

## 6. 次フェーズ（優先順）

1. `ztb/analysis/v4xx_unified_analyzer.py` / `ztb/analysis/promotion.py`  
   - git-lfs pointer 管理下の差分運用を整理したうえで、`_as_object_map` / `_as_float` の `safety` 統合を再適用。  
2. `ztb/training/reward_function_optimizer/reward_function_optimizer.py`  
   - `Any` debt 上位のため、result/config payload 型固定と evaluator 系 `TypedDict` の横展開を優先。  
3. `ztb/training/algorithms/sac/sac_algorithm.py`  
   - 学習ループ payload を型固定し、ログ/集計の重複分岐を helper 化。  
4. `ztb/utils/file_utils.py`  
   - `safe_json_load/dump` の戻り値契約を明確化し、`read_json_object/read_json_array` ベースの型付き helper を追加。  
5. `ztb/analysis/features/re_evaluate_features.py`  
   - 評価 result payload の型固定と集計ループ helper 化を進め、重複整形コードを削減。  
6. `ztb/training/utils/sac_utils.py`
   - `check-config` / `validate-data` の詳細 payload を `TypedDict` 化し、結果 JSON の契約を固定。  
7. `ztb/utils/run_metadata.py`
   - package hash 対象 package の allow-list 指定 (`--package-hash-target`) を追加し、hash 実行時の時間をさらに圧縮。  
8. `ztb/experiments/job_manager.py`
   - `parallel_backend=process` の強制終了戦略（job isolation + watchdog）を設計し、timeout 時のリソース占有をさらに低減。  
9. `ztb/trading/live/simulation/paper_trader.py`
   - `run_replay` 返却 payload と state snapshot を `TypedDict` 化し、残存 `Any` を段階削減。  
10. `scripts/v460/ml/train_sg_v3.py`
   - `scripts/v460` 内の残存 type debt 上位（`6`）を優先削減し、学習設定 payload 契約を固定。  

---

## 7. 運用ルール（品質維持）

1. 変更前後で `any_inventory.py` を実行し、`type_debt` の純減を確認する。  
2. CI では段階的に `--max-type-any` の上限を下げる。  
3. 1PR で触る領域は 2-4 ファイルに限定し、型統合と動作確認を同時に行う。  
4. 036 を正本とし、037-039 は統合済み参照スタブのみを維持する。  

---

## 8. リファクタリング候補（重複削減）

1. `scripts/v460/run_gate_check.py`  
   - `load_gate_thresholds().get(<gate>, {})` の同型処理を helper 化可能。  
2. `ztb/features/core/registry.py`  
   - `_as_bool` のローカル実装が複数設定処理で再利用されており、共通キャスト utility への統合余地あり。  
3. `ztb/training/unified_optimizer.py`  
   - `OptimizationResult` 周辺の `dict[str, object]` を段階的に TypedDict 化し、キー存在保証を強化可能。  
4. `ztb/trading/comprehensive_backtest.py`  
   - `RiskMetrics` 初期化の例外フォールバック依存を減らし、失敗時ログを追加すると診断性が上がる。  
5. `ztb/analysis/v4xx_unified_analyzer.py`  
   - module-level 互換ラッパー（旧 API）を段階的に縮退し、class method 実装へ一本化する余地あり。  
6. `ztb/trading/strategies/action_signal_guide/interfaces/common_types.py`  
   - marker 基底を軸に、必要箇所で protocol 化（read-only 属性契約）へ進める余地あり。  
7. `ztb/trading/strategies/action_signal_guide/pattern_recognition/candlestick_patterns.py`  
   - Step31で導入した 4つの family base は、今後 `shooting_star` / `dark_cloud_cover` / `harami` 追加時にそのまま水平展開可能。  
8. `ztb/trading/strategies/action_signal_guide/pattern_recognition/wave_counting.py`  
   - Step32 の `_WavePatternBase` 横展開は `fibonacci_patterns.py`（Step34）/ `harmonic_patterns.py`（Step35）へ適用済み。次候補は `gann_analysis.py` の共通抽出。  
9. `ztb/data/data_validation.py`  
   - `DataValidator` / `DataIntegrityChecker` の payload 生成規約が共通化済み。次段階で基底（check pipeline）へ抽出するとテスト容易性が上がる。  
10. `ztb/utils/config.py` / `ztb/utils/safety.py` / `ztb/io/json_io.py`  
   - config 値変換と JSON object/list 契約が整理されたため、次段階は `config_cast` / `json_contract` の共通 utility 化で責務分離を進める余地あり。  
11. `ztb/trading/production/*`  
   - state ファイルI/Oは `state_persistence` に統合済み。次段階は各コンポーネントの payload schema を `TypedDict` 化して復元時の契約不整合を抑止する余地あり。  
12. `ztb/training/callbacks/monitoring/metrics_collector.py`  
   - 現在導入した payload serialize/restore を `TypedDict` 化し、`Any` が残る `metrics`/`metadata` の復元契約を段階的に固定化可能。  
