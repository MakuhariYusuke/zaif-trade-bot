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

---

## 6. 次フェーズ（優先順）

1. `ztb/analysis/features/auto_feature_generator.py`  
   - 生成パイプラインの result payload を段階的に型固定し、feature registry 連携の重複マップ操作を整理。  
2. `ztb/evaluation/promotion.py`  
   - fallback 実装と `analysis/promotion` の責務境界を整理し、将来的な評価ロジック共通化（mixin/utility）に備える。  
3. `ztb/analysis/status.py`  
   - 運用 status payload（通知/集計）を型固定し、`dict` 合成の重複を削減。  
4. `ztb/data/data_validation.py`  
   - バリデーション結果 payload を `TypedDict` 化し、ルール評価の重複 map 操作を helper 化。  
5. `ztb/utils/config.py`  
   - config merge/cast 系の `Any` を段階的に削減し、既存型別名へ集約。  

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
