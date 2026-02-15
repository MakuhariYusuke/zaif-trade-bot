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

---

## 6. 次フェーズ（優先順）

1. `ztb/trading/environment/heavy_env/core.py`  
   - 環境状態 payload と feature 計算戻り値の `Any` を段階的に TypedDict/alias 化。  
2. `ztb/utils/env_metrics.py`  
   - metrics 集計 payload を `TypedDict` へ寄せ、動的 key 依存と `Any` を段階削減。  
3. `ztb/trading/strategies/action_signal_guide/pattern_recognition/candlestick_patterns.py`  
   - 返却 payload と中間解析データの `Any` を `TypedDict` / `ObjectMap` ベースへ寄せ、判定ロジックのキー契約を固定。  
4. `ztb/analysis/features/auto_feature_generator.py`  
   - 生成パイプラインの result payload を段階的に型固定し、feature registry 連携の重複マップ操作を整理。  

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
