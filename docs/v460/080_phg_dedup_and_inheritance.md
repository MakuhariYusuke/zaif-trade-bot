# 080# PHG: 重複排除 & 継承ベース統合

**日時**: 2025-02-24 (+ 2026-02-16 / 2026-02-20 追補)  
**コミット**: `63c557e2b` (phase1), `22733338b` (phase2), `3655e17d3` (phase7), `(current)` (phase8追補)  
**先行**: `6ed99506a` (dead file cleanup + type safety)

---

## 概要

ph2 fill_test データ収集中に実施可能なコード品質改善として、
重複排除 (deduplication) と継承ベース統合 (inheritance consolidation) を実施。

16カテゴリの重複パターンを特定し、安全に実施可能なものから優先順に処理。

## Phase 1: デッドコード削除 + re-export + LFS修正 (`63c557e2b`)

### デッドコード削除 (6ファイル, ~1,900行)

| ファイル | 行数 | 理由 |
|---|---|---|
| `ztb/trading/production/circuit_breaker.py` | 660 | 参照0件 |
| `ztb/trading/signal/enhanced_risk_manager.py` | — | 参照0件 |
| `ztb/trading/signal/quality_scorer_backup.py` | 286 | バックアップ、参照0件 |
| `ztb/analysis/regime/regime_evaluation.py` | 347 | deprecated、参照0件 |
| `ztb/trading/end_to_end_validator.py` | 584 | テスト以外の参照0件 |
| `tests/.../test_end_to_end_validator.py` | — | 上記の専用テスト |

### re-export シム変換 (3ファイル)

| ファイル | 変換前 | canonical import 先 |
|---|---|---|
| `ztb/trading/entry_system.py` | ダミースタブ | `ztb.trading.signal.entry_system` |
| `ztb/training/sell_mitigation_ppo_trainer.py` | ダミースタブ | `ztb.training.experiments.sell_mitigation_ppo_trainer` |
| `ztb/evaluation/auto_feature_generator.py` | ダミースタブ | `ztb.analysis.features.auto_feature_generator` |

### LFS 修正

`.gitattributes` に LFS 否定ルール追加:
- `ztb/analysis/**/*.py`, `ztb/evaluation/**/*.py` → text (LFS除外)
- `docs/**/*.md` → text (LFS除外)

## Phase 2: 継承ベース統合 (`22733338b`)

### P0: SELLBiasMitigationCallback

`sell_mitigation_ppo_trainer.py` 内のインラインクラス定義 (~70行) を削除し、
canonical `ztb.training.callbacks_lib.sell_mitigation_callback` からの import に統一。

壊れた `.pyi` スタブ (`sell_mitigation_ppo_trainer.pyi`) も削除。

### P1: CircuitBreaker 統合

- `ztb/risk/circuit_breakers.py` の自前 `CircuitBreaker`/`CircuitBreakerConfig`/`CircuitBreakerState`/`CircuitBreakerOpenError` 定義 (~130行) を削除
- `ztb/utils/circuit_breaker.py` (canonical) からの import に置換
- alias 提供: `CircuitBreakerState = CircuitState`, `CircuitBreakerOpenError = CircuitBreakerOpenException`
- `KillSwitch`, `CircuitBreakerRegistry`, factory 関数は risk 固有機能として残存
- 未使用 `circuit_breakers_compat.py` を削除
- `CircuitBreaker.__init__` の config を Optional 化 (API統一)

### P1: BaseTrainer → スキップ

2つの `BaseTrainer` は設計目的が異なる:
- `core/base_trainer.py`: `TrainerParams` + ABC + ConfigurableMixin (PPO/SAC用)
- `trainers/base_trainer.py`: `Dict[str, Any]` + plain class (Ensemble/Unified用)

コンストラクタのシグネチャ不一致のため、shim化するとダウンストリームが破壊される。

### P2: RegimeType → MarketRegime enum 統一

canonical: `ztb/analysis/regime/market_regime_types.py` の `MarketRegime(Enum)` (21メンバー)

| ファイル | 変換前 | 変換後 |
|---|---|---|
| `market_regime_classifier.py` | `RegimeType(Enum)` 21メンバー | `RegimeType = MarketRegime` |
| `v444_regime_classifier.py` | `RegimeType(Enum)` 21メンバー | `RegimeType = MarketRegime` |
| `v444_regime_analyzer.py` | `RegimeType(Enum)` 13メンバー | `RegimeType = MarketRegime` |

互換 alias 追加 (`market_regime_types.py`):
- `HIGH_VOLATILITY_RANGE` → `HIGH_VOLATILITY_RANGING`
- `MODERATE_VOLATILITY_RANGE` → `MODERATE_VOLATILITY_RANGING`
- `LOW_VOLATILITY_RANGE` → `LOW_VOLATILITY_RANGING`

### P3: signal/regime/classifier.py → スキップ

plain class (`class RegimeType:`) で値にも差異 (`_range` vs `_ranging`)。
文字列比較の破壊リスクが高いため、段階的移行として保留。

## Phase 3 追補: quality/indicators 継承整理 + 重複削減 (2026-02-16)

### 1) `BaseTechnicalIndicator` 基底強化

- `temporary_config()` を基底に導入し、サブクラスが設定の一時上書きを共通実装で利用できるよう統合。
- `on_config_updated()` フックを導入し、設定変更時の派生属性同期を継承側へ集約。
- `calculate()` は cache hit 時にもコピー返却へ変更し、呼び出し側更新でキャッシュ本体が破壊されるリスクを低減。

### 2) `AdaptiveIndicator` の重複/不具合修正

- `calculate()` / `calculate_adaptive()` の重複ロジックを `_calculate_with_regime()` へ統合。
- base indicator の計算結果 dict を直接更新しないよう修正し、regime metadata が base 側キャッシュへ混入する不具合可能性を解消。
- `temporary_config` 非対応の mock 指標向け fallback を追加し、既存テスト互換を維持。

### 3) 水平展開 (`RSIIndicator`, `MACDIndicator`)

- `on_config_updated()` を導入し、adaptive config 更新時に
  `periods` / `fast_period` / `slow_period` / `signal_period` が即時反映されるよう統一。
- `quality/indicators` 配下 (`base.py`, `rsi.py`, `macd.py`) を `Any=0` 化。

### 4) 検証

- `tests/unit/trading/signal/indicators/test_signal_indicators.py`
- `tests/unit/trading/signal/test_modular_indicators.py`
- 結果: `60 passed`

## Phase 4 追補: integrated_backtest_runner 集計最適化 + 重複削減 (2026-02-17)

### 1) 戦略アダプタの重複整理

- `_run_enhanced_backtest()` 内部のローカル `FunctionStrategyAdapter` を module-level `_FunctionStrategyAdapter` へ統合。
- 呼び出しごとにクラス定義される重複コストを削減。
- `signal` payload / `action` 文字列の両方を受ける後方互換アダプタへ拡張。

### 2) 集計・検証の計算重複削減

- `_aggregate_results()` を配列ベースへ再構成し、`mean/std` の再計算を排除。
- `_validate_statistically()` の二重ループを単一ループに統合し、returns 再計算を削減。
- `_calculate_returns_from_portfolio_values()` を `np.diff` ベクトル化し、ゼロ除算を `where` で安全処理。

### 3) 不具合可能性の解消

- `initial_capital` / `commission` 引数が実質無視される経路を修正し、`BacktestEngine` へ反映。
- ATR の取引ごと再計算を廃止し、イテレーション単位の事前計算へ変更。
- 空 `portfolio_values` 時の `[-1]` 参照と `n_iterations=0` の 0除算をガード。

### 4) 型安全

- `ztb/trading/backtest/integrated_backtest_runner.py` を `Any=0` 化 (`any_type_debt_tokens: 19 -> 0`)。
- `Mapping[str, object]` / `ObjectMap` / `TradeList` / `IterationList` ベースへ統一。

## テスト結果

- v460: 602 passed, 1 failed (xgboost 既知)
- CircuitBreaker: 35 passed
- リグレッション: なし

## 削減効果

| 項目 | Phase 1 | Phase 2 | 合計 |
|---|---|---|---|
| 削除ファイル | 6 | 2 | 8 |
| 削除行数 | ~2,633 | ~380 | ~3,000 |
| 重複定義の排除 | — | 5クラス/enum | 5 |

## 残存重複 (今後の課題)

- `signal/regime/classifier.py` の `RegimeType` plain class (P3)
- `MarketRegimeDetector` 内部クラス重複 (P3)
- `PPOTrainer` / `SACTrainer` archive版 (低優先)
- `RiskManager` 5箇所 (低優先)

## Phase 5 追補: JSON/state helper 水平展開 + metrics 安定化 (2026-02-20)

### 1) helper の共通化範囲拡張

- `ztb/io/state_persistence.py` を追加し、state JSON I/O の canonical helper を `ztb.io` に昇格。
- `ztb/trading/production/state_persistence.py` は互換ラッパーとして維持し、既存 import を壊さずに共通 helper へ委譲。

### 2) signal/training への水平展開

- `ztb/trading/signal/entry_system.py` の `save_state/load_state` を helper 統合。
- 正規化ロジックを `_normalize_action()` に抽出し、`process_signal` / `update_outcome` 重複を削減。
- `ztb/training/callbacks/monitoring/metrics_collector.py` の `_export_json` / `load_state` を helper 統合。

### 3) 不具合可能性の解消

- `metrics_collector` の latest cache 無効化漏れを修正（新規追加・cleanup・load 後）。
- `register_metric()` が `max_series_size` を反映しない不整合を修正（メモリ上限を実効化）。
- `get_performance_stats()` の `pool_size` 属性参照ミスを修正。
- `WeakRefRegistry` に `registry` property を追加し、統計取得時の属性不一致例外を回避。
- 危険な pooled object 再利用経路を撤去し、履歴 series 参照破壊リスクを排除。

## Phase 6 追補: io 契約の追加統合 + 復元耐障害性の改善 (2026-02-20)

### 1) training/trading の io 統合

- `ztb/training/components/regime_adaptive_trainer.py` の state I/O を `read_state_payload` / `write_state_payload` に統一。
- `ztb/trading/cost/venue_transaction_cost_manager.py` の config I/O を `read_json_object` / `write_json` に統一。

### 2) 例外契約と入力正規化

- `regime_adaptive_trainer` は復元 payload を型検証し、無効値をスキップする設計へ変更。
- `venue_transaction_cost_manager` は venue 名を lowercase 正規化し、lookup の取りこぼしを防止。
- 同 manager のロードは「不正レコード1件で全体失敗」から「不正だけスキップ」へ改善。

### 3) ops/features への水平展開

- `ztb/ops/health/performance_monitor.py` の履歴 I/O を helper 化し、壊れた履歴行を parser でスキップ。
- `ztb/features/feature_set_config.py` の config load/save を helper 化し、`open + json.load/dump` 重複を削減。

## Phase 7 追補: multi_task/meta callback 継承統合 + 不具合修正 (2026-02-20)

### 1) 継承導入による重複削減

- `ztb/training/callbacks/multi_task/multi_task_callbacks.py` に
  `_BaseFrequencyCallback`（`NoOpMemoryOptimizedCallback` 継承）を導入。
- `TaskBalancingCallback` / `SharedRepresentationCallback` /
  `TaskInterferenceCallback` の共通処理（frequency gating / logger / lifecycle no-op）を基底化。
- `ztb/training/callbacks/meta/meta_callbacks.py` に
  `_BaseMetaCallback`（`NoOpMemoryOptimizedCallback` 継承）を導入。
- `MAMLCallback` / `FewShotCallback` / `MetaAdaptationCallback` の同型処理を継承で共通化。

### 2) 既存不具合の解消

- `SharedRepresentationCallback` と `MetaAdaptationCallback` の
  `super().__init__` 未実行を修正（cache/frequency 初期化欠落を解消）。
- `MetaAdaptationCallback` の `adaptation_steps` / `stability_threshold` /
  `compute_frequency` 未設定不具合を修正。
- `TaskInterferenceCallback` に欠落していた
  `task_interference_scores` / `interference_events` を追加し、
  実行時 `AttributeError` リスクを除去。
- `TaskInterferenceCallback.on_epoch_end` に `logs is None` ガードと
  frequency gating を追加し、無効入力時の不安定挙動を抑制。

### 3) 型安全と保守性

- `multi_task_callbacks.py` / `meta_callbacks.py` の `Any` を全撤去し、
  `ObjectMap` + `object` ベースへ統一（両ファイル `any_type_debt_tokens=0`）。
- 変換・履歴更新 helper（`_as_float`, `_append_bounded` など）を追加し、
  callback 間の重複実装を削減。
- repo 全体 `any_type_debt_tokens` は `2,571 -> 2,537`（-34）。

## Phase 8 追補: distributed/performance 継承統合 + 並行実行安定化 (2026-02-20)

### 1) 継承導入で thread lifecycle を共通化

- `ztb/training/callbacks/distributed/threading_mixin.py` を追加し、`BackgroundThreadController` を導入。
- `DistributedCoordinator` / `WorkerPool` / `DistributedTrainingManager` を同基底継承へ変更。
- 背景 thread の start/join 重複を基底 API（`_start_background_thread`, `_join_background_thread`）へ集約。

### 2) distributed 実行系の不具合・性能改善

- `WorkerPool` の task ごと thread 生成を廃止し、`ThreadPoolExecutor(max_workers=num_workers)` へ置換（スレッド増殖抑制）。
- 先頭 worker 固定になっていた疑似 round-robin を修正し、実際のローテーション選択へ変更。
- dispatch callback の late-binding 問題（`task_info`/`worker` 取り違え）を `_on_task_done` 経由で解消。
- `result_queue` を bounded 化し、満杯時に古い結果を落とすことでメモリ増加を抑制。
- `DistributedWorker.send_task()` を task lock で直列化し、`heartbeat/sync_ack` 混入時の誤判定を解消。
- `DistributedTrainingManager` の同期 thread 起動タイミングを修正（`initialize` で即終了していた不具合を解消）。

### 3) memory monitor 契約の整合化

- `memory_optimizer.py` に `_ThreadSafeStatsBase` を導入し、`LRUCache` / `MemoryPool` / `MemoryMonitor` / `WeakRefRegistry` の統計計算重複を削減。
- `MemoryMonitor.get_memory_stats()` に `memory_pressure` を追加し、`real_time_monitor` / `distributed` 側の判定契約と整合。
- `MemoryMonitor.emergency_cleanup()` 互換 alias を追加し、`integration` 側の呼び出し不整合を解消（実装は `force_cleanup()` に統一）。

### 4) 型安全の進捗

- `ztb/training/callbacks/distributed/worker.py` / `integration.py` / `threading_mixin.py` / `ztb/training/callbacks/performance/memory_optimizer.py` を `Any=0` 化。
- repo 全体 `any_type_debt_tokens` は `2,537 -> 2,502`（-35）。

## Phase 9 追補: callback helper 横展開 + coordinator 分岐統合 (2026-02-20)

### 1) callback 共通ヘルパの抽出

- `ztb/training/callbacks/shared/utils/value_utils.py` を追加し、
  `as_optional_float` / `as_optional_array` / `append_bounded` を共通実装化。
- `supervised/sac/transfer/unsupervised/meta/multi_task` の6モジュールで、
  重複していた helper 実装本体を共通化（モジュール側は薄い wrapper のみ維持）。

### 2) distributed coordinator の重複削減 + 不具合低減

- `DistributedCoordinator` に worker 状態アクセサ helper 群を追加し、
  `status/metrics/last_heartbeat` の dict/dataclass 分岐重複を集約。
- `register_worker` で入力 payload を正規化し、`worker_id` 不正値や
  属性欠落での登録不整合を早期 reject するよう改善。
- `_handle_error` / `_heartbeat_loop` を helper 経由へ統一し、
  worker 表現差異時の `status`/heartbeat 更新崩れリスクを解消。

### 3) その他改善（I/O）

- `ztb/training/unified_optimizer.py` の `safe_json_dump` 呼び出しで
  重複していた `open(..., 'w')` 包装を削除し、不要ファイル I/O を削減。

### 4) 検証

- `py_compile`（9ファイル）通過。
- 本環境は `pytest`/依存不足のため、ユニットテストは未実施。

## Phase 10 追補: experiments/results/optimization の JSON 統合 + 不具合修正 (2026-02-21)

### 1) JSON I/O の分散実装を統合

- `ztb/experiments/base.py` の集約/保存経路を `read_json_object` / `write_json` へ統一。
- `ztb/utils/results_utils.py` の読込経路を `read_json_object` に統一し、`json.load` 直書きを削減。
- `ztb/training/run_optimization.py` の config/result 読み書きを `read_json_object` / `write_json` に統一。

### 2) 重複削減と保守性向上

- `run_optimization.py` に `_write_temp_config` / `_run_unified_trainer` を追加し、
  一時 JSON 作成 + subprocess 実行の重複コードを共通化。
- `experiments/base.py` で run metadata 文字列化を `_serialize_run_metadata()` に抽出し、
  成功/失敗経路の重複を削減。
- `results_utils.py` へ `TrainingResultsPayload` / `BacktestMetricsPayload` (`TypedDict`) を導入し、
  payload 契約を明示。

### 3) 不具合可能性の解消

- `run_optimization.py` の `DataLoader` import インデント崩れを修正（実行時不整合リスク解消）。
- `optimize_hyperparameters()` の `self.config.copy()` 起因 trial 間設定汚染を
  `deepcopy` 化で解消。
- `results_utils.py` の保存処理で `safe_json_dump` 失敗を見逃す経路を修正し、
  失敗時は `OSError` を送出するよう改善。

### 4) 型安全と在庫

- `ztb/experiments/base.py` / `ztb/utils/results_utils.py` /
  `ztb/training/run_optimization.py` は `Any=0` を維持。
- repo 全体 `any_type_debt_tokens`: `2,516 -> 2,494`（-22）。

### 5) 検証

- `py_compile`（`base.py`, `results_utils.py`, `run_optimization.py`）通過。
- 本環境は `pytest`/依存不足のため、ユニットテストは未実施。

## Phase 11 追補: 横断課題探索（不具合・重複・性能）(2026-02-21)

### 1) P0 不具合候補

- `ztb/training/utils/sac_utils.py` は `py_compile` で構文エラー（未閉鎖 docstring）を確認し、
  現状 import 不可。
- 同ファイルで `self.project_root` 未初期化参照と `config_dir` / `data_dir` 名称不整合を確認し、
  実行時 `NameError` リスクを特定。

### 2) P1 実行安定性・性能候補

- `ztb/experiments/job_manager.py` は `ProcessPoolExecutor` で bound method +
  任意 `train_function` を渡す設計のため、環境依存で pickling 失敗しやすい。
- timeout 経路で future のキャンセル/停止制御がなく、
  timeout 扱い後に worker が継続実行して結果を上書きする競合リスクがある。
- `ztb/utils/run_metadata.py` は package hash 取得で site-packages を再帰全走査するため、
  メタデータ収集の高コスト要因になっている。

### 3) P1-P2 型安全候補（Any debt 上位）

- `ztb/training/algorithms/sac/sac_algorithm.py` (`type_debt=19`)
- `ztb/training/reward_function_optimizer/reward_function_optimizer.py` (`type_debt=19`)
- `ztb/training/checkpoint/checkpoint_manager.py` (`type_debt=18`)
- `ztb/training/core/config_builder.py` は `UnifiedConfig = Dict[str, Any]` と
  `get_config_value -> Any` が下流へ型曖昧性を伝播。

### 4) 次アクション指針

- 先に `sac_utils` を復旧（構文/初期化/責務分離）してから、
  `job_manager` の timeout/cancel 契約と並列実行設計を修正する順序が安全。

## Phase 12 追補: 既存 safety util への coercion 統合 (2026-02-21)

### 1) util 抽出の水平展開

- `ztb/utils/safety.py` の `ensure_dict` / `safe_to_float` を
  `dict/float` coercion の canonical helper として再利用。
- `ztb/training/run_optimization.py` の `_as_object_map` / `_as_float` を
  上記 helper へ委譲。
- `ztb/experiments/job_manager.py` の `_as_object_map` / `_as_float` を
  上記 helper へ委譲。
- `ztb/experiments/run_sac_experiments.py` の `_as_object_map` / `_as_float` を
  上記 helper へ委譲。
- `ztb/utils/run_manifest.py` の `_as_object_map` を `ensure_dict` 委譲へ置換。

### 2) 効果

- 4モジュールで重複していた coercion 実装を既存 util に統合し、
  振る舞い修正時の変更点を `safety` 側へ集約可能な構造に整理。
- 対象ファイルは `Any=0` を維持。

### 3) 検証

- `py_compile`（`run_optimization.py`, `job_manager.py`, `run_sac_experiments.py`, `run_manifest.py`）通過。
- `any_inventory`: repo 全体 `any_type_debt_tokens=2,501`（`scanned_files=1,289`、同時進行差分による母数変動あり）。

## Phase 13 追補: SAC utility 復旧 + git helper 抽出 + metadata 軽量化 (2026-02-22)

### 1) `sac_utils` の復旧と出力/走査負荷の抑制

- `ztb/training/utils/sac_utils.py` の構文崩れ/初期化不整合を解消し、CLI サブコマンド群を実行可能状態へ復旧。
- `clean_project_files()` に scan timeout (`max_scan_seconds`) を導入し、大規模 tree 走査の時間上限を明確化。
- `fix_common_issues()` に `max_files` を導入し、不要な全走査を抑止。
- `check_config_consistency()` に `max_details` を導入し、巨大 report 出力の I/O コストを削減。

### 2) `run_metadata` の高コスト経路を縮退

- `ztb/utils/run_metadata.py` の package hash 計算を opt-in (`--include-package-hashes`) 化。
- 従来の site-packages 全再帰を廃止し、distribution 配下 path + file stat ベース hash に変更。
- JSON I/O を `read_json_object` / `write_json` に統一し、I/O 契約を `ztb/io/json_io.py` へ集約。
- direct script 実行時の import 失敗を避ける path fallback を追加。

### 3) git 情報取得の重複統合（横展開）

- `ztb/utils/git_utils.py` を追加し、git SHA/branch/dirty/status/remote 取得を共通 helper 化。
- git-lfs 不在環境で失敗しにくい `git -c filter.lfs.*` 設定を helper 側へ集約。
- `ztb/utils/run_manifest.py` の git 取得実装を同 helper 委譲へ置換。
- `run_metadata` も同 helper へ寄せ、重複 subprocess 実装を削減。

### 4) 型安全・検証

- `any_inventory`（変更対象）:
  - `ztb/training/utils/sac_utils.py`: `any_type_debt_tokens=0`
  - `ztb/utils/run_metadata.py`: `any_type_debt_tokens=0`
  - `ztb/utils/run_manifest.py`: `any_type_debt_tokens=0`
  - `ztb/utils/git_utils.py`: `any_type_debt_tokens=0`
- `py_compile`:
  - `ztb/training/utils/sac_utils.py`
  - `ztb/utils/run_metadata.py`
  - `ztb/utils/run_manifest.py`
  - `ztb/utils/git_utils.py`

## Phase 14 追補: job_manager の timeout/cancel 競合対策 + 並列安定化 (2026-02-22)

### 1) 競合の根本原因

- 旧 `run_all_jobs()` は timeout 判定を親側で行う一方、
  worker 側 `execute_job()` が output/manifest/state を直接更新していたため、
  timeout 後の遅延完了で `timeout -> completed` 上書きが起き得た。
- `ProcessPoolExecutor + bound method` 実行で、環境依存の pickling 失敗リスクが高かった。

### 2) 実装改善（重複削減 + 不具合排除）

- worker 処理を副作用なしの `_execute_training_job()` に分離し、
  永続化を `_finalize_job()`（親側）へ一本化。
- `execute_job()` / parallel 実行の終了処理を `_normalize_job_result()` / `_finalize_job()` に集約し、
  status/manifest/state 更新の重複を削減。
- `run_all_jobs()` に `parallel_backend` を追加し、既定を `thread` に変更
  （local callable を含む実行で pickling 依存を低減）。
- scheduler を `wait(FIRST_COMPLETED)` ループへ置換し、
  timeout 判定済み job は親側で即 `timeout` 確定して遅延完了結果を無視。
- `executor.shutdown(wait=False, cancel_futures=True)` で、
  timeout job 待機による scheduler 停滞を回避。
- polling 間隔を `timeout_seconds` 連動の動的値へ変更し、小さい timeout でも追従性を確保。

### 3) 水平展開（既存 helper 活用）

- `job_manager._get_code_hash()` の git 取得を `ztb/utils/git_utils.py` へ統合し、
  git subprocess 実装の重複を削減。

### 4) 検証

- `py_compile`:
  - `ztb/experiments/job_manager.py`
  - `tests/unit/experiments/test_job_manager.py`
- 追加テスト:
  - `tests/unit/experiments/test_job_manager.py`
    - default backend（thread）で local callable が実行可能
    - timeout 後に遅延完了しても `status=timeout` が上書きされない
- `pytest` は実行環境に未導入のため未実施（`pytest: command not found`）。

## Phase 15 追補: streaming helper 統合 + config/checkpoint 型固定 + replay metadata 正常化 (2026-02-22)

### 1) helper 横展開（重複削減）

- `ztb/trading/strategies/action_signal_guide/realtime_adaptation/streaming_processor.py`
  の `_as_object_map/_as_float` を `safety` helper 委譲へ統一。
- `ztb/analysis/v4xx_unified_analyzer.py` / `ztb/analysis/promotion.py` については、
  現在 git-lfs pointer 管理の差分運用で全体置換差分が発生するため、運用整理後に再適用する方針。

### 2) `config_builder` の型契約整理

- `ztb/training/core/config_builder.py` の `Any` 注釈を撤去し、
  `ConfigMap` + generic default (`TypeVar`) ベースの getter 契約へ移行。
- section 取得値に `ensure_dict()` を適用し、非dict混入時の取得不整合を解消。
- `UnifiedConfig` を `dict[str, object]` へ縮退して `Any` 伝播を抑止。

### 3) `checkpoint_manager` の payload/schema 型固定

- `ztb/training/checkpoint/checkpoint_manager.py` に
  `CheckpointPayload` / `CheckpointMetadata` / `RNGStatePayload` /
  `CheckpointValidationResult` を導入。
- `BaseAlgorithm = Any` fallback を `Protocol` へ置換し、
  runtime import 回避を維持しつつ `Any` alias を排除。
- `_build_payload()` の policy state 収集を防御的実装へ更新し、
  state 取得失敗時の復旧不能リスクを低減。

### 4) 追加の機能改善（運用信頼性）

- `ztb/trading/live/simulation/paper_trader.py` の replay metadata を
  dummy JSON から `RunMetadata.capture_all_metadata()` + `save_to_file()` へ置換し、
  実運用に近い実行メタデータを保存するよう改善。

### 5) 検証

- `py_compile`:
  - `ztb/trading/strategies/action_signal_guide/realtime_adaptation/streaming_processor.py`
  - `ztb/training/core/config_builder.py`
  - `ztb/training/checkpoint/checkpoint_manager.py`
  - `ztb/trading/live/simulation/paper_trader.py`
- `any_inventory`（対象）:
  - `streaming_processor.py` / `config_builder.py` / `checkpoint_manager.py` は `Any=0`

## Phase 16 追補: retrain_scheduler 重複統合 + 評価高速化 + LFS汚染ガード (2026-02-22)

### 1) 重複削減（helper 統合）

- `scripts/v460/ml/retrain_scheduler.py` に
  `_extract_numeric_column()` / `_compute_skip_metrics()` を追加し、
  single/multi の skip 評価ロジックを共通化。
- history JSONL 追記を `_append_jsonl_record()` に統一し、
  scheduler / one-shot / side-specific の重複 I/O 実装を削減。

### 2) 不具合可能性の低減

- `multi-window` 評価で `X_val` が小さい/空のケースに対し、
  early-stopping 用 transform を必要時のみ実行するよう変更。
- `_safe_import_ztb_module()` に spec/loader の fail-fast 検証を追加し、
  import 異常時の原因特定性を向上。

### 3) 性能改善

- WF評価で PnL 列をループ外で一括抽出し、window 内の反復 `DataFrame.loc` を削減。
- 前処理中間 `DataFrame` の再構築を減らし、配列ベース処理へ寄せてオーバーヘッドを圧縮。

### 4) 型安全 + LFS運用改善

- `retrain_scheduler.py` の `Any` を全撤去し、`ConfigMap` / `object` ベースへ移行（対象 `Any=0`）。
- `.gitattributes` へ `*.py` / `*.pyi` の非LFS override を明示し、
  source code の LFS pointer 汚染を抑止。

### 5) 検証

- `py_compile`:
  - `scripts/v460/ml/retrain_scheduler.py`
- `any_inventory`:
  - `--roots scripts/v460/ml`: `retrain_scheduler.py` は `Any=0`
  - repo 全体 `any_type_debt_tokens=2,465`（`scanned_files=1,302`）

## Phase 17 追補: reward optimizer クラス統合 + Any削減 + git/LFS 複雑化点検 (2026-02-22)

### 1) クラス群改善（継承/責務整理・重複削減）

- `ztb/training/reward_function_optimizer/reward_function_optimizer.py`
  の巨大 `_define_parameter_spaces()` を削減し、
  `RewardFunctionParameterSpace` へ委譲する構成へ統一。
- 同時に `create_parameter_space_from_config()` を optimizer 本体に復元し、
  `optimize_from_config_file()` / `optimize_hyperparameters_from_config()` からの呼び出し不整合
  （実行時 `AttributeError` リスク）を解消。
- `OptimizationEngine` に `sample_parameters_for_trial()` を導入し、
  3 箇所に散っていた Optuna parameter sampling 重複を統合。

### 2) 不具合可能性の解消

- `RewardFunctionParameterSpace.create_parameter_space_from_config()` で
  `bool` 値が `int` として探索対象に混入する問題を除外。
- 同メソッドで int 境界の丸め後逆転 (`high < low`) を補正し、
  不正探索空間生成を防止。
- `SACAlgorithm` の transfer/compression/explainability まわりに
  `safe_to_float/safe_to_int` を拡張適用し、設定値の型揺れ起因の実行時例外リスクを低減。
- `EvaluationEngine` の比較ソート・t検定結果を数値正規化し、
  NaN/非数入力時の不安定挙動を緩和。

### 3) Any削減（対象領域の完了）

- 以下を `Any=0` 化:
  - `ztb/training/reward_function_optimizer/reward_function_optimizer.py`
  - `ztb/training/reward_function_optimizer/parameter_space.py`
  - `ztb/training/reward_function_optimizer/display_manager.py`
  - `ztb/training/reward_function_optimizer/components/optimization_engine.py`
  - `ztb/training/reward_function_optimizer/components/evaluation_engine.py`
  - `ztb/training/algorithms/sac/sac_algorithm.py`
- `any_inventory`（`--roots ztb/training/algorithms/sac ztb/training/reward_function_optimizer`）:
  `any_type_debt_tokens = 0`

### 4) git 複雑化の点検結果（LFS）

- 現在の repo は `filter.lfs.required=true` かつ `git-lfs` バイナリ未導入のため、
  `git status` など全体走査系コマンドが失敗する状態を確認。
- `.gitattributes` で `*.csv` を含む広範囲が LFS 対象のため、
  `assets/images/latest_training_rewards.csv` の clean filter 失敗が顕在化。
- 本作業では source 側変更に限定し、LFS 管理対象ファイルには非介入。

### 5) 検証

- `py_compile`:
  - `ztb/training/reward_function_optimizer/reward_function_optimizer.py`
  - `ztb/training/reward_function_optimizer/parameter_space.py`
  - `ztb/training/reward_function_optimizer/display_manager.py`
  - `ztb/training/reward_function_optimizer/components/optimization_engine.py`
  - `ztb/training/reward_function_optimizer/components/evaluation_engine.py`
  - `ztb/training/algorithms/sac/sac_algorithm.py`
- `pytest` は実行環境に未導入 (`No module named pytest`) のため未実施。
- ランタイム smoke は `numpy` 未導入 (`No module named numpy`) のため限定的。

## Phase 18 追補: reward optimizer 評価経路リファクタ + レポート不具合修正 (2026-02-22)

### 1) 重複削減（保守性向上）

- `ztb/training/reward_function_optimizer/reward_function_optimizer.py` に
  `_evaluate_reward_params()` を導入し、
  `optimize_from_config_file` / `optimize_adaptive` / robust fallback の
  `create_backtest_config + run_backtest_evaluation` 重複経路を統一。
- `create_backtest_config()` の reward settings 組み立てを
  scalar key loop + helper へ整理し、分岐重複を削減。
- `run_backtest_evaluation()` の入力抽出を
  `_extract_reward_inputs_from_settings()` に抽出し、
  multipliers/weight の取り出し重複を削減。

### 2) 不具合可能性の解消

- `_print_scores()` で `format='d'` に float が入ると例外になり得る問題を修正。
- `optimize_adaptive()` の `Best Score` 表示で、
  `'N/A'` 文字列に `:.4f` が適用され得る問題を修正。
- `generate_optimization_report()` の `Study Best Value` も同様に
  safe float へ正規化してから整形するよう修正。
- `run_backtest_evaluation()` は `reward_settings` を `ensure_dict()` で受け、
  setting 型揺れ時の実行時例外リスクを低減。

### 3) 検証

- `py_compile`:
  - `ztb/training/reward_function_optimizer/reward_function_optimizer.py`
- venv test:
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reward/test_reward_optimization.py`
  - 結果: `5 passed`（warning 1）

## Phase 19 追補: SACハイパーパラメータ反映不具合修正 + 設定ブリッジ統合 (2026-02-22)

### 1) 不具合修正（評価実効性）

- `ztb/training/reward_function_optimizer/reward_function_optimizer.py` で
  `optimize_hyperparameters_from_config()` が探索する SAC 値
  (`learning_rate` / `batch_size` / `buffer_size` / `gamma` / `tau` / `ent_coef` / `reward_scale`)
  を `create_backtest_config()` に確実反映する経路を追加。
- これにより、従来「SAC探索しても synthetic 評価にほぼ効かない」状態を解消。
- `_update_dynamic_weights_from_history()` の drawdown 判定符号を修正し、
  正の drawdown 入力で常時 `low risk` になる誤判定を解消。

### 2) 重複削減・保守性向上

- パラメータ分類と設定反映を helper 化:
  - `_split_sac_and_reward_params()`
  - `_extract_reward_settings_from_params()`
  - `_apply_sac_hyperparameters()`
  - `_extract_sac_inputs_from_config()`
  - `_compute_sac_adjustment_factors()`
- `create_backtest_config()` は base config を保持しつつ差分適用する形へ整理し、
  SAC 更新時に既存 `reward_settings` が失われるリスクを回避。
- `optimize_hyperparameters_from_config()` で
  固定 reward 設定の `base_backtest_config` を再利用し、
  trial ごとの不要な設定再構築を削減（軽量化）。
- config 側に数値 SAC パラメータが無いケースでは
  `DEFAULT_SYNTHETIC_SAC_HYPERPARAMETERS` へフォールバックし、
  「探索次元 0 で最適化が実質無効化される」ケースを回避。
- `optimize_reward_function()` 完了時に
  `_update_dynamic_weights_from_history()` を連携し、
  history 依存ロジックの死蔵を防止（次回最適化への反映を有効化）。

### 3) 性能/品質改善

- `run_backtest_evaluation()` に SAC 品質係数を導入し、
  reward 設定に加えて SAC 設定差分もスコアへ反映。
- `total_trades` を `max(0, ...)` へ変更し、低品質設定時の負値混入を防止。

### 4) 追加テスト（回帰防止）

- 新規: `tests/unit/reward/test_reward_optimizer_sac_bridge.py`
  - base config 利用時に reward 設定を保持したまま SAC 値を更新できること
  - SAC 設定品質差で `profit/max_drawdown` が変化すること

### 5) 検証

- `py_compile`:
  - `ztb/training/reward_function_optimizer/reward_function_optimizer.py`
  - `tests/unit/reward/test_reward_optimizer_sac_bridge.py`
- venv test:
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reward/test_reward_optimizer_sac_bridge.py`
    - 結果: `2 passed`（warning 1）
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reward/test_reward_optimization.py`
    - 結果: `5 passed`（warning 1）

## Phase 20 追補: 大規模Git整理（履歴可読性 + EOL再発防止） (2026-02-22)

### 1) コミット整理（用途別分離）

- 大量差分（`1099 files`）を以下 2 つへ分離コミットして履歴の追跡性を改善:
  - `40eacd2f3`: docs/analysis artifact バッチ
  - `b060005ba`: source/tests/config バッチ

### 2) blame ノイズ低減

- `.git-blame-ignore-revs` を追加し、
  大規模 snapshot commit（`40eacd2f3`, `b060005ba`）を blame 対象から除外可能化。

### 3) EOL/改行の再発防止

- `.gitattributes` に LF 強制ルールを追記:
  - `*.py`, `*.pyi`, `*.md`, `*.rst`, `*.yaml`, `*.yml`, `*.toml`, `*.sh`, `docs/**/*.md`
- 目的: Windows 環境での CRLF 混入による「全体差分化」再発の抑止。

### 4) リポジトリ保守

- `git gc --prune=now` 実行で pack サイズを圧縮:
  - `size-pack: 241.47 MiB -> 47.18 MiB`
- `git fsck --full` 実行（dangling object はあるが破損なし）。
- 差分汚染の再発源 3 ファイルを tracking から除外:
  - `test_results.json`
  - `assets/images/latest_training_rewards.csv`
  - `data/performance/performance_history.json`
  - `.gitignore` へ明示し、`git rm --cached` で index からのみ削除（ローカル実体は保持）。

## Phase 21 追補: dynamic weight 更新ロジックの重複整理 + 型揺れ耐性強化 (2026-02-23)

### 1) 重複削減（保守性向上）

- `ztb/training/reward_function_optimizer/reward_function_optimizer.py` に
  動的重み更新向け helper を追加:
  - `_extract_numeric_metric()`
  - `_classify_risk_level()`
  - `_classify_market_regime_from_win_rate()`
- `_update_dynamic_weights_from_history()` と `update_dynamic_weights()` の
  risk 判定ロジックを helper 経由に統一し、閾値の散在を解消。

### 2) 不具合可能性の解消

- `_update_dynamic_weights_from_history()` で
  history score が文字列などの型揺れを含む場合も `safe_to_float` 経由で扱うよう変更し、
  `sum()` 時の型エラーリスクを除去。
- `_print_scores()` はカテゴリ値を数値正規化してから色判定/整形するよう修正し、
  非数値入力混在時でも表示処理が落ちないよう改善。

### 3) 検証

- 新規テスト:
  - `tests/unit/reward/test_reward_optimizer_dynamic_weights.py`
    - history 文字列スコアの正規化と dynamic weight 更新
    - risk level 閾値境界（0.05/0.15）
    - `_print_scores()` の非数値耐性
- `py_compile`:
  - `ztb/training/reward_function_optimizer/reward_function_optimizer.py`
  - `tests/unit/reward/test_reward_optimizer_dynamic_weights.py`
- venv test:
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reward/test_reward_optimizer_dynamic_weights.py`
    - 結果: `3 passed`（warning 1）
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reward/test_reward_optimization.py`
    - 結果: `5 passed`（warning 1）

## Phase 22 追補: candidate evaluator の報告集計健全化 + 重複処理整理 (2026-02-23)

### 1) 不具合可能性の解消

- `ztb/training/reward_function_optimizer/candidate_evaluator.py` で
  実行前の report baseline（`mtime_ns + size`）をスナップショットし、
  実行後は「今回新規/更新された report」のみを集計対象に変更。
- これにより、同一 `model_name` の過去 report 混入で評価値が歪むリスクを低減。
- retry 時の partial cleanup も baseline 差分だけ削除するよう変更し、
  既存 report の誤削除を回避。

### 2) 重複削減・型安全

- `safe` helper へ統合:
  - `_safe_float` の重複実装を廃止し `safe_to_float` を利用
  - JSON 読み込みは `safe_open_json` + `ensure_dict` 経由へ統一
- report 収集/スナップショット/差分抽出を helper 化:
  - `_find_reports_for_model()`
  - `_snapshot_report_state()`
  - `_is_new_or_updated_report()`
  - `_collect_current_run_reports()`

### 3) テスト整備（回帰防止）

- `tests/unit/training/reward_function_optimizer/test_candidate_evaluator.py` を再整理:
  - dry-run / report parsing / retry / timeout / missing model_name
  - pre-existing report 保持と current-run のみ集計されること
  - retry cleanup が新規 partial のみ削除すること
- 既存テスト内のモック不備（retry時 returncode 取り扱い、monkeypatch 漏れ）も修正。

### 4) 検証

- `py_compile`:
  - `ztb/training/reward_function_optimizer/candidate_evaluator.py`
  - `tests/unit/training/reward_function_optimizer/test_candidate_evaluator.py`
- venv test:
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/training/reward_function_optimizer/test_candidate_evaluator.py`
    - 結果: `6 passed`（warning 1）
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/training/reward_function_optimizer/test_mtf_optimizer.py tests/unit/training/reward_function_optimizer/test_mtf_optimizer_score_report.py`
    - 結果: `6 passed`（warning 1）

## Phase 23 追補: report catalog 高速化（キャッシュ）+ メモリ上限制御 (2026-02-23)

### 1) 実行時間短縮

- `ztb/reporting/services/catalog.py` に report model-name キャッシュを導入:
  - key: `(resolved_path, mtime_ns, size)`
  - 変更なしファイルは JSON 再読み込みを回避。
- `find_reports_for_model()` は cache 経由で一致判定し、反復呼び出し時の
  `training_report_*.json` 全件 parse コストを削減。
- `candidate_evaluator` 側の report 探索も
  catalog の `find_reports_for_model()` へ委譲し、改善効果を水平展開。

### 2) メモリリーク防止

- キャッシュは `REPORT_MODEL_NAME_CACHE_MAX_SIZE=2048` の上限付き
  `OrderedDict` で管理し、古いエントリを自動evict。
- `clear_report_cache()` を追加し、長時間実行プロセスで
  明示解放できるようにした（`ztb/utils/report_utils.py` 経由でも公開）。

### 3) 追加の健全化

- `extract_action_distribution()` は `safe_to_float` 正規化を通し、
  文字列/不正値混入時の型揺れを抑制。
- `get_latest_report_for_model()` は辞書順ではなく
  `mtime_ns` 基準で最新 report を返すよう修正。

### 4) テスト

- 新規: `tests/unit/reporting/services/test_catalog.py`
  - cache ヒットで再parseされないこと
  - report 更新時に cache が更新されること
  - latest report が mtime 基準で選ばれること
  - action distribution の数値正規化
  - cache 上限制御
- 既存回帰:
  - `tests/unit/training/reward_function_optimizer/test_candidate_evaluator.py`
  - `tests/unit/training/reward_function_optimizer/test_mtf_optimizer.py`
  - `tests/unit/training/reward_function_optimizer/test_mtf_optimizer_score_report.py`

### 5) 検証

- `py_compile`:
  - `ztb/reporting/services/catalog.py`
  - `ztb/utils/report_utils.py`
  - `ztb/training/reward_function_optimizer/candidate_evaluator.py`
  - `tests/unit/reporting/services/test_catalog.py`
- venv test:
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reporting/services/test_catalog.py`
    - 結果: `5 passed`（warning 1）
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/training/reward_function_optimizer/test_candidate_evaluator.py tests/unit/training/reward_function_optimizer/test_mtf_optimizer.py tests/unit/training/reward_function_optimizer/test_mtf_optimizer_score_report.py`
    - 結果: `12 passed`（warning 1）

## Phase 24 追補: catalog cache stale-key 抑止 + lookup 軽量化 (2026-02-23)

### 1) メモリリーク防止の追加対策

- `ztb/reporting/services/catalog.py` の model-name cache で、
  同一 report path の旧キー（旧 mtime/size）を新規格納前に削除する処理を追加。
- これにより、同一ファイルが頻繁更新される運用でも
  stale key が蓄積し続ける状態を抑止。

### 2) 実行時間短縮

- `find_reports_for_model()` の `sorted(glob(...))` を外し、
  大量 report 走査時の不要ソートコストを削減。
- 最新選択は `get_latest_report_for_model()` 側で `mtime_ns` 基準判定を維持。

### 3) テスト

- `tests/unit/reporting/services/test_catalog.py` を拡張:
  - `test_cache_does_not_accumulate_stale_entries_for_same_path`
  - 既存 5 テストと合わせて `6 passed`

### 4) 検証

- `py_compile`:
  - `ztb/reporting/services/catalog.py`
  - `tests/unit/reporting/services/test_catalog.py`
- venv test:
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reporting/services/test_catalog.py`
    - 結果: `6 passed`（warning 1）
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/training/reward_function_optimizer/test_candidate_evaluator.py tests/unit/training/reward_function_optimizer/test_mtf_optimizer.py tests/unit/training/reward_function_optimizer/test_mtf_optimizer_score_report.py`
    - 結果: `12 passed`（warning 1）

## Phase 25 追補: reward optimizer cache の省メモリ化 + eviction検証 (2026-02-23)

### 1) メモリ節約

- `ztb/training/reward_function_optimizer/reward_function_optimizer.py` の
  `_build_evaluation_cache_key()` を、巨大 JSON 文字列キー保持から
  `SHA-1(40文字)` ハッシュキーへ変更。
- 効果: 評価キャッシュ (`evaluation_cache`) の key メモリ使用量を削減し、
  長時間最適化時のメモリ圧迫を抑制。

### 2) 品質担保（回帰防止）

- 新規テスト: `tests/unit/reward/test_reward_optimizer_cache.py`
  - cache key が順序非依存で安定して生成されること
  - パラメータ差分で key が変わること
  - `_store_evaluation_cache()` が `max_evaluation_cache_size` 超過時に
    oldest を eviction すること

### 3) 検証

- `py_compile`:
  - `ztb/training/reward_function_optimizer/reward_function_optimizer.py`
  - `tests/unit/reward/test_reward_optimizer_cache.py`
- venv test:
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reward/test_reward_optimizer_cache.py tests/unit/reward/test_reward_optimizer_dynamic_weights.py tests/unit/reward/test_reward_optimizer_sac_bridge.py`
    - 結果: `7 passed`（warning 1）
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reward/test_reward_optimization.py`
    - 結果: `5 passed`（warning 1）

## Phase 26 追補: report helper の水平展開 + tools 集計重複削減 (2026-02-22)

### 1) report helper の共通化拡張（catalog）

- `ztb/reporting/services/catalog.py` に以下を追加:
  - `list_training_reports()`
  - `get_recent_training_reports()`
  - `load_training_report()`
  - `extract_action_distribution_from_payload()`
  - `extract_reward_components_from_payload()`
  - `extract_reward_components()`
- 既存の model_name 抽出処理は `load_training_report()` 経由へ寄せ、
  JSON load + dict化の重複を削減。
- `ztb/utils/report_utils.py` の re-export を拡張し、既存 import 互換を維持。

### 2) 水平展開（tools）

- `tools/check_recent_reports.py`
  - report 走査/JSON load/分布抽出を catalog helper 経由へ統一。
- `tools/analyze_recent_reports.py`
  - `glob + open + json.load` を helper 経由へ置換。
  - `RecentReportAnalysis` (`TypedDict`) を導入し、`Any` 依存を削減。
  - report 0件時の統計計算で落ちる経路をガード。
- `tools/monitor_ab_progress.py`
  - report 列挙/JSON load/action_distribution 抽出を helper 経由へ統一。
- `tools/run_ab_searches.py`
  - reward/action 抽出を helper へ統合し、重複 JSON パースを削減。
  - `read_json_object` / `write_json` へ I/O を統一。
- `tools/ci/evaluate_training_runs.py`
  - helper 統合 + `TypedDict` 導入で型を明確化。
  - summary 表示の不整合（group数を report数扱い、`sharpe_ratio` 未参照）を修正。

### 3) 性能・保守性観点の効果

- 最近N件取得を `heapq.nlargest` ベースへ統一し、
  全件ソートの常時コストを削減。
- report payload の安全ロード窓口を1箇所へ集約し、
  不正JSON/型揺れ時の扱いを共通化。
- 同種の report 解析スクリプト間での重複実装を削減し、
  今後の仕様変更時の修正箇所を縮小。

### 4) 検証

- `py_compile`:
  - `ztb/reporting/services/catalog.py`
  - `ztb/utils/report_utils.py`
  - `tools/check_recent_reports.py`
  - `tools/analyze_recent_reports.py`
  - `tools/monitor_ab_progress.py`
  - `tools/run_ab_searches.py`
  - `tools/ci/evaluate_training_runs.py`
  - `tests/unit/reporting/services/test_catalog.py`
- venv test:
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reporting/services/test_catalog.py`
    - 結果: `9 passed`（warning 1）

## Phase 27 追補: report分析スクリプトの水平展開（balance/profitability/components）(2026-02-22)

### 1) helper 水平展開（重複削減）

- `tools/analyze_balance_reports.py`
  - `glob + json.load` の直実装を `get_recent_training_reports()` / `load_training_report()` /
    `extract_action_distribution_from_payload()` へ統一。
- `tools/analyze_profitability_vs_balance.py`
  - 同様に report 列挙・読込・action distribution 抽出を catalog helper へ統一。
- `tools/analyze_ab_with_components.py`
  - report 収集/読込/reward_components 抽出/action 抽出を helper 経由へ全面移行。
  - default pattern (`training_report_*.json`) かつ `--filter-recent` 指定時は
    `get_recent_training_reports()` を優先利用し、不要な全件ロードを回避。
  - JSON 出力は `write_json` へ統一し、I/O 契約を共通化。

### 2) 不具合可能性の解消

- `analyze_balance_reports.py`:
  - `balance_shaping` が文字列等の場合に `sum()` で落ちる経路を `safe_to_float` で防御。
- `analyze_ab_with_components.py`:
  - component 統計集計時の `float(value)` 直変換を廃止し、型揺れを `safe_to_float` で吸収。
  - 壊れた JSON report は warning のみでスキップし、全体処理を継続。

### 3) 型安全・保守性

- 3スクリプトで `TypedDict` を導入し、解析 payload 契約を明示。
- report payload 取り扱いを `ensure_dict` / `safe_to_float` 経由へ統一し、
  script 間での型処理分岐重複を削減。

### 4) 検証

- `py_compile`:
  - `tools/analyze_balance_reports.py`
  - `tools/analyze_profitability_vs_balance.py`
  - `tools/analyze_ab_with_components.py`
- 回帰確認:
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reporting/services/test_catalog.py`
    - 結果: `9 passed`（warning 1）

## Phase 28 追補: report補助スクリプト群の型安全化 + helper横展開 (2026-02-22)

### 1) helper 横展開（重複排除）

- `tools/ab_search_result_summary.py`
  - `training_report` 走査・JSON読込・action distribution 抽出を
    `list_training_reports()` / `load_training_report()` /
    `extract_action_distribution_from_payload()` に統一。
- `tools/fix_broken_json_reports.py`
  - report 列挙を `list_training_reports()` へ統合。
- `tools/test_reward_components_fix.py`
  - 最新 report 取得を `get_recent_training_reports()` に統合し、
    components 抽出を `extract_reward_components_from_payload()` 経由へ統一。
- `tools/analysis/action_distribution_window.py`
  - report 読込を `load_training_report()`、列挙を `list_training_reports()` へ統合。

### 2) 不具合可能性の解消

- `test_reward_components_fix.py`
  - `ab_test_runner.py` 呼び出し引数を `--config` から
    現行契約の `--configs` へ修正。
  - 実行 python を固定文字列 `python` から `sys.executable` に変更し、
    venv 環境差異による実行失敗を低減。
- `fix_broken_json_reports.py`
  - backup 拡張子衝突時に `.json.bak1`, `.json.bak2` ... を採番するよう改善し、
    再実行時の rename 失敗を回避。

### 3) 型安全・保守性

- `tools/ab_search_result_summary.py`
  - `ActionAverage`, `BalanceSearchSummary` (`TypedDict`) を導入。
- `tools/duplicate_report_summary.py`
  - `read_json_object` + `ensure_dict/safe_to_float` で
    入力揺れに耐える要約処理へ変更。
- `tools/analysis/action_distribution_window.py`
  - `ActionDistribution` (`TypedDict`) と
    `_normalize_action_distribution()` を導入し、集計前の数値正規化を統一。

### 4) 検証

- `py_compile`:
  - `tools/ab_search_result_summary.py`
  - `tools/duplicate_report_summary.py`
  - `tools/fix_broken_json_reports.py`
  - `tools/test_reward_components_fix.py`
  - `tools/analysis/action_distribution_window.py`
- 回帰確認:
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reporting/services/test_catalog.py`
    - 結果: `9 passed`（warning 1）

## Phase 29 追補: JSONユーティリティ横展開 + duplicate削除処理の安全化 (2026-02-22)

### 1) JSON I/O 共通化の追加適用

- `tools/analyze_v447_configs.py`
  - `json.load` 直呼びを `read_json_object()` + `ensure_dict()` へ置換。
  - `ConfigSummary` (`TypedDict`) を導入し、設定要約の契約を明示。
- `tools/inspect_env.py`
  - config 読込を `read_json_object()` へ統一し、
    `environment.config` 取得を `ensure_dict()` で正規化。
- `tools/check_signals.py`
  - `read_json_object` + `ensure_dict/safe_to_int/safe_to_float` へ統一し、
    欠損キー時の参照例外を回避。
- `tools/compare_results.py`
  - 比較対象 JSON を `read_json_object` で読み込み、
    数値項目を `safe_to_float` 経由で差分計算。

### 2) 不具合可能性の解消（重複除去ツール）

- `tools/remove_duplicates.py` を再構成:
  - report 読込を `read_json_object()` へ統一。
  - `DuplicateOccurrence` / `RemovalRange` (`TypedDict`) を導入。
  - 旧実装の「同一ファイルを逐次編集して行番号がずれる」問題に対応し、
    file 単位に removal を集約して **降順削除** する方式へ変更。
  - path 解決時に root 外パスを reject するガードを追加。
  - ファイルごとの read/write を1回化し、I/O 重複も削減。

### 3) 検証

- `py_compile`:
  - `tools/analyze_v447_configs.py`
  - `tools/inspect_env.py`
  - `tools/check_signals.py`
  - `tools/compare_results.py`
  - `tools/remove_duplicates.py`
- 回帰確認:
  - `.venv/Scripts/python.exe -m pytest -q tests/unit/reporting/services/test_catalog.py`
    - 結果: `9 passed`（warning 1）
