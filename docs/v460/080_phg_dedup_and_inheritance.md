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
