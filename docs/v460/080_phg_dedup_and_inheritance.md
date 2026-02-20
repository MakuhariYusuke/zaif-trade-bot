# 080# PHG: 重複排除 & 継承ベース統合

**日時**: 2025-02-24 (+ 2026-02-16 追補)  
**コミット**: `63c557e2b` (phase1), `22733338b` (phase2), `(current)` (phase3追補)  
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
