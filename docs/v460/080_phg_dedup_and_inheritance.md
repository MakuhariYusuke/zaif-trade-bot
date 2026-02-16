# 080# PHG: 重複排除 & 継承ベース統合

**日時**: 2025-02-24  
**コミット**: `63c557e2b` (phase1), `22733338b` (phase2)  
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
