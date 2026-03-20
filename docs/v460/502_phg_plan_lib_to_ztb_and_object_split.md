# 502# PHG: `scripts/v460/lib` → `ztb` 移行 / オブジェクト分割 実行計画

## 背景

`106#` と `108#` で残課題として認識されていた以下は、現在も有効:

- `scripts/v460/lib` に置かれた domain logic の一部は、本来 `ztb` にあるべき
- `run_fill_test.py` の分割は進んだが、抽出先の一部が `v460 lib` に留まっている
- `v461` 以降で本格移行する前提だと、今のうちに責務境界だけでも固めておかないと再び God Object 化しやすい

504# レビューを踏まえ、この計画書は以下を反映して改訂する:

- `ztb -> scripts.v460.lib` の逆依存を最優先で解消する
- 被参照数が多いモジュールは「低リスク移行」ではなく façade 前提の中リスク移行として扱う
- 既に分割済みのものは過剰分割しない
- 500 行超の大ファイルは少なくとも分類を明示する

## 106# / 108# との接続

### 106# で残課題扱いだったもの

- `fast_fill_defense`
- `param_adapter`
- `lot_sizer`
- `regime_detector`

現在は shared helper の昇格が進み、単独モジュール移行の下地はある。
ただし 504# の指摘どおり、`fast_fill_defense` と `regime_detector` は依存が重く、
façade なしの即時移行は危険。

### 108# で残課題扱いだったもの

- `UnifiedTrainer` God Object
- `HeavyTradingEnv` / 訓練環境まわりの責務過多

こちらは `lib -> ztb` 移行と別軸だが、共通する論点は
「orchestration と domain logic を分ける」ことにある。

## 現在の整理方針
## 実装進捗 (2026-03-20 時点)

既に canonical 化済み:

- `cancel_reasons.py` → `ztb/trading/common/cancel_reasons.py`
- `param_adapter.py` → `ztb/trading/sizing/param_adapter.py`
- `lot_sizer.py` → `ztb/trading/sizing/lot_sizer.py`
- `fast_fill_defense.py` → `ztb/trading/risk/fast_fill_defense.py` (shim 維持)
- `sac_common.py` → `ztb/training/sac/runtime.py` (shim 維持)
- `regime_detector.py` → `ztb/trading/signal/regime/regime_detector.py` (shim 維持)
- `bayesian_regime_filter.py` → `ztb/trading/signal/regime/bayesian_regime_filter.py` (shim 維持)

未着手の本命:

- `maker_price.py`
- `skip_gate_evaluator.py`
- `order_monitor.py`
- `ab_judgment.py`


### `scripts/v460/lib` に残すべきもの

`v460` 固有の orchestrator / CLI / wiring。

対象:

- `fill_test_cli.py`
- `fill_loop_orchestrator.py`
- `fill_cycle_executor.py`
- `orchestrator_*`
- `event_logger.py`
- `manifest.py`
- `cycle_gate_aggregator.py`
- `stopgap_health.py`
- `offset_pipeline.py`

理由:

- `v460` 固有の運用フローに強く依存
- `results_dir` / `run_id` / event log / fill test stop behavior などの文脈を持つ
- `ztb` に上げると逆に version-specific policy が混ざる

### `ztb` に上げるべきもの

version 固有ではなく、domain logic / reusable helper / shared training support。

#### Phase 0 / Phase 1 の低リスク

- `cancel_reasons.py`
- `param_adapter.py`
- `lot_sizer.py`
- `sac_common.py`

#### Phase 1.5 の中リスク（façade 必須）

- `fast_fill_defense.py`
- `regime_detector.py`
- `bayesian_regime_filter.py`
- `daily_drawdown_guard.py`
- `phantom_position_guard.py`
- `cross_venue_lead_lag.py`
- `config_hot_reload.py`

### まず分割してから移すべきもの

- `maker_price.py`
- `skip_gate_evaluator.py`
- `order_monitor.py`
- `adaptation_engine.py`
- `ab_judgment.py`

### 追加判断: 既に分割済みのもの

- `fill_config.py`
- `fill_config_parser.py`
- `fill_config_validation.py`
- `fill_config_results.py`

`fill_config` 系は 329# で既に 4 分割済み。現状の `fill_config.py` は schema 定義中心なので、
Phase 3 の split-first 対象から外す。

## 実ファイル所見

### 低リスク移行候補

| ファイル | 実測行数 | 被参照数 | 所見 |
|---|---:|---:|---|
| `cancel_reasons.py` | 212 | 高 | 依存数は多いがロジックが薄く、canonical 化しやすい。`ztb -> scripts` 違反の解消が先 |
| `param_adapter.py` | 311 | 3 | 実質 pure domain helper。最初に移しやすい |
| `lot_sizer.py` | 445 | 4 | sizing policy として独立性が高い |
| `sac_common.py` | 494 | 8 | shared cleanup を既に `ztb` に一部昇格済み |

### 中リスク移行候補（façade 前提）

| ファイル | 実測行数 | 被参照数 | 所見 |
|---|---:|---:|---|
| `fast_fill_defense.py` | 314 | 23 | `tests/unit/v460/conftest.py` を含め依存が広い。即時パス変更は危険 |
| `regime_detector.py` | 683 | 30 | `maker_price` / `order_monitor` / `adaptation_engine` など広範囲が依存 |
| `bayesian_regime_filter.py` | 576 | 中 | `regime_detector` と同時に寄せるのが自然 |
| `daily_drawdown_guard.py` | 667 | 中 | reusable な risk policy に寄せやすい |
| `phantom_position_guard.py` | 483 | 中 | risk guard として `ztb` 側が自然 |

### 分割先行候補

| ファイル | 実測行数 | 所見 |
|---|---:|---|
| `maker_price.py` | 1091 | 既に `maker_risk_guards.py` / `maker_microstructure.py` へ部分分割済み。残る core と inventory skew を再分離したい |
| `skip_gate_evaluator.py` | 866 | feature build / model load / decision policy が混在 |
| `order_monitor.py` | 645 | polling / stale judgement / retry / logging が混在 |
| `adaptation_engine.py` | 634 | adaptation policy と state mutation が密結合 |
| `ab_judgment.py` | 1178 | lib 内でも大型。AB 判定ロジックの責務が重い |

## 既知の是正事項（504# 反映）

### 1. `ztb -> scripts` 逆依存

確認済み:

- `ztb/metrics/fill_quality.py` が `scripts.v460.lib.cancel_reasons` を import

対応方針:

- `cancel_reasons.py` を Phase 0 で canonical 化
- `scripts/v460/lib/cancel_reasons.py` は compatibility shim とする

### 2. 配置先の意味衝突

504# 指摘どおり、`fast_fill_defense.py` を `ztb/trading/execution/` に入れるのは意味が悪い。
既存の `execution/` は simulation/backtest 文脈が強いため、以下へ修正する:

- `fast_fill_defense.py` → `ztb/trading/risk/fast_fill_defense.py`
- `regime_detector.py` / `bayesian_regime_filter.py` → 新規 top-level `ztb/trading/regime/` ではなく、既存の `ztb/trading/signal/regime/` namespace を優先する

### 3. façade 戦略を必須化

`fast_fill_defense` と `regime_detector` は import 影響が大きいため、移行時は必ず façade を挟む。

例:

```python
# scripts/v460/lib/regime_detector.py
"""Compatibility shim — canonical は ztb.trading.signal.regime.regime_detector."""
from ztb.trading.signal.regime.regime_detector import (  # noqa: F401
    FillTestRegime,
    RegimeDetector,
    RegimeResult,
)
```

### 4. `tests/unit/v460/conftest.py` の扱い

`fast_fill_defense` は `tests/unit/v460/conftest.py` に影響するため、ここを壊すと v460 全体が落ちる。
移行時は focused test だけでなく `tests/unit/v460/` broad を必須とする。

## 目標アーキテクチャ

### `ztb.trading.fill_test` は作らない

`fill_test` 自体は `v460` 運用体であり、canonical domain としては広すぎる。

代わりに reusable な単位へ寄せる。

### 推奨配置

| 現在 | 移行先候補 |
|---|---|
| `lib/cancel_reasons.py` | `ztb/trading/common/cancel_reasons.py` |
| `lib/param_adapter.py` | `ztb/trading/sizing/param_adapter.py` |
| `lib/lot_sizer.py` | `ztb/trading/sizing/lot_sizer.py` |
| `lib/sac_common.py` | `ztb/training/sac/runtime.py` |
| `lib/fast_fill_defense.py` | `ztb/trading/risk/fast_fill_defense.py` |
| `lib/regime_detector.py` | `ztb/trading/signal/regime/regime_detector.py` |
| `lib/bayesian_regime_filter.py` | `ztb/trading/signal/regime/bayesian_regime_filter.py` |
| `lib/daily_drawdown_guard.py` | `ztb/trading/risk/daily_drawdown_guard.py` |
| `lib/phantom_position_guard.py` | `ztb/trading/risk/phantom_position_guard.py` |
| `lib/cross_venue_lead_lag.py` | `ztb/trading/risk/cross_venue_lead_lag.py` or `ztb/trading/market_microstructure/` |
| generic parts of `lib/sidecar_signal_io.py` | `ztb/ml/sidecar_signal_io.py` |

### 分割後の推奨配置

| 現在 | 分割案 |
|---|---|
| `maker_price.py` | `ztb/trading/pricing/` 配下へ `pricing_core`, `inventory_skew`, `microstructure_hooks` |
| `skip_gate_evaluator.py` | `ztb/ml/skip_gate_features.py`, `ztb/ml/skip_gate_runtime.py`, `scripts/v460/lib/skip_gate_evaluator.py` は façade |
| `order_monitor.py` | `ztb/trading/execution/order_polling.py`, `ztb/trading/execution/stale_order_policy.py` |
| `adaptation_engine.py` | `ztb/trading/adaptation/state.py`, `ztb/trading/adaptation/policy.py` |
| `ab_judgment.py` | `ztb/trading/decision/ab_policy.py` + `scripts/v460/analysis/` 側 facade |

## 実行計画

### Phase 0: 事前固定 + 逆依存解消

対象:

- `cancel_reasons.py`
- 依存方向ルールの固定

やること:

1. canonical namespace を先に決める
2. `ztb` が `scripts.v460.lib` を import しないルールを固定する
3. `cancel_reasons.py` を `ztb/trading/common/` へ移し、旧 path は shim にする

完了条件:

- `ztb -> scripts.v460.lib` 逆依存が 0
- `scripts/v460/lib/cancel_reasons.py` は compatibility shim のみ

### Phase 1: 低リスク移行

対象:

- `param_adapter.py`
- `lot_sizer.py`
- `sac_common.py`

やること:

1. `ztb` 側に canonical module を作る
2. `scripts/v460/lib/*.py` は re-export / thin wrapper に縮める
3. focused + broad で import 互換を守る

### Phase 1.5: façade 必須の中リスク移行

対象:

- `fast_fill_defense.py`
- `regime_detector.py`
- `bayesian_regime_filter.py`

やること:

1. canonical module を `ztb` 側へ配置
2. 旧 module は façade のみ残す
3. `tests/unit/v460/conftest.py` を起点に broad regression を必須化

### Phase 2: SAC shared runtime 整理

対象:

- `sac_common.py`
- `retrain_scheduler.py` と `ztb.training` の cleanup/debug helper 重複

やること:

1. env cleanup / cuda cleanup / resource teardown を `ztb.training.sac` 系へ集約
2. script 側は orchestration だけ残す
3. debug summary / event detail 生成の共通部を `ztb` helper 化

進捗:

- `cleanup_training_resources(...)` は `ztb` 側へ昇格済み
- `build_training_debug_details(...)` は `ztb.training.sac.debug` へ昇格済み
- `retrain_scheduler` / `sac_retrain_scheduler` の timestamp は UTC helper へ寄せ始めている
- `retrain_once()` 結果は debug summary を保持できるようになり、history/debug 比較の足場ができた

### Phase 3: God Object 分割

対象:

- `maker_price.py`
- `skip_gate_evaluator.py`
- `order_monitor.py`
- `adaptation_engine.py`
- `ab_judgment.py`

順番:

1. `maker_price.py`
2. `skip_gate_evaluator.py`
3. `order_monitor.py`
4. `ab_judgment.py`
5. `adaptation_engine.py`

進捗:

- `maker_price` / `order_monitor` / `skip_gate_evaluator` の shared contract を先行抽出
- pricing / execution / skip-gate の import 面を `ztb` 側 contract へ寄せ始めた
- class 本体分割前に、protocol / result type の置き場を固定できた

### Phase 4: import 収束

1. `scripts.v460.lib` からの direct import を grep で棚卸し
2. `ztb` の canonical import へ順次置換
3. 最後に unused façade を archive 候補へ送る

足場:

- `ztb/trading/pricing/contracts.py`
- `ztb/trading/execution/contracts.py`
- `ztb/ml/skip_gate_contracts.py`

## テスト方針

### 移行時に必須のテスト

1. canonical module の pure unit test
2. façade が旧 import 契約を壊していないことの smoke test
3. `tests/unit/v460/` broad
4. fill test 影響点の focused regression

### 特に守るべきもの

- `tests/unit/v460/conftest.py` に依存するテスト群
- `test_enricher_skip_gate.py`
- `test_sac_retrain_scheduler.py`
- `test_retrain_hot_reload.py`
- `test_gate_check.py`
- `test_fill_quality.py`
- `test_145_structural_fixes.py`

## 当面の着手順

1. `cancel_reasons.py` を canonical 化して `ztb -> scripts` 違反を解消する
2. `param_adapter.py` を `ztb/trading/sizing/` へ上げる
3. `lot_sizer.py` を `ztb/trading/sizing/` へ上げる
4. `sac_common.py` を `ztb/training/sac/runtime.py` へ寄せる
5. `fast_fill_defense.py` を façade 前提で `ztb/trading/risk/` へ移す
6. `regime_detector.py` / `bayesian_regime_filter.py` を `ztb/trading/signal/regime/` へ寄せる

## リスク

### 高リスク

- `maker_price.py`
- `skip_gate_evaluator.py`
- `order_monitor.py`
- `ab_judgment.py`

### 中リスク

- `fast_fill_defense.py`
- `regime_detector.py`
- `bayesian_regime_filter.py`
- `daily_drawdown_guard.py`
- `phantom_position_guard.py`

### 低リスク

- `cancel_reasons.py`
- `param_adapter.py`
- `lot_sizer.py`
- `sac_common.py`
