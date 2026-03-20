# 505# PHG: 504# レビュー反映と Phase 0 着手

## 目的

504# のレビューを受けて、502# の `scripts/v460/lib` → `ztb` 移行計画を実コードベースに合わせて補正し、
そのまま Phase 0 の最初の実装に着手する。

## 504# の指摘で妥当だったもの

### 1. `cancel_reasons.py` の優先度が低すぎた

妥当。実際に `ztb/metrics/fill_quality.py` が `scripts.v460.lib.cancel_reasons` を import しており、
`ztb -> scripts` の逆依存が発生していた。

### 2. `fast_fill_defense.py` / `regime_detector.py` を「低リスク移行」と見ていた

妥当。実測の被参照数を踏まえると、これらは façade なしの直接移行には向かない。
特に `fast_fill_defense.py` は `tests/unit/v460/conftest.py` にも効く。

### 3. `fill_config.py` を split-first に置いていた

妥当。329# 時点で

- `fill_config.py`
- `fill_config_parser.py`
- `fill_config_validation.py`
- `fill_config_results.py`

へ分割済みで、追加分割の優先度は低い。

### 4. `ab_judgment.py` など大型未分類ファイルの明示分類不足

妥当。今の時点で少なくとも分類は固定しておく必要がある。

## 今回の軌道修正

### 1. 502# を改訂

反映内容:

- Phase 0 に `cancel_reasons.py` canonical 化を追加
- `fast_fill_defense.py` / `regime_detector.py` を Phase 1.5 の façade 必須へ格上げ
- `fill_config.py` を split-first から除外
- `ab_judgment.py` / `cycle_gate_aggregator.py` / `stopgap_health.py` / `daily_drawdown_guard.py` などの位置づけを明示
- `tests/unit/v460/conftest.py` 影響をテスト方針へ追記
- `fast_fill_defense` の移行先を `ztb/trading/risk/` に修正

### 2. Phase 0 を先に実装

最初の着手対象は `cancel_reasons.py` とした。

理由:

- ロジックが薄く canonical 化しやすい
- `ztb -> scripts` 違反を即解消できる
- 後続の `param_adapter` / `lot_sizer` / `sac_common` より先に片付ける価値がある

## 実装した Phase 0

### canonical module

- [cancel_reasons.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/ztb/trading/common/cancel_reasons.py)

### compatibility shim

- [cancel_reasons.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/cancel_reasons.py)

### 逆依存解消

- [fill_quality.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/ztb/metrics/fill_quality.py)
  - `AUDIT_CANCEL_REASONS` の import を canonical path に変更

### 型参照の追随

- [fill_record_helpers.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_record_helpers.py)
  - `CancelReason` の TYPE_CHECKING import を canonical path に変更

## テスト方針

### focused

- canonical module と shim の整合
- `fill_quality` が canonical path を使っていること
- 既存 `cancel_reasons` 構造テストが落ちないこと

### broad

- `tests/unit/v460/` filtered broad

## 次の着手順

1. `param_adapter.py` → `ztb/trading/sizing/param_adapter.py`
2. `lot_sizer.py` → `ztb/trading/sizing/lot_sizer.py`
3. `sac_common.py` → `ztb/training/sac/runtime.py`
4. `fast_fill_defense.py` façade 移行
5. `regime_detector.py` / `bayesian_regime_filter.py` façade 移行

## 実装しながら見えた追加補正

### 1. `regime_detector` / `bayesian_regime_filter` の移行先

実装を進める中で、tree には既に `ztb/trading/signal/regime/` が存在していた。
そのため 502# の当初案だった `ztb/trading/regime/` 新設よりも、
既存 namespace に寄せるほうが責務の置き場として自然。

このため移行先は次へ補正する。

- `regime_detector.py` → `ztb/trading/signal/regime/regime_detector.py`
- `bayesian_regime_filter.py` → `ztb/trading/signal/regime/bayesian_regime_filter.py`

### 2. `sac_common.py` は Phase 2 着手済み

`ztb/training/sac/runtime.py` を canonical module とし、
`scripts/v460/lib/sac_common.py` は compatibility shim に整理した。

追加進捗:

- `ztb/training/sac/debug.py` を追加し、training debug summary を `ztb` 側へ寄せた
- `scripts/v460/ml/sac_retrain_scheduler.py` は thin wrapper を残して既存 private 契約を維持
- `retrain_scheduler.py` の UTC timestamp は shared helper に追随

### 3. `bayesian_regime_filter.py` は `regime_detector.py` より先に移行可能

被参照範囲が狭く、shim 互換で十分守れるため、`regime_detector.py` 本体より先に
canonical 化して足場を作る方針が妥当だった。

### 4. Phase 3 は「shared contract 抽出」から入るのが安全

504# の懸念どおり、`maker_price.py` / `skip_gate_evaluator.py` / `order_monitor.py` は
本体を直接割り始めると import 影響が広い。

そのため先に

- pricing contracts
- execution contracts
- skip-gate contracts

を `ztb` 側へ抜き、旧 module はその contract を再利用する形に寄せるのが安全だった。
これは Phase 4 の import 収束にも直結する。

### 5. `order_monitor` は status / result type から抜くのが安全

`order_monitor.py` は `monitor()` 本体に async orchestration が密集しているため、
本体分割を急ぐよりも先に

- order status 正規化
- cancel → fill recheck の結果型

を `ztb/trading/execution/stale_order_policy.py` へ抜く方が安全だった。

この順なら source 契約テストを大きく壊さず、Phase 3 の split-first を前へ進められる。

### 6. Phase 4 の逆依存は小さいうちに潰す

shared contract 抽出後に再点検すると、
`ztb/trading/pricing/contracts.py` と `ztb/ml/skip_gate_contracts.py` が
まだ `scripts.v460.lib.ob_utils.OrderBookSnapshot` を見ていた。

これは `ztb -> scripts` 逆依存なので、
`ztb.trading.live.exchanges.base.broker_interfaces.OrderBookSnapshot`
へ寄せて先に閉じるのが妥当だった。

### 7. `maker_price` は state object 化より pure math 抽出から入る

`maker_price.py` は `_inv_buy_count` / `_inv_net_imbalance` / `_inv_last_update_time`
の内部状態を直接見るテストが多い。

そのため Phase 3 の最初の一手は state object 化ではなく、

- inventory counter 更新
- inventory imbalance の exp-decay

の pure 計算だけを `ztb.trading.pricing.inventory_math` へ抜く方が安全だった。

これで canonical 化を進めつつ、公開 API と内部属性契約は維持できる。

### 8. `skip_gate_evaluator` は runtime helper 抽出から進める

`skip_gate_evaluator.py` は hot-reload / ev_weighted / FillRecord 早期返却まで抱えており、
`evaluate()` 本体を先に割ると戻りが大きい。

そのため次の順が安全:

- `build_features_from_market_state()` は canonical のまま維持
- recent trades 正規化を `ztb.ml.skip_gate_runtime` へ抽出
- `OrderBookSnapshot` など shared contract は `ztb` 側参照へ寄せる

この順なら、`skip_gate_evaluator` の public 契約と source-based test を守りつつ、
Phase 4 の import 収束を前に進められる。

### 9. Phase 4 は `skip_gate` 周辺から本格化した

今回までで、

- deploy scripts
- model loader
- ev_weighted decision builder

の `SkipGate` / `SkipDecision` 参照は canonical `ztb.ml.skip_gate` に寄った。

一方で、以下はまだ script 文脈依存が強い:

- `SkipGateEvaluator._make_skip_fill_record(...)`
- `_assign_result_fields(...)`
- `_apply_decision_to_result(...)`

ここは `FillRecord` / log / config offset の責務が混ざるため、
Phase 4 で無理に上げず、Phase 3 で先に result assembly の詳細設計を詰めるのが妥当。

### 10. result assembly は 2 段で分けるのが安全

実装を進めると、`skip_gate_evaluator` の result assembly は次の 2 層に割れることが確認できた。

1. `SkipDecision -> result metadata`
2. `result metadata -> FillRecord early return`

今回 1 は `ztb.ml.skip_gate_result_fields` に抽出済み。
2 は `build_skip_fill_record(...)` と v460 固有の event/log 文脈に結びつくため、
まだ script 側に残すのが安全だった。
