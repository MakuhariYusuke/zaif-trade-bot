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

その後の整理として、2 の内部でも

- rule-based skip
- velocity hard skip
- final decision skip

に共通する `result + early_return_record` 組立は
`skip_gate_evaluator` の local helper へ集約した。

これにより、次の検討対象は

- `build_skip_fill_record(...)` を `ztb` に残したまま payload 境界だけ抜くか
- `FillRecord` 生成自体を別の canonical helper に分けるか

の 2 択まで狭まっている。

現時点の判断としては前者が安全寄りである。

- `ztb.metrics.fill_quality.build_skip_fill_record(...)` は canonical builder として残す
- `skip_gate_evaluator` 側は payload を作る local helper / small value object までに留める
- これなら `FillRecord` ownership を動かさずに Phase 3 を締められる
- 実装はさらに一歩進めて、`skip_gate` 由来の extra payload 自体は
  `ztb.ml.skip_gate_result_fields` で canonical 化し、script 側は v460 文脈の core fields だけを保持する

### 11. `maker_price` は inventory math の次に offset math を抜く

`maker_price.py` の次の安全な抽出対象は、

- `effective_max_ratio(...)`
- `scale_offset_ratio(...)`

のような純粋な ratio helper だった。

これらは `FillRecord` / detector / FFD state に依存せず、
`inventory_math` と同じく `ztb` 側へ昇格しても責務がぶれない。

一方で

- `effective_sell_offset_floor`
- spread adaptive の stage orchestration
- final result assembly

は config 文脈や stage 順序に依存するため、まだ `maker_price.py` 側に残すほうが安全。

ただし `effective_sell_offset_floor` の中でも

- base floor
- bypass threshold
- inventory imbalance
- discount ratio

だけで決まる割引計算そのものは純ロジックなので、
これは次の一手として `ztb.trading.pricing.offset_math` へ抜くのが妥当だった。

### 12. real-data integration test は境界を測ってから削る

`test_enricher_skip_gate.py` の real-data setup は broad の主因のひとつだが、
ここは単純に sample を削ると不安定になりやすい。

2026-03-20 時点の tail 実測では、

- `20 trainable samples` を満たす最小 tail は `50 rows`

だったため、現在の guard は

- initial `52`
- fallback `72`
- expanded `96`

としている。

この方針なら、速度を落とさずに「今の実データ」での成立境界を docs に残せる。

### 13. Phase 4 は production だけでなく test-side import 収束も有効

shim 互換は残しているが、移行末期では test 側も canonical import に寄せておくほうが
後続の修正面積を減らせる。

今回の判断:

- migration/shim 契約を検証する test は旧 path のまま維持
- 通常の unit/integration test は `ztb` canonical import を優先

これにより、Phase 4 の残りは「shim を残す必要がある production 文脈」へ集中できる。

### 14. `maker_price` は pure finalization まで canonical 化してよい

`maker_price.py` は stateful な stage orchestration がまだ重いが、

- `best_bid`
- `best_ask`
- `spread`
- `offset`
- `effective_offset_ratio`
- `side`

だけで決まる spread guard 付き finalization は pure helper として安全に抜ける。

このため、

- `ztb.trading.pricing.price_finalization.finalize_price_with_spread_guard(...)`

を canonical helper とし、
script 側は wrapper を残して source/契約テスト互換を保つ方針が妥当だった。

この切り方なら、Phase 3 を進めつつ

- logging を含む stage orchestration
- detector / FFD / config 文脈

は無理に `ztb` へ上げずに済む。

### 15. `skip_gate_evaluator` は local value object で締める

`skip_gate_evaluator` の終盤は shared helper を増やすより、

- v460 固有 core context
- canonical extra payload

を明示的に分けたほうが安全だった。

そのため、

- local `_SkipFillRecordContext`
- canonical `SkipFillRecordExtraFields`

の 2 層とし、
`FillRecord` 最終組立自体は script ownership に残す方針へ寄せている。

これで残る責務は

- final builder call
- event/logger/run_id/git_sha の実行文脈

にかなり限定された。

さらに local `_build_skip_fill_record_context(...)` を置くことで、

- unknown regime rule skip
- velocity rule skip
- final decision skip

の 3 経路で重複していた context 構築も解消した。

`maker_price` についても、pure helper 抽出だけでなく
invalid mid/spread guard を先に入れておくことで、
stage orchestration 分割前の安全性を高める方針を取っている。

同様に Phase 4 では、production だけでなく

- lot_sizer
- param_adapter
- bayesian_regime_filter

の functional test も canonical import へ寄せ、
shim 契約テストだけを legacy path 側へ残す方向を継続している。

また `maker_price` は、

- spread guard finalization
- loss boost decay multiplier
- spread adaptive pure math

まで pure helper を `ztb.trading.pricing` 側へ寄せられている。

加えて `skip_gate_evaluator` の velocity hard skip では、
cancel reason literal をやめて canonical
`CR.SKIP_GATE_RULE_VELOCITY_SELL/BUY` へ統一した。
これは小さい変更だが、cancel reason をキーにした集計や後続分析の
SSOT を守るうえで重要である。

このため、残る Phase 3 論点は

- spread adaptive / FFD / loss boost の stage orchestration
- `FillRecord` 最終 builder ownership

へさらに限定された。
