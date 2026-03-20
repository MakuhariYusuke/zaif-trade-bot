# 521# PHG: master deferred docs / architecture carry-forward

## 目的

本書は、`docs/v460` に散在していた

- deferred docs の棚卸し
- `lib -> ztb` carry-forward
- Phase 3/4 の残課題
- future 維持が妥当な設計課題

を、以後は **1 本の living document** として更新し続けるための基準文書である。

以後の運用では、

- `514#` は deferred docs 監査の履歴
- `520#` は deferred 項目スクリーニングの履歴
- `521#` は現在の一本化された更新先

と扱う。

## 運用原則

### 1. ドキュメント番号を主にする

複数エージェントが同時に関わるため、内部の作業単位やセッション番号ではなく、
`docs/v460` の document number を主軸に置く。

### 2. 計画と履歴を分ける

- 502# / 505# は `lib -> ztb` 移行計画とレビュー反映の履歴
- 514# / 520# は deferred docs 監査の履歴
- 521# は現時点の carry-forward と設計判断の集約先

### 3. 「前進」と「完了」を混同しない

shim / façade が残るもの、あるいは `scripts/v460` 側に orchestrator ownership が残るものは、
`done` ではなく `収束段階` として記録する。

### 4. 実コード設計と docs 保守を分離しない

設計上の carry-forward は docs だけに閉じず、

- canonical module の責務
- shim の残存理由
- stateful orchestration の ownership
- テストと観測の維持方針

まで併記する。

## 現在の全体像

### 既に大きく前進したもの

- `cancel_reasons`
- `param_adapter`
- `lot_sizer`
- `fast_fill_defense`
- `sac_common`
- `regime_detector`
- `bayesian_regime_filter`
- `skip_gate` の contract / runtime / result metadata / fill-record context
- `maker_price` の pure math 群
- observability / memory diagnostics / cycle revenue logging

### まだ収束途中の本命

1. `maker_price.py`
2. `order_monitor.py`
3. `ab_judgment.py`
4. `UnifiedTrainer`
5. `RewardCalculator`

### docs 上は stale だったが、今は更新済みのもの

- `106#`
- `108#`
- `113#`
- `118#`
- `121#`
- `158#`
- `168#`
- `420#`
- `502#`
- `505#`

## 基本設計

### A. 層の切り分け

#### `scripts/v460`

責務:

- CLI / orchestrator
- `run_id` / `results_dir` / event log
- fill test 固有の運用制御
- compatibility shim

残してよいもの:

- `run_fill_test.py`
- `fill_test_cli.py`
- `fill_loop_orchestrator.py`
- `fill_cycle_executor.py`
- `event_logger.py`
- `manifest.py`

#### `ztb`

責務:

- reusable domain logic
- shared contracts
- type-safe pure helper
- training/runtime support
- pricing/risk/signal/execution policy

### B. canonical module 化の基準

`ztb` に上げる条件:

1. version-specific path に依存しない
2. `run_id` / event log / `results_dir` を直接触らない
3. pure helper か shared policy として再利用可能
4. type contract を stable に置ける

### C. shim を残す条件

shim を残す条件:

1. import 影響が広い
2. source-inspection test が存在する
3. `scripts/v460` path が documentation 上の参照点になっている

shim を外しにいく条件:

1. production import が canonical へ収束済み
2. test import も canonical へ追随済み
3. 残るのが compatibility のみ

## モジュール別設計メモ

### 1. `maker_price.py`

進捗:

- inventory math 抽出済み
- offset math 抽出済み
- sell floor discount 抽出済み
- loss boost decay 抽出済み
- spread adaptive 抽出済み
- spread guard finalization 抽出済み
- final ceiling clamp は stage 化済み

残る責務:

- stateful orchestration
- stage の適用順
- side/state/logging の結線

設計方針:

- state object 化を急がず、stage orchestration を明示化する
- pure helper は引き続き `ztb.trading.pricing.*` へ寄せる
- `compute()` の public/inspection 契約は壊さない

### 2. `skip_gate_evaluator.py`

進捗:

- runtime helper 抽出済み
- result metadata 抽出済み
- extra payload 抽出済み
- fill-record context / builder は canonical 側へ前進済み

残る責務:

- v460 固有 run context
- logger / event 文脈
- early-return の orchestration ownership

設計方針:

- `FillRecord` そのものを `ztb` へ全面移管しない
- `context -> payload -> record builder` の境界を維持する
- `cancel_reason` と result field の SSOT を崩さない

### 3. `order_monitor.py`

進捗:

- stale order policy
- cancel/fill recheck result

は先行抽出済み。

残る責務:

- async polling orchestration
- retry / heartbeat / event/log 連携

設計方針:

- pure policy は `ztb.trading.execution` 側へ寄せる
- async orchestration は `scripts/v460` に残す

### 4. `ab_judgment.py`

現状:

- 大型ファイルのまま残る本命
- ロジックの重要度が高く、雑な分割は危険

方針:

- 先に basic contract と phase split を設計書で固定
- 最初の切り出しは pure judgment helper から
- 一気に `ztb` へ送らず、split-first を徹底する

### 5. `UnifiedTrainer` / `RewardCalculator`

現状:

- docs 上でも future 寄り
- ただし無期限 defer は危険

方針:

- 実装の大分割は将来
- ただし責務境界だけは先に設計書へ落とす
- `env / trainer / reporting / reward` の 4 軸で切る

## テスト設計

### 基本方針

1. shim 契約テスト以外は canonical import に寄せる
2. `TemporaryDirectory()` は `tmp_path` へ寄せる
3. real-data test は guard を実測ベースで詰める
4. timeout test は短すぎて不安定にしない

### 重いテストの扱い

#### `test_enricher_skip_gate.py`

- real-data sample guard は実測に基づいて維持
- 本体コスト削減で改善する領域

#### `test_sac_retrain_scheduler.py`

- timeout/error 系は保守性重視
- 短縮しすぎて flaky にしない

## 性能・メモリ・不具合観点

### 計算量削減

- pure math の重複は helper に寄せる
- DataFrame 生成前 filter を優先
- real-data tests は候補 tail を実測で最小化

### メモリリーク防止

- long-lived cache は bounded + clearable を前提にする
- cycle/scheduler 終了時 cleanup を明示する
- diagnostics は event log と history の双方へ残す

### subtle bug の監視対象

- timestamp / UTC day の手書き
- cancel_reason literal の逸脱
- fallback path の固定値化
- `ztb -> scripts` 逆依存の再発

## 今やるもの / future 維持のもの

### 今やるもの

1. `maker_price` orchestration の最終整理
2. `order_monitor` / `ab_judgment` の split-first detailed design
3. canonical import sweep の最終残件確認
4. docs の stale deferred 表現の補正

### future 維持でよいもの

1. `utils/` 70+ 分割
2. `UnifiedTrainer` 大分割本体
3. event-driven cycle
4. WebSocket API 活用
5. online learning
6. `asyncio.to_thread` 残件の全面整理

## 更新ルール

1. 今後の carry-forward はまず本書を更新する
2. `514#` / `520#` には新しい判断を足し込まず、必要なら本書への参照だけ追加する
3. 個別 docs は historical record を保ちつつ、必要なら補遺を追記する
4. docs 番号を主に扱い、内部作業番号を主語にしない

## 2026-03-21 時点の判定

docs の先送り管理は、本書へかなり一元化できる状態に入った。

コード側も、

- Phase 3 は終盤
- Phase 4 はかなり進行
- 残る大物は限定的

という水準まで来ている。

今後は、新しい monitoring note を増やすより、
本書を current carry-forward の母表として更新し続けるのが最も安全である。
