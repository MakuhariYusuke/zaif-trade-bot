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
- offset stage recording も helper 化済み

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
- timeout / stale-reprice policy
  までは先行抽出済み

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

進捗:

- fill_rate / avg_pnl30 / downside_p10 の純粋な判定規則は
  `ztb.adaptation.ab_test.judgment_rules` へ前進済み
- script 側は dataclass / statistical comparison / report ownership を維持

### 5. `UnifiedTrainer` / `RewardCalculator`

現状:

- `ztb/training/unified_trainer/trainer.py`: 2607 行
- `ztb/trading/environment/components/calculators/reward_calculator.py`: 2197 行
- ただし無期限 defer は危険

方針:

- 実装の大分割は将来
- ただし責務境界だけは先に設計書へ落とす
- `env / trainer / reporting / reward` の 4 軸で切る

具体的な split 軸:

1. `UnifiedTrainer`
   - algorithm lifecycle
   - adaptation/ensemble/distributed option wiring
   - memory/performance monitoring
   - reporting/UI/session persistence
   - first extraction priority:
     - runtime feature flags
     - advanced-feature enablement gating
2. `RewardCalculator`
   - component initialization / config cache
   - state bookkeeping
   - reward rule composition
   - diagnostics / structured logging
   - first extraction priority:
     - stage bookkeeping
     - diagnostic payload shaping

先にやるべきこと:

- 実装分割そのものではなく、import 境界と ownership を固定する
- public compatibility を崩さない façade の必要有無を先に判断する

2026-03-21 update:

- `UnifiedTrainer` では runtime feature flag 解決を
  `ztb.training.unified_trainer.runtime_flags` へ先行抽出した
- これにより `distributed/federated/ensemble/mixed_precision/continual`
  の enablement 判定は pure helper として追える状態になった
- `runtime_flags` は現時点では `UnifiedTrainer` 専用色が強く、
  `v460` SAC script 群へ無理に横展開するより、trainer 側の SSOT として維持するのが妥当
- `advanced_feature_setup` は `UnifiedTrainer` 内の
  - model availability 判定
  - continual config 構築
  のような repeated setup 前提に対して有効だった
- 一方で SAC 側には `ztb.training.sac.memory_monitor` のような shared helper を置き、
  post-cycle の RSS / cache entry 監視を script 側から再利用する方が自然だった
- `RewardCalculator` はまだ実分割には入っていないが、
  先に抜く対象を
  - stage bookkeeping
  - diagnostics payload shaping
  に固定した
- その first step として、reward component payload の stage bookkeeping を
  `reward_component_tracking` helper へ寄せる方針を採る
- diagnostics payload shaping も同 helper を extend する形が自然で、
  `default/simple/safety` の各経路で診断項目の追加方法を揃えやすい
- さらに `UnifiedTrainer` では
  - continual learning 実行時の model 解決
  - fallback task data 用の model attr 参照
  - input/output dim 解決
  も `extract_algorithm_model(...)` ベースへ寄せられることを確認した
- これは `runtime_flags` のような広域 helper 化ではなく、
  `advanced_feature_setup` の適用範囲を trainer 内で最後まで揃える方が安全だった
- `RewardCalculator` では
  - PnL diagnostics
  - action/balance diagnostics
  - forced_balance / action_discovery / balanced_transition
  の bookkeeping も `reward_component_tracking` へ寄せる余地があり、順次整理を進める
- `reward_component_tracking` は `RewardCalculator` の stage payload SSOT として扱い、
  他の reward 系ファイルへ無理に広げない方針が妥当
- `UnifiedTrainer` では model availability だけでなく
  - continual learning 実行前の model 解決
  - fallback task data 用の input/output dim 解決
  - input/output dim fallback
  まで `advanced_feature_setup` 側へ寄せる余地があり、実装上も有効だった
- 特に `model` 自体はあるが `input_dim/output_dim` 属性を持たないケースでは、
  parameter shape fallback を helper 側に持つことで subtle bug を減らせる
- ただしこれは `UnifiedTrainer` の repeated setup / fallback に閉じる helper であり、
  SAC scheduler など別系統へ同名 helper を広げるより用途別 helper を維持するほうが安全
- integration 後半では
  - meta learner の task-buffer 判定
  - federated stats の取得
  - training_stats への書き戻し
  も helper 化しやすい層で、ownership を trainer 本体から少しずつ切り離せる
- `RewardCalculator` では
  - `simple_reward`
  - `trading_focused`
  - `profit_optimized`
  も stage payload の shape を canonical helper に寄せられる
- bool flag を `0.0/1.0` に揃える現在の payload 仕様は、
  後続の比較/集計では扱いやすいが、将来 JSON contract を変える場合は影響確認が必要
- さらに non-scalar telemetry は scalar payload と分けて扱うのがよく、
  `mtf_weights` のような辞書 payload は telemetry helper で stage 契約だけ維持する形が自然

## テスト設計

### 基本方針

1. shim 契約テスト以外は canonical import に寄せる
2. `TemporaryDirectory()` は `tmp_path` へ寄せる
3. real-data test は guard を実測ベースで詰める
4. timeout test は短すぎて不安定にしない
5. training 系の `TemporaryDirectory()` は、簡単な persistence/setup から `tmp_path` へ寄せる

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

## 2026-03-21 時点の Wave 判定

### Wave 1: 設計収束

- かなり進んだ
- `UnifiedTrainer`
  - `runtime_flags`
  - `advanced_feature_setup`
  - integration helper
  まで ownership が見えた
- `RewardCalculator`
  - scalar payload
  - non-scalar telemetry
  - stage bookkeeping
  の境界がかなり揃った
- 残りは「大分割」ではなく、helper 境界の追加固定が中心

### Wave 2: v460 本体の stateful orchestrator 整理

- 主要な pure policy / pure math 抽出はかなり進んだ
- `maker_price`
  - pure math / finalization / ceiling / stage tracking はかなり外出し済み
  - 残りは stateful orchestration 本体
- `order_monitor`
  - timeout / stale-reprice policy は抽出済み
- `ab_judgment`
  - judgment rule は抽出済み
  - 残りは statistical comparison / reporting ownership

### Wave 3: 性能・リーク・観測

- 引き続き継続
- 特に training/SAC 周りでは
  - cycle 後メモリ診断
  - cache stats
  - transient memory 削減
  を優先する
- 直近では reward component 集計を running-sum 化し、一時 list 保持を避ける

### Wave 4: テスト最適化

- 継続中
- `tmp_path` 化
- timeout/sleep の安定化
- real-data guard の実測最小化
  は引き続き有効
- training 系は persistence/setup 固定費から削る

## 直近の追加前進

### training stats payload 共通化

- `ztb/training/training_stats_payloads.py` を追加
- `record_training_stat(...)` を `UnifiedTrainer` 専用 helper から training 共通 helper へ昇格
- `build_optimization_training_stats(...)` により `UnifiedTrainer` の optimization payload を shared 化
- `average_reward_component_history(...)` により `SACTrainer` の reward component 集計を
  list 蓄積ではなく running-sum で処理するようにした

### Wave 3 観点の効果

- reward component averaging 時の一時メモリ保持を減らした
- SAC 側でも training stats の shaping を共通化でき、後続の observability 追加で payload がぶれにくくなった

### Wave 4 観点の効果

- `tests/training/algorithms/sac/test_sac_compression.py`
  の `TemporaryDirectory()` を `tmp_path` に置き換え
- training 系テストの I/O 固定費と boilerplate を少し削減した
