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
- 550# は `maker_price` の state/stage 境界の詳細設計
- 551# は `550#` 後の Wave 2-5 実行順の整理

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

### B-2. `metrics` との責務境界

`metrics` と `adaptation` / `scripts` が近接する領域は、次の線引きを維持する。

- `ztb.metrics.fill_quality`
  - FillRecord schema
  - fill-quality / gate judgment
  - fill record load/filter/date helpers
- `ztb.metrics.record_metrics`
  - fill record 群からの共通集計
  - `fill_rate / avg_pnl30 / downside / AS / reprice / VG` の shared aggregation
- `ztb.adaptation.ab_test.*`
  - A/B 比較ルール
  - assessment / statistical comparison / verdict combination
- `scripts/v460/*`
  - run context
  - report formatting
  - orchestration

この方針により、旧 `scripts/v460/lib/metrics_utils.py` は compatibility shim に留め、
canonical 実装は `ztb.metrics.record_metrics` に寄せる。

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
- stage apply + stage record の重複も local helper 化済み
- stage store seed / final serialize も local helper 化済み
- source-contract test も direct call ではなく stage 契約を見る方向へ寄せる
- `offset_stages` には schema version を持たせ、mixed-SHA の解析を安全化済み

残る責務:

- stateful orchestration
- stage の適用順
- side/state/logging の結線

設計方針:

- state object 化を急がず、stage orchestration を明示化する
- pure helper は引き続き `ztb.trading.pricing.*` へ寄せる
- `compute()` の public/inspection 契約は壊さない
- state 分類と `compute()` stage 境界の詳細は `550#` を正本とする

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
- source-contract test も inline 算術ではなく policy helper 契約を見る方向へ寄せる

### 4. `ab_judgment.py`

現状:

- 大型ファイルのまま残る本命
- ロジックの重要度が高く、雑な分割は危険

方針:

- 先に basic contract と phase split を設計書で固定
- 最初の切り出しは pure judgment helper から
- 一気に `ztb` へ送らず、split-first を徹底する
- insufficient early return は canonical assessment + local result helper で薄く保つ

進捗:

- fill_rate / avg_pnl30 / downside_p10 の純粋な判定規則は
  `ztb.adaptation.ab_test.judgment_rules` へ前進済み
- insufficient early-return も small helper 化できる形まで揃い、
  sample/calendar/PnL-data 不足の判定 payload は pure helper 側へ寄せやすくなった
- primary criteria result append も local helper 化済み
- script 側は dataclass / statistical comparison / report ownership を維持

### 4.5. `toxicity_types`

進捗:

- `ToxicityAssessment` / `ToxicityLevel` は shared type として独立可能

設計方針:

- type 定義は `ztb.risk.toxicity_types` に寄せる
- `sell_dynamic_kill` は kill ロジックの ownership を維持しつつ re-export 互換を残す
- `toxicity_budget` / `cycle_gate_aggregator` / orchestrator 系は shared type を直接参照する

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
- 一方で SAC 側の post-cycle RSS / cache entry 監視は、
  `ztb.utils.memory_monitor` に shared helper を一本化し、
  `ztb.training.sac` 側は re-export に留める方が自然だった
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
- `test_113_resilience.py` のような state persistence test も
  `tmp_path` へ寄せやすい構造に順次揃える

## 直近の追加前進

### training stats payload 共通化

- `ztb/training/utils/training_stats_payloads.py` を追加
- `record_training_stat(...)` を `UnifiedTrainer` 専用 helper から training 共通 helper へ昇格
- 配置は `ztb/training/` 直下ではなく、既存 `training/utils` 構造に寄せる
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

## 2026-03-21 追加整理

### RewardCalculator

- `merge_reward_components(...)` を導入し、
  stage payload への後段 detail merge でも `stage` 契約を保つようにした
- 特に `forced_balance` の detail merge は raw `dict.update(...)` から脱して、
  scalar/telemetry 境界を崩しにくい形になった

### UnifiedTrainer

- reporting/session persistence 側も ownership を一段整理した
- `persist_training_report(...)`
- `persist_ensemble_report(...)`
  を reporting 側へ置き、
  trainer では UI 表示と error boundary に集中する形へ寄せた

### Wave 3 への効き

- reward component history の平均化は running-sum になり、
  training 後半の transient memory を減らせる
- training report / ensemble report の生成保存経路が一箇所に寄り、
  observability 追加時の payload drift を起こしにくくなった

### Wave 4 への効き

- `tests/unit/training/test_reward_components_persistence.py`
  でも `tmp_path` 化と shared averaging helper への追随を入れた
- reward/reporting helper の focused 回帰も増やし、保守時の破壊半径を下げた

## 2026-03-21 training 配置整理メモ

### 置き場所を固定したもの

- `ztb/training/utils/training_stats_payloads.py`
  - `TrainingStats` class と同じ `training/utils` 配下に置く
  - 理由:
    - trainer 専用ではなく `UnifiedTrainer` / `SACTrainer` / training tests で共有する
    - payload shaping helper であり、`unified_trainer` 専用 package に閉じると再利用境界が狭くなる

### 動かさないほうが良いもの

- `ztb/training/unified_trainer/runtime_flags.py`
- `ztb/training/unified_trainer/advanced_feature_setup.py`
- `ztb/training/unified_trainer/reporting.py`

理由:

- いずれも `UnifiedTrainer` の orchestration ownership に強く結びついている
- `training/utils` に出すと generic helper に見えてしまい、SAC や legacy trainer へ無理に流用しやすくなる
- `components/` は manager / compatibility shim の置き場として既に使っており、
  pure helper 群を混在させると逆に見通しが悪くなる

### shim / canonical の判断

- `components/reporter.py`
  - canonical 実装ではなく compatibility shim として残す
  - `TrainingReporter` の legacy method signature を吸収する役割がある
- canonical 実装は `ztb/training/unified_trainer/reporting.py`
  - 新規 import はこちらを優先する

### 現時点の整理方針

1. `training/utils`
   - package 横断で使う pure helper
2. `training/unified_trainer/*`
   - `UnifiedTrainer` 専用の runtime/setup/reporting helper
3. `training/unified_trainer/components/*`
   - manager / UI / compatibility shim

この切り分けなら、今後 `UnifiedTrainer` をさらに分割するときも
「generic helper を外へ」「orchestration helper は trainer 内へ」
という基準を維持しやすい。

### helper の出どころを曖昧にしない

- `record_training_stat(...)` のように既に `training/utils` へ昇格した helper は、
  `advanced_feature_setup` など trainer 専用 module から再 export しない
- 目的は、
  - helper の canonical path を一つに保つ
  - `unified_trainer` 専用 helper と training 共通 helper を混同しない
  こと
- こうしておくと、今後 `UnifiedTrainer` を分割するときも import 探索が単純になる

### training 既存資産の再利用で今後も使うもの

- `components/reporter.py`
  - legacy signature を吸収する shim として再利用価値がある
  - ただし canonical import 先にはしない
- `reporting.py`
  - current reporting/persistence の canonical 実装として使い続ける
 - `components/config_manager.py`
  - `UnifiedTrainer` の runtime config 正規化として使い続ける
  - `core/config_manager.py` とは責務が違うため、現時点では統合しない

この 3 つは「似ているからまとめる」より、
「責務差を明文化して使い分ける」方が安全である。

## 2026-03-21 Wave2/Wave3 補記

- `maker_price`
  - stage orchestration 自体は script 側に残しつつ、stage apply + stage tracking の重複を local helper へ集約
  - これにより stateful ownership を崩さず compute 本体の見通しを上げる
- `ab_judgment`
  - insufficient 判定は canonical assessment helper と local result helper の二層に整理
  - pure rule は `ztb.adaptation.ab_test`、`ABJudgmentResult` ownership は script 側、の境界を維持
- `heavy_env`
  - terminal penalty は helper 化済み
  - `reward_components` は reward delta の符号、`info` は監視しやすい penalty 量、で責務を分ける
- `SAC`
  - post-cycle memory diagnostics は `ztb.utils.memory_monitor` に一本化する
  - `ztb.training.sac` 側は re-export に留め、scheduler 側は logging と lifecycle ownership に寄せる
  - background monitor の停止は `Event.wait()` ベースにして、停止待ちの固定 sleep を持たない
- `Wave4`
  - training 系 test の `sleep` は、契約確認に不要なものから CPU work ベースへ置換して固定費を削る
