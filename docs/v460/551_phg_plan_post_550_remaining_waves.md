# 551# PHG: Post-550 remaining waves plan

## 目的

`550#` で `maker_price` の state/stage 境界を可視化した後、
現在の残課題を **Wave 2-5** の実行計画として整理する。

本書は「何が終わっていて、何がまだ重いか」を自分向けに素早く再確認するための運用計画であり、
詳細な設計判断は 521# と 550# を正本とする。

## 現在地

### すでにほぼ固まったもの

- `skip_gate` の contract / runtime / result metadata / fill-record context
- `toxicity_types` の shared 化
- `offset_stages` schema version 付与
- `maker_price` の pure helper 群
- `order_monitor` の pure policy 群
- `ab_judgment` の pure judgment rule 群
- `UnifiedTrainer` の runtime/setup/reporting helper 境界
- `RewardCalculator` の scalar payload / telemetry / bookkeeping 境界

### まだ重いもの

1. `scripts/v460/lib/maker_price.py`
2. `scripts/v460/lib/ab_judgment.py`
3. trainer / SAC / heavy_env の telemetry payload 整列
4. broad 前の real-data / wait / setup 固定費
5. broad 最終確認そのもの

### filtered broad の最新確認

- `tests/unit/training tests/unit/evaluation tests/training`
  - `677 passed, 17 skipped, 8 warnings in 28.41s`
- `Wave5` の入口確認は取れている
- 以後は broad failure 対応よりも、Wave3/4 の残差整理と current suite 固定費削減を優先してよい
- current suite (`tests/unit/training tests/unit/evaluation tests/training`) では
  - `TemporaryDirectory()`
  - `NamedTemporaryFile()`
  - `time.sleep()`
  の grep hit を解消済み
- `tests/unit/v460` broad では `LiteTradingEnv` の `RewardKernel` wiring 漏れを 1 件拾って修正済み
- 修正後の `tests/unit/v460` broad は `4762 passed, 2 skipped` まで進み、assertion failure の再発はなかったが、環境側 `KeyboardInterrupt` で完走確認までは至っていない

## Wave 別の残課題

### Wave 2: stateful ownership の最終整理

#### A. `maker_price.py`

残る本丸:
- stateful orchestration 自体
- veto / telemetry / cache の最終 ownership
- `compute()` 前半の preflight/cache resolve の最終整理

次の打ち手:
1. stage seed / final serialize / preflight-cache-resolve helper 化までは完了として扱う
2. veto telemetry と cache ownership の update 点を local helper にさらに寄せる
3. `compute()` source-contract test を stage/preflight 契約ベースへさらに寄せる

Done の基準:
- `compute()` が「preflight」「stage pipeline」「finalize」に 3 分割で読める
- source-inspection test が direct 実装断片より helper/stage 契約を見る
- optional stage (`kyle` / `amihud` / `imb_risk` / `buy_as_guard`) も helper 契約で読める

現在の見立て:
- `maker_price` / `ab_judgment` ともに終盤
- 以後は大分割ではなく、Wave3/4 を進めながら residual を拾う運用でよい
- 2026-03-25 時点で stale source-contract 1件
  - `run_single_cycle()` 直参照
  を `_submit_order_phase()` helper 契約へ更新済み

#### B. `ab_judgment.py`

残る本丸:
- `ABJudgmentResult` orchestration
- statistical comparison の残る ownership
- report summary / dashboard wording

次の打ち手:
1. result 初期化 + insufficient early return + primary criteria append の local helper 化までは完了として扱う
2. statistical comparison payload shaping も完了として、残る比較 ownership を見直す
3. reporting 文面は最後まで script ownership に残す

Done の基準:
- pure rule / local orchestration / reporting の 3 層が混ざらない
- `judgment_rules.py` は pure rule に留まり、script 側は result/report の責務だけ持つ

2026-03-25 追加前進:
- targeted mypy:
  - `scripts/v460/lib/ab_judgment.py`
  - `Success: no issues found in 1 source file`
- low-risk fix:
  - `FillRecord` を `TypeAlias` 化
  - ndarray payload / analyzer protocol を明示
  - `ABTestAnalyzer` fallback 経路の callable/type 契約を固定

### Wave 3: telemetry / diagnostics / leak prevention

対象:
- `UnifiedTrainer`
- `SACTrainer`
- `sac_retrain_scheduler`
- `heavy_env`

次の打ち手:
1. training stats / reward telemetry / memory diagnostics の payload shape を揃える
2. `record_average_reward_components(...)` による training stats 収束と、heavy_env の `_sync_terminal_reward_outputs(...)` / `_append_reward_diagnostics_to_info(...)` を基準に `info` と `reward_components` の責務分離を横展開する
3. leak warning / rss warning / cache entry count の観測を一貫化
4. callback / reporting で扱う `reward_components` 取得経路も shared helper に寄せる
5. `RewardCalculator` の `get_last_reward_components()` は snapshot 契約へ寄せ、mutable payload alias を避ける
6. `RewardCalculator` の stage payload 更新点は local helper へ寄せ、`_last_reward_components` の ownership を一本化する
7. `UnifiedTrainer` の advanced feature stats は helper 経由で記録し、feature ごとの payload drift を減らす

Done の基準:
- telemetry field の符号/意味が module ごとにぶれない
- memory diagnostics が utils/training helper から辿れる

### Wave 4: broad 前の固定費削減

対象:
- `tests/unit/v460`
- `tests/unit/training`
- `tests/training`

次の打ち手:
1. `TemporaryDirectory()` 残件の `tmp_path` 化
2. `time.sleep()` ベース wait の `Event.wait()` / predicate wait 化を継続
3. real-data setup の fixture 再利用 / sample cap 見直し
4. stale source-contract test の stage/helper 契約化
5. current suite に残る `NamedTemporaryFile()` の `tmp_path` 化を進める

直近で前進したもの:
- `tests/training/callbacks/performance/test_performance.py` の skipped benchmark fixed wait を `Event.wait()` 化
- `tests/test_analyze_fill_logs.py` の tempdir fixture を `tmp_path` 化
- `tests/unit/utils/test_path_utils.py` の tempdir 使用を `tmp_path` 化
- `tests/unit/trading/components/test_performance_optimizer.py` の fixed `sleep` を小さい CPU work に置換
- `cache/*.db-shm` / `cache/*.db-wal` / `cache/sidecar_signal.json` を ignore し、`sidecar_signal.json` も追跡から外して broad 前の worktree ノイズを削減
- `tests/unit/training/test_unified_data_loading.py` の CSV/Parquet fixture を `tmp_path` ベースへ整理
- `tests/training/distributed/test_distributed_training.py` の checkpoint fixture を `tmp_path` ベースへ整理
- `tests/unit/evaluation/test_unified_evaluation.py` の temp file fixture を cleanup-aware path helper に整理
- `tests/unit/v460/test_v460_core.py` の gate-check JSON fixture を `tmp_path` ベースへ整理
- `tests/unit/v460/test_189_alt_horizon_macro_integration.py` の YAML fixture を `tmp_path` ベースへ整理

Done の基準:
- broad 上位が「本物の計算 / 実データ / I/O」に再集中する
- artificial wait / tempdir boilerplate が top offenders から外れる

### Wave 5: broad 最終確認

次の打ち手:
1. filtered broad を実行
2. top durations を再抽出
3. truly remaining bottleneck だけ追加で触る
4. 521# / 037# に区切りの要約を残す

Done の基準:
- broad が安定して回る
- 残課題が「future に送るもの」と「今やるべきもの」に明確に分かれる

## 追加方針: targeted mypy をどう Wave に織り込むか

Wave 3-5 では、型改善を独立タスクとして扱わず、各 Wave の入口確認として使う。

### Wave 3

- telemetry / payload の outward contract を変える前に targeted mypy を回す
- `dict[str, object]` / `Protocol` / `TypeAlias` へ寄せられるものだけ直す
- runtime の意味変更はしない

### Wave 4

- cleanup 対象 test の helper / fixture 変更前後で targeted mypy を回す
- 「cleanup で型が崩れた」事故を早めに止める

### Wave 5

- broad 前に、直近で触った module 群だけ targeted mypy を回す
- repo-wide mypy は broad の gate にしない
- 差分確認の入口は 589# の targeted runner を正本にする

## 実装判断を減らすための優先規則

次に着手するときは、この順で判断する。

1. 既存の shared type があるか
2. `Any` を増やさず `Protocol` / `TypeAlias` / `cast` で止められるか
3. focused pytest で守れるか
4. 並行差分に触れずに切り出せるか

4 を満たさない場合は、その module は後ろに回す。

## 優先順

1. `maker_price` veto/telemetry/cache ownership の最終整理
2. `ab_judgment` result/report ownership の最終整理
3. trainer/SAC/heavy_env telemetry payload 整列
4. broad 前の wait/setup 固定費削減
5. broad 最終確認

## 実行順の深掘り

### Step 1: Wave 2 を「終盤」から「完了判定可能」へ進める

先にやる理由:
- `maker_price` と `ab_judgment` は downstream test / telemetry の基準点になっている
- ここが揺れたままだと Wave 3/4 の観測や broad の結果が読みづらい

具体手順:
1. `maker_price`
   - veto / cache / telemetry の update 点を local helper に寄せる
   - source-contract test を helper/stage 契約へ寄せる
   - `compute()` の残る inline ownership を減らす
2. `ab_judgment`
   - result container
   - statistical payload
   - summary/report 文面
   の 3 層をさらに固定する
3. 「これ以上は state object 化になる」地点で止める

止めどころ:
- public shape や source-inspection contract を崩し始めるなら、そこで止める
- state object 化や大分割は Wave 2 の守備範囲から外す

### Step 2: Wave 3 を payload 契約の収束に限定して進める

先にやる理由:
- broad 前に最も壊れやすいのは、計算そのものより payload の意味ズレ
- `RewardCalculator` / `heavy_env` / trainer / SAC の outward shape が揃うと調査負荷が大きく下がる

具体手順:
1. `RewardCalculator`
   - `557#` を正本として payload/telemetry 契約を詰める
   - `_last_reward_components` の更新点を helper に寄せ続ける
2. `heavy_env`
   - `reward_components` と `info` の責務を崩さない
   - outward payload は snapshot を原則にする
3. trainer / SAC
   - `training_stats` / `reward_components` / `memory diagnostics` を canonical helper 経由へ寄せる

止めどころ:
- `RewardKernel` へ stateful logic を押し込み始めたら止める
- telemetry の意味統一が終わる前に大分割へ進まない

### Step 3: Wave 4 は broad 前の「ノイズ取り」に徹する

先にやる理由:
- broad の遅さのうち、価値が低いものを先に削ると最後の判断がしやすい

具体手順:
1. `TemporaryDirectory()` を `tmp_path` へ
2. `time.sleep()` を `Event.wait()` / CPU work / predicate wait へ
3. real-data fixture の再利用
4. worktree ノイズになる cache 生成物の ignore/untrack

止めどころ:
- テストの意味を変えるほど synthetic 化しない
- 実運用に近い real-data 契約まで削らない

### Step 4: Wave 5 は「本物の残課題だけ」を拾う

具体手順:
1. filtered broad 実行
2. top durations 抽出
3. failure / duration の上位だけ個別に読む
4. `521#` / `037#` / `551#` に区切りを書く

止めどころ:
- broad の数字だけを追って、設計を崩さない
- その場しのぎの flaky 対策を増やさない

## 着手判断ルール

- `557#` に関わる報酬系は、原則として Wave 3 の一部として扱う
- ただし `RewardCalculator` の local ownership 圧縮のように、payload 契約を壊さずに進められるものは先行してよい
- 逆に `RewardKernel` への本格寄せは、`557#` の境界固定が済むまで急がない

## 今の判断

- いま無理に state object 化へ進むのは早い
- Wave 2 はかなり終盤で、次は ownership の最終整理と Wave 3/4 の仕上げを並行で進めるのが安全
- `550#` は設計の基準、`551#` は実行順の基準、`521#` は全体の母艦、という役割分担で運用する
- 報酬系の詳細設計と実行順は `557#` を正本とし、`551#` では Wave 全体との接続だけを持つ
- `scripts/v460` 側の helper 再利用は
  - `metrics`
  - `memory_monitor`
  - `offset_stages`
  周辺の主要ポイントはかなり回収済みで、残るものは無理に shared 化せず local ownership を保つ方が安全

## 直近の前進メモ

- `maker_price`
  - cross-venue stage + veto raise を local helper に寄せ、`compute()` の責務をさらに薄くした
- `ab_judgment`
  - per-regime criteria 構築と single-regime evaluation を local helper に寄せた
- Wave 3
  - `training_stats_payloads.py` に reward payload 抽出 helper を追加し、
    callback 側の `reward_components` 取得経路を canonical 化した
  - reporting 側も同じ helper を使う形に寄せ、flat stats からの reward metrics も同一経路で扱えるようにした
- Wave 4
  - `tests/unit/utils/test_utils.py`
  - `tests/unit/utils/test_file_utils.py`
  - `tests/unit/evaluation/test_evaluate.py`
  を `tmp_path` ベースへ整理した
  - `tests/unit/evaluation/test_walk_forward_checkpoint.py`
  - `tests/unit/evaluation/test_walk_forward_integration_e2e.py`
  も `tmp_path` ベースへ整理した
 - prompt `587#`
   - additive / eDRC / entry gate の設定線と hot-reload 範囲を整理
   - final clamp は robust inputs を通す形へ前進
