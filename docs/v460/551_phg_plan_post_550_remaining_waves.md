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

直近で前進したもの:
- `tests/training/callbacks/performance/test_performance.py` の skipped benchmark fixed wait を `Event.wait()` 化
- `tests/test_analyze_fill_logs.py` の tempdir fixture を `tmp_path` 化

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

## 優先順

1. `maker_price` veto/telemetry/cache ownership の最終整理
2. `ab_judgment` result/report ownership の最終整理
3. trainer/SAC/heavy_env telemetry payload 整列
4. broad 前の wait/setup 固定費削減
5. broad 最終確認

## 今の判断

- いま無理に state object 化へ進むのは早い
- Wave 2 はかなり終盤で、次は ownership の最終整理と Wave 3/4 の仕上げを並行で進めるのが安全
- `550#` は設計の基準、`551#` は実行順の基準、`521#` は全体の母艦、という役割分担で運用する
