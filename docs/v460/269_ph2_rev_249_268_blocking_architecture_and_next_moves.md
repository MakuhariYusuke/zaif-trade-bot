# 269# 249#–268# レビュー — ブロッキング構造・機能性・保守性・市場理論の再点検

> **日付**: 2026-03-04
> **対象**: `249#`〜`268#`, 現行コード, `results/v460/fill_test/` 実ログ
> **観点**: blocking 構造の改善状況, 残存 deadlock, 収益改善に直結する追加施策

---

## 0. 総括

結論は次の通りです。

1. `249#`〜`268#` で、**日次 DD の再武装・JST リセット・quiescence・guard 分類・型安全化**は前進しています。
2. ただし、**blocking の根本問題は未解決**です。停止の主因が「aggregate DD halt」から「per-side halt + balance_forced」へ移っただけです。
3. 現在の支配的ボトルネックは市場ではなく、**システム内部のブロッキング相互作用**です。
4. `258#` / `264#` / `266#` の市場理論追加は実装としては有益ですが、**現行 live では大半が未配線または無効のまま**で、現時点の収益・liveness 改善にはほぼ効いていません。
5. 最優先は「新しい理論を足すこと」ではなく、**在庫解消用の逃がし経路を top-level で明示的に作ること**です。

要するに、

> **安全装置は増えたが、在庫を逃がす責務が依然として通常の alpha 取引パスに混在している**

のが本質です。

---

## 1. 確認できた改善

### 1.1 268# の JST 日次リセットは実ログ上で確認できる

`results/v460/fill_test/logs/fill_test.log` では、以下が確認できます。

- `2026-03-04 09:00:11` に `Day reset: 20260303 → 20260304`
- 同時に `sell_dynamic_kill still active ... resetting kill state for clean day start`

したがって、`268#` の主修正である

- `_utc_today()` 依存の解消
- JST 基準のリセット
- 日替わり時の kill 競合補正

は、少なくとも **3/4 09:00 の切替では機能した** と見てよいです。

### 1.2 観測性は改善している

`fill_test_state.json` の `guard_fire_counts` / `guard_category_totals` が復元・保存されており、`244#` の guard 分類も使えています。

レビュー時点の状態では:

- `market = 39`
- `system = 113`
- `recovery = 25`

です。

この比率は重要です。**詰まりの主因が市場条件ではなく、システム側制御である**ことを、実測で裏付けています。

### 1.3 249#〜250# の「No Trade = 正常」思想は一部機能している

`249#` の `dual_kill_quiescence_enabled` と `242#` 系の quiescence ログは実装されています。  
`scripts/v460/lib/cycle_gate_aggregator.py:235` の分岐で、dual kill 時の旧 bypass を止める思想自体は入っています。

ただし、後述の通り **それでも別経路の liveness 破綻は残っています**。

---

## 2. 主要 findings

### 2.1 [CRITICAL] blocking の主因は「aggregate DD halt」ではなく、現在は「per-side halt + balance_forced」になっている

これは今回の最重要点です。

レビュー時点の `results/v460/fill_test/fill_test_state.json` では、

- `daily_drawdown_state.halted = false`
- `daily_drawdown_state.side_halted_sell = true`
- `daily_drawdown_state.cooldown_released = false`

でした。

つまり、**246# の cooldown release が効く対象（aggregate halt）は、今の詰まりの主因ではありません。**

現行 run `1772594447_fde90973` (`git_sha=4b863d211de0`) でも、実ログは以下です。

- `12:57` `balance_forced → sell is per-side halted — refusing to bypass halt`
- `13:03` 同上
- `13:09` 同上
- `14:05` 以降も同様のループ継続

さらに、同 run の `fill_records_20260304.jsonl` を見ると、レビュー時点で

- `28 records`
- `2 fills`
- `20` 件が `cancel_reason=per_side_dd_halt`

です。

これは、

> **aggregate halt の長時間停止は軽減したが、per-side halt 側に liveness 破綻が移った**

ことを意味します。

### 2.2 [CRITICAL] 234# の degraded liquidation は、支配的 deadlock パターンでは到達不能

`234#` の縮退清算モード自体は有用です。実際、3/4 朝の sell dynamic kill には効いています。

しかし、現在の主 deadlock では **設計上そのコードに到達しません。**

理由は順序です。

- `scripts/v460/lib/fill_loop_orchestrator.py:1699`
  - balance_forced 後に `is_side_halted(next_side)` を再チェック
- `scripts/v460/lib/fill_loop_orchestrator.py:1701`
  - side halt なら即 reject
- `scripts/v460/lib/fill_loop_orchestrator.py:1732`
  - そのまま `continue`

一方で degraded liquidation に入るのは、

- `scripts/v460/lib/fill_loop_orchestrator.py:2166`

以降です。

つまり、**per-side halt で弾かれた時点で gate 集約フェーズに入れない**ため、
`234#` の liveness 弁は Pattern A の本丸では働きません。

これは構造的にかなり重い問題です。

> **「逃がし弁」は実装されているが、最も必要な箇所より後段に置かれている**

という状態です。

### 2.3 [HIGH] `balance_forced_halt_recheck` 経路では state 保存が再び stale 化している

これは見落としです。

`223#` で gate block 経路に、`225#` で normal cycle 経路に時間ベース state save が入りました。  
しかし、現在もっとも長く滞在している `balance_forced_halt_recheck` 経路には同等の保存がありません。

コード上でも、

- `scripts/v460/lib/fill_loop_orchestrator.py:1726`
- `scripts/v460/lib/fill_loop_orchestrator.py:1728`

では batch flush はありますが、state save はありません。

レビュー時点では、

- `fill_test_state.json.saved_at = 2026-03-04T14:02:12+0900`
- 同時に `fill_test.log` は `14:59` 台まで進行

でした。

つまり、

> **最も詰まりやすい blocking 経路で、監視 state だけ止まる**

状態です。

これは監視だけの問題ではなく、

- 復旧時の状況認識
- watchdog の判定
- 「今どの guard が支配的か」の分析

まで歪めます。

### 2.4 [HIGH] per-side halt は「時間で解除するだけ」で、損失負債を整理していない

`224#` の recovery は lot 縮小としては妥当です。  
しかし、**halt の原因になった side の損失負債そのものは残ったまま**です。

コード上、

- `scripts/v460/lib/daily_drawdown_guard.py:323`
- `scripts/v460/lib/daily_drawdown_guard.py:330`

では、release 時にやっているのは

- `side_halted_* = False`
- `side_recovery_remaining_* = N`

だけです。

`daily_pnl_bps_sell` / `daily_pnl_bps_buy` のアンカー再設定はありません。

実ログでも、

- `13:27` sell halt release
- `13:45` sell fill, `pnl=-5.46bps`
- `13:47` `PER-SIDE HALT: sell PnL -37.61bps <= -30.0bps`

と、**1 回の負 fill で即再 halt** しています。

これは `268#` が指摘した「release → 1-2 fills → 即再 halt」が、現行でも継続している裏付けです。

### 2.5 [HIGH] `sell_dynamic_kill` の probe / force-release はまだ生きており、quiescence 思想と衝突している

`250#` は probe 廃止「基盤」ですが、実装としてはまだ残っています。

コード上、

- `ztb/risk/sell_dynamic_kill.py:92`
  - `max_stale_kill_cycles = 10`
- `ztb/risk/sell_dynamic_kill.py:97`
  - `max_force_release_probes = 5`
- `ztb/risk/sell_dynamic_kill.py:410`
  - stale 到達で probe 発火
- `ztb/risk/sell_dynamic_kill.py:417`
  - 条件次第で force release

です。

しかも、これらは **現行 YAML から無効化しにくい** 状態です。  
`configs/v460/fill_test.yaml` には `max_stale_kill_cycles` / `max_force_release_probes` の配線がありません。

実ログでも 3/4 に、

- `07:27` `sell dynamic kill probe`
- `07:34` `sell dynamic kill probe`
- `08:16` `sell dynamic kill probe`
- `08:43` `sell dynamic kill probe`
- `08:56` `sell dynamic kill probe`

が出ています。  
`fill_test_state.json.guard_fire_counts.dynamic_kill_probe_sell = 5` とも一致します。

したがって、現状は

> **No Trade を正常化したいのに、内部ではまだ「一定時間ごとに穴を開ける」**

設計です。

### 2.6 [MEDIUM] 258# / 264# / 266# の市場理論実装は、現行 run ではほぼ dormant

これは実装の否定ではありません。  
むしろコード資産としては有益です。

ただし、現行の `configs/v460/fill_test.yaml` には、少なくとも今回確認した範囲で

- `as_reservation`
- `kelly`
- `kyle_lambda`
- `amihud`
- `vpin_continuous`

の設定がありません。

対応する `FillTestConfig` のデフォルトは、

- `scripts/v460/lib/fill_config.py:365`
- `scripts/v460/lib/fill_config.py:489`
- `scripts/v460/lib/fill_config.py:500`
- `scripts/v460/lib/fill_config.py:504`
- `scripts/v460/lib/fill_config.py:509`

で、いずれも `False` 側です。

また、`fill_records_20260304.jsonl` のレビュー時点 65 件では

- `macro_trend != null` は `0`
- `macro_aligned != null` も `0`

でした。

つまり、大局観を掴むための部品は増えていますが、**live の意思決定にまだ乗っていません。**

### 2.7 [MEDIUM] 保守性は改善したが、blocking ロジックはまだ巨大な横断責務のまま

`260#` / `261#` / `262#` / `265#` の refactor は正しい方向です。  
ただし、blocking の本丸は依然として複数巨大ファイルに分散しています。

レビュー時点の行数:

- `scripts/v460/lib/fill_loop_orchestrator.py`: `2515`
- `scripts/v460/lib/fill_cycle_executor.py`: `1378`
- `scripts/v460/lib/maker_price.py`: `1407`
- `scripts/v460/lib/skip_gate_evaluator.py`: `1258`
- `scripts/v460/lib/cycle_gate_aggregator.py`: `747`

blocking 判定だけでも、

- DD
- per-side DD
- dynamic kill
- cycle gate
- ML gate
- balance rescue
- toxic veto

が複数モジュールを横断しています。

この構造では、**1 箇所の止血が別経路の deadlock を誘発しやすい**ままです。

---

## 3. 根本原因の整理

現状の根本原因は、単に「guard が厳しすぎる」ことではありません。

### 3.1 alpha 取りと inventory 解消が同じ経路に混在している

今の bot は、

- 通常の maker alpha を取りに行く注文
- 残高回復のために在庫を解消すべき注文

を、基本的に同じ gate 群で処理しています。

そのため、

- alpha 目的では正しい block
- inventory 回復目的では誤った block

が同じように適用されます。

`234#` degraded liquidation はその補修ですが、前述の通り **主 deadlock では到達不能** です。

### 3.2 safety が各層で局所最適になっている

今の制御は各層が個別に「止める」判断をします。

- per-side halt: side の追加損失回避
- dynamic kill: rolling 負エッジ回避
- skip_gate: 低期待値回避
- balance_forced: 実行不能回避

各判断は局所的に正しいですが、**在庫保有リスクを統括する層がない**ため、
全体としては「何もしない」が選ばれやすくなっています。

しかし、BTC を持ったまま売れない状態では、

> **No Trade は中立ではなく、方向リスクの受動保有**

です。

### 3.3 評価単位が still too short

`249#`〜`266#` で方向 α の思想は導入されましたが、live 実装の支配判断はなお

- microstructure
- 直近 fill PnL
- fill 後 30s/60s/120s

中心です。

大きな値動きへの追随を本当にやるなら、少なくとも

- 5分
- 15分
- inventory 保有コスト

を main loop の上位意思決定に昇格させる必要があります。

---

## 4. 推奨施策

### 4.1 P0: `Inventory Escape Mode` を top-level で新設する

最優先です。

今必要なのは、guard をさらに足すことではありません。  
**在庫解消専用の別モード**です。

具体的には、

1. `balance_forced`
2. 対向 side が `per-side halt`
3. 反対側残高が不足
4. inventory が target band を超過

の組み合わせで、通常 alpha パスではなく

> `Inventory Escape Mode`

に入るべきです。

このモードでは、

- 目的: PnL 最大化ではなく在庫解消
- size: `min_lot` か `recovery_scale` の小ロット
- price: wide offset の maker 指値
- bypass: `skip_gate` / `sell_dynamic_kill` の一部を限定緩和

とし、**per-side halt を「通常 side 封鎖」から「alpha side 封鎖」に格下げ**します。

重要なのは、これを `cycle_gate_aggregator` の後段ではなく、**`fill_loop_orchestrator.py:1699` より前で分岐する**ことです。

### 4.2 P0: `balance_forced_halt_recheck` に state save を入れる

`fill_loop_orchestrator.py:1726` のパスにも、`223#` と同じ時間ベース保存を入れるべきです。

最低限必要なのは:

- `_STATE_SAVE_INTERVAL_SEC` 経過時の保存
- `last_block_reason`
- `last_block_side`
- `last_block_mode` (`balance_forced_halt_recheck`)

です。

これをしない限り、**最も詰まりやすい経路で観測が壊れ続けます。**

### 4.3 P1: per-side halt を「累積絶対値」から「release 後アンカー方式」に変える

現状の「当日累積 PnL」のままでは、release しても実質的に債務超過です。

より筋が良いのは以下です。

1. release 時点で `side_release_anchor_pnl` を記録
2. 再 halt 判定は `current_side_pnl - anchor` の追加損失だけで見る
3. あるいは EWMA / time-decay にして古い負債を減衰させる

これなら「8分で 1 fill → 即再 halt」という不自然な自己ループが減ります。

### 4.4 P1: probe / force-release は YAML で殺せるようにするか、既定で 0 に寄せる

今の主題は liveness ですが、**toxic flow への不用意な probe は逆方向の穴**です。

少なくとも、

- `max_stale_kill_cycles`
- `max_force_release_probes`

は YAML に露出すべきです。

その上で、

- live 本番: `0`
- 検証 run: 明示的に有効化

に分ける方が安全です。

`No Trade = 正常` を採るなら、probe は「たまたま残っている legacy fallback」ではなく、
**意図的に opt-in する機能**に変えるべきです。

### 4.5 P1: `Liveness Budget` を導入し、緩和順序を固定する

今の問題は「何をどこまで緩めるか」が各層に散っています。

top-level で次を持つべきです。

- `minutes_since_last_fill`
- `inventory_excess`
- `blocked_same_reason_count`
- `holding_risk_score`

その上で、緩和順序を固定します。

1. `skip_gate`
2. `narrow spread`
3. `dynamic kill`
4. `per-side halt` の liquidation-only override

これで「その場しのぎの bypass」が減り、挙動が読めます。

### 4.6 P2: 市場理論の追加は、liveness 修正後に「inventory risk」へ寄せて使う

`258#` / `264#` / `266#` の実装は、今の主問題を直接は解かないものの、次の用途には使えます。

1. `AS reservation`
   - 通常 alpha 用ではなく、inventory escape 時の price 設定に使う
2. `Kelly`
   - 通常ロットの増減より、`escape lot` の上限に使う
3. `Kyle λ` / `Amihud`
   - 「今は逃げると滑る」場面の liquidation 抑制に使う
4. `VPIN continuous`
   - hard skip ではなく、escape mode の offset 増幅に使う

要するに、理論拡張は **alpha 改善用の飾り**ではなく、
**在庫解消の安全な価格形成**に転用した方が、現状には効きます。

---

## 5. 追加で見落としやすい点

### 5.1 `degraded_liquidation` の責務が曖昧

現状は

- kill gate blocked + balance_forced

のときの救済ですが、実際には

- per-side halt
- inventory escape
- force liquidation

まで含む文脈で使われがちです。

この曖昧さが、到達不能や順序バグの温床です。

`degraded_liquidation` は名前を含めて、

- `InventoryEscapePolicy`
- `KillGateRescue`

のどちらかに責務分離した方がよいです。

### 5.2 macro 系の null は「未使用フィールド」ではなく、未統合のサイン

`fill_records` に `macro_trend` / `macro_aligned` のフィールドがありながら、本日の記録は全件 `null` でした。

これは単なる出力の空欄ではなく、

> **大きな相場感を使うための導線が live で閉じている**

という意味です。

今の「大値動きに付いていけない」は、この点とも整合します。

### 5.3 refactor は進んだが、policy 抽出がまだ足りない

今必要なのは util 抽出より、**意思決定 policy の抽出**です。

特に分離すべきは:

- `BlockingPolicy`
- `InventoryEscapePolicy`
- `LivenessSupervisor`

です。

ここを分けない限り、`fill_loop_orchestrator.py` の肥大化は再発します。

---

## 6. deadlock / liveness trap 候補

### 6.1 現在も観測中のもの

1. `buy` 不足 + `sell` per-side halt
   - 現行の主 deadlock。実際に継続観測。

### 6.2 まだ潜在的なもの

1. `sell` 不足 + `buy` per-side halt
   - 現行の鏡像。発生頻度は低いが構造上は同型。
2. `aggregate halt` + `dual kill quiescence` + `balance_forced`
   - 250# 側の degraded で拾えるかが未十分検証。
3. `state stale` + watchdog 判定
   - stop ではないが、誤監視・誤復旧の温床。

古典的な mutex/lock の意味での deadlock というより、

> **control-flow 上の logical deadlock / liveness trap**

がまだ中心です。

---

## 7. 最終判断

`249#`〜`268#` は、闇雲に複雑化したわけではなく、**必要な止血と品質改善**をかなり入れています。  
その点は正当です。

ただし、現時点では

- aggregate DD 問題はかなり改善
- その代わり、side-halt 系の blocking が主障害化
- しかも escape 経路が順序上そこに届かない

という段階です。

したがって評価は、

> **「改善は本物だが、根本解決にはまだ届いていない。今は blocking architecture の再設計が最優先」**

です。

次にやるべきことは、追加の guard ではありません。

1. `Inventory Escape Mode` の top-level 分離
2. `balance_forced_halt_recheck` の state 保存
3. per-side halt のアンカー方式化
4. probe / force-release の明示 opt-in 化

この順です。

