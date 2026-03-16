# 222# 213#–221# レビュー — デッドロック対策の実証、盲点、残存リスク

> **日付**: 2026-03-02  
> **対象**: `213#`〜`221#`, `results/v460/fill_test/` 実ログ, 現行コード (`2243c90f44cf` 系)  
> **目的**: デッドロック解消策が本当に機能しているかを、ログとコードで裏付けし、見落としと残存リスクを洗い出す

---

## 0. 総括

結論は 3 点です。

1. **218#→219#→220# の流れで、「完全停止」は実際に大きく改善した。**
2. **ただし 220# は「安全に回復した」のではなく、「危険な状態でも通す」ことで liveness を確保している。**
3. **最重要の盲点は、`per-side halt` が `balance_forced` で破られる順序バグである。**

したがって、現状は

> **デッドロックは弱まったが、保護優先順位が崩れている**

という評価が正確です。

---

## 1. 主要 findings

### 1.1 [CRITICAL] `per-side halt` が `balance_forced` によって同一サイクル内で破られる

これは実ログで確認できた、今回の最重要不具合です。

### 実ログの事実

- `2026-03-02 16:40:56`: `sell PnL -30.40bps <= -30.0bps — sell 封鎖 (cycles=15)`
- `2026-03-02 16:44:31`: `Per-side DD halt: sell blocked, switching to buy`
- その直後に `buy` 残高不足
- `2026-03-02 16:44:33`: `buy insufficient, switching to sell immediately`
- `2026-03-02 16:44:33`: `balance_forced but one_sided_balance — proceeding with sell`
- `2026-03-02 16:44:34`: 実際に `sell` 発注
- `2026-03-02 16:46:19`: `sell` 約定、`pnl=-1.15bps`

つまり、

1. `sell` は `per-side halt` で一度ブロックされた
2. しかし `buy` 不足で `balance_forced` に入る
3. `sell` へ戻される
4. **halt 中の `sell` がそのまま実行される**

### コード上の原因

- `scripts/v460/lib/fill_loop_orchestrator.py:877`
  - ここで `per-side halt` を見て `next_side` を反対側へ切替
- `scripts/v460/lib/fill_loop_orchestrator.py:1049`
  - その後の残高チェックで `buy` 不足なら即 `sell` に再切替
- `scripts/v460/lib/fill_loop_orchestrator.py:1179`
  - `balance_forced` 分岐でそのまま実行許可

**問題は、`balance_forced` で side を再変更した後に `is_side_halted()` を再評価していないことです。**

### 推奨対応

`next_side` を書き換えるたびに、最終決定直前で再度 `per-side halt` を検証すべきです。  
最低でも `balance_forced` 分岐の直後に

- `self._daily_drawdown_guard.is_side_halted(next_side)`

を再チェックし、halt 中なら

- 反対側再試行
- それも不可なら `PER_SIDE_DD_HALT`

へ落とすべきです。

---

### 1.2 [HIGH] 220# は「デッドロック根絶」ではなく、「bypass 依存の縮退運転」

ログと fill records は、220# が liveness を改善したことを示しています。  
ただし、改善の質は「健全化」ではなく「bypass 強行」です。

### 実データ（`results/v460/fill_test/fill_records_20260302.jsonl`）

- 総レコード: `192`
- 約定: `45`
- 合計 PnL: `-27.9457bps`
- 平均 PnL: `-0.6210bps`

cancel reason 上位:

- `buy_dynamic_kill`: `59`
- `sell_dynamic_kill`: `43`
- `skip_gate`: `19`
- `stale_adverse_drift`: `10`
- `toxic_fill_side_veto`: `3`
- `sell_guard_reject`: `3`

ログカウント:

- `DUAL KILL bypass`: `9`
- `PER-SIDE HALT`: `3`
- `dynamic kill probe`: `8`
- `force release`: `0`

### 解釈

- kill は依然として支配的
- 取引継続の主因は `DUAL KILL bypass`
- `force_release` は未発火で、219# の最終安全弁は未実証

これは、

> **「kill が要らなくなった」のではなく、「kill を無視して回している」**

状態です。

### 220# 単体評価（`git_sha=2243c90f44cf`）

`20260302` 日次ファイルを SHA 切り分けすると、220# 本体は以下でした。

- レコード: `10`
- 約定: `6`
- 合計 PnL: `+11.7339bps`
- 平均 PnL: `+1.9556bps`

内訳:

- `buy`: `3 fills`, `+14.3076bps`, `+4.7692bps/fill`
- `sell`: `3 fills`, `-2.5737bps`, `-0.8579bps/fill`

したがって、**220# 自体の小標本はプラス**です。  
ただし sample は小さく、しかも **sell はなお負けています**。

### 推奨対応

220# の評価は「日次合算」ではなく、**SHA 単位**で続けるべきです。  
220# を 221# の日次集計だけで裁くのは粗すぎます。

---

### 1.3 [HIGH] 221# の「本日」評価は 5 つのコード世代を混ぜており、因果がぼやけている

`results/v460/fill_test/fill_records_20260302.jsonl` には、同一日でも複数の `git_sha` が混在しています。

- `b56ba1eea015`: `67`
- `94e024e21a8c`: `41`
- `36177b2ae412` (218#): `31`
- `e9a979dbe4fc` (219#): `43`
- `2243c90f44cf` (220#): `10`

これは重要です。

`221#` の「本日 fill 数 / 本日 PnL」は、**211#〜220# の混合結果**です。  
そのため、

- 218# の純 deadlock
- 219# の partial recovery
- 220# の dual-kill bypass

を同じ箱に入れて議論しています。

### 切り分け結果

- `218#` (`36177b2ae412`): `31 records`, `0 fills`
  - まだ deadlock そのもの
- `219#` (`e9a979dbe4fc`): `43 records`, `6 fills`, `-5.8006bps`
  - 動いたが収益はまだ負
- `220#` (`2243c90f44cf`): `10 records`, `6 fills`, `+11.7339bps`
  - 小標本だが liveness と PnL は改善

### 推奨対応

今後の評価軸は日付ではなく、最低でも

1. `git_sha`
2. `run_id`
3. `date_from/to`

の 3 つで固定すべきです。  
これは `000#` の再現性方針とも整合します。

---

### 1.4 [MEDIUM] `216#` の永続化強化は正しいが、skip-heavy 時に state が長時間 stale になる

現 `results/v460/fill_test/fill_test_state.json` の末尾は:

- `saved_at_iso = 2026-03-02T11:07:26+0900`

一方、実ログは

- `2026-03-02 16:48:21`

まで進んでいます。

### 意味

state file は **5 時間以上古い**。  
これは単なる観測ズレではなく、現実に以下を意味します。

- 最新の `guard_fire_counts`
- 最新の kill manager 状態
- 最新の per-side halt 残カウント

が disk に落ちていない可能性が高い

### コード上の原因

- `scripts/v460/lib/fill_loop_orchestrator.py:1302`
  - gate block 時はここで `continue`
- `scripts/v460/lib/fill_loop_orchestrator.py:1565`
  - 通常の `save()` は末尾の progress path でしか走らない

つまり、**gate-block や early-continue が続くと、通常の state 保存に到達しません。**

218#〜220# の対象そのものが skip-heavy なので、ここは地味に重要です。

### 推奨対応

以下のいずれかを入れるべきです。

1. `gate_block` でも N サイクルごとに簡易 state save
2. `_effective_sleep()` 直前に「一定時間経過なら save」
3. `guard` 系 state だけ別 cadence で保存

---

### 1.5 [MEDIUM] 218#〜220# の新 liveness イベントが計測不足

`216#` で `guard_fire_counts` は入りましたが、現在の計測対象は以下中心です。

- `dd_halt`
- `hard_skip_utc`
- `toxic_veto_set`
- `toxic_veto_block`
- MCB/SAD 系

一方、今回の主役である以下は未計測です。

- `dual_kill_bypass`
- `per_side_halt_switch`
- `deadlock_warning`
- `dynamic_kill_probe`
- `force_release`

### 実害

現状、218#〜220# の効き方を定量化するには

- 生ログ grep
- 日次 JSONL の cancel reason 集計

に戻るしかありません。

### 推奨対応

`guard_fire_counts` か、あるいは `deadlock_metrics` を別 dict として追加し、

- bypass 回数
- probe 回数
- force release 回数
- per-side halt switch 回数

を run 中に残すべきです。

---

## 2. 追加の見落とし

### 2.1 `per_side_halt` の発動自体が artifacts に残りにくい

`results/v460/fill_test/fill_records_20260302.jsonl` では、

- `per_side_dd_halt`: `0`

です。

しかしログでは `PER-SIDE HALT` が `3` 回あります。

これはコード上、

- 片側だけ halt のときは side を切り替えるだけ
- **両側 halt のときだけ** `PER_SIDE_DD_HALT` record を書く

ためです。

結果として、**「halt は起きたがスイッチされた」ケースが JSONL 監査に乗りません。**

今回は 1.1 の順序バグまであるので、なおさら見えにくくなっています。

### 2.2 `force_release` は未発火。219# の最終安全弁はまだ runtime 未検証

ログカウントでは:

- `force release: 0`

です。

つまり、219# の

> 「最悪でも 5 probes で完全復帰」

は、現時点ではまだ**実運用で証明されていません**。

しかも `ztb/risk/sell_dynamic_kill.py:116` で `track()` が走るたびに

- `_consecutive_probes = 0`

へ戻るため、**悪い fill が断続的に入るループでは force release に到達しない**可能性があります。

これは deadlock そのものではないですが、**219# の期待値を少し盛りすぎています。**

---

## 3. 現時点での「デッドロック可能性」点検

### 3.1 残る logical deadlock 候補

1. **`per-side halt` → `balance_forced` 再侵入**
   - 現在進行形で実例あり
   - `scripts/v460/lib/fill_loop_orchestrator.py:877`
   - `scripts/v460/lib/fill_loop_orchestrator.py:1049`
   - `scripts/v460/lib/fill_loop_orchestrator.py:1179`

2. **`unknown regime` 連続バイパスのカウンタ非永続化**
   - `CycleGateAggregator._consecutive_unknown_blocks` は process-local
   - 再起動を挟むと 10-cycle bypass への到達がリセットされる
   - `scripts/v460/lib/cycle_gate_aggregator.py:111`
   - 重大度は低いが、long unknown 時の stall は伸び得る

3. **skip-heavy 時の state stale による「回復進捗の巻き戻り」**
   - state save が遅れたまま再起動すると、anti-deadlock 進捗が古い snapshot に戻る
   - これは pure deadlock ではないが、運用上は「また詰まった」に見える

### 3.2 現時点で見当たらないもの

**古典的なスレッド/ロックの相互待ちは見当たりません。**

理由:

- lock は file-lock だが、`scripts/v460/lib/fill_loop_orchestrator.py:642` の heartbeat task が 60 秒ごとに更新
- `scripts/v460/lib/lock_manager.py:81` の stale 判定は heartbeat ベース
- `asyncio.Lock` のような相互待ち構造は今回の経路にない

したがって、今の主問題は **concurrency deadlock ではなく control-flow deadlock** です。

---

## 4. 次にやるべきこと

### P0

1. `balance_forced` 後に `per-side halt` を再チェックする
2. `per-side halt switch` を JSONL または state に残す
3. `220#` 評価は `git_sha=2243c90f44cf` で独立継続する

### P1

4. `gate_block` / `balance_forced_skip` / `preflight_insufficient` でも軽量 state save を入れる
5. `dual_kill_bypass`, `dynamic_kill_probe`, `force_release` をカウントする
6. `force_release` の到達条件を「空振り probe のみ」から、「断続的な悪い fill ループ」でも効く形に見直す

### P2

7. `sell` 側だけ別で kill 閾値・DD 閾値・offset を再調整する
8. `221#` のレポートは日次集計版とは別に、`git_sha` 別の小標本評価版を併記する

---

## 5. 最終評価

`218#` は deadlock を「検知」した段階です。  
`219#` は deadlock を「揺り動かす」段階です。  
`220#` は deadlock を「通過させる」段階です。

ここまでは実際に進んでいます。

ただし現状は、

- deadlock は減った
- しかし bypass が強く
- その結果、守るべきガード (`per-side halt`) を破る場面が出た

という状態です。

したがって、次のフェーズでやるべきことは新しい bypass 追加ではありません。  
**「どのガードが最終優先か」を固定し、保護の優先順位を崩さないこと**です。

まず直すべきは、`per-side halt` を `balance_forced` より上位に置き直すことです。


---


---

---

---

## 6. 追記: 213#～221# に対するセカンドオピニオンと市場理論に基づく最終審判 (Gemini 3.1 Pro)

### 6.1 総評: 「生存性（Liveness）」と「安全性（Safety）」の致命的混同

Codexの検証と指摘（とりわけ本ドキュメント222#の内容）は、システム工学・データ解析の両面において完璧に正しい。対して、218#〜220#の「デッドロック解消アプローチ」には、金融取引システムとして**「決して越えてはならない一線」**を越えた形跡が見られる。
プログラムが「停止することなく動き続けること（Liveness）」と、「市場の毒性から身を守ること（Safety）」が相反した場合、トレードシステムにおいては**常にSafetyが優先**されねばならない。現在実装されている「Dual-kill bypass」や、「残高不足によるGuardの強行突破」は、警告音が鳴り響く計器盤のランプを叩き割って「車は正常に走っている」と錯覚しているに過ぎない。

### 6.2 Dual-Kill Deadlock Breaker の論理的破綻と「相場のレジーム崩壊」

220#で実装された「BuyとSellの両方がKill判定されたら、両方とも強行突破する（Dual-kill bypass）」というロジックは、HFTや統計的裁定取引の常識から完全に逸脱している。
Mandelbrot（マンデルブロ）のフラクタル市場仮説や、統計的裁定取引（StatArb）の世界では、「双方向同時にエッジがマイナスになる（Dual Killが発動する）」状況は、単なるバグやデッドロックではなく、**「市場のレジームが未知の領域（Uncharted Territory）に入り、手持ちのモデルの前提が崩壊した」**ことを示す最強のシグナルである。
Kelly Criterion（ケリー基準）において、期待値（EV）がマイナス、かつ不確実性が最大化しているときの最適ベットサイズは「厳密にゼロ（完全停止）」である。これを「デッドロックだから」というシステム都合の理由でバイパスするのは、統計的自殺に他ならない。「両側が燃えているなら、取引を止める」のが唯一の正解である。Dual-kill bypassは即刻削除すべきである。

### 6.3 経路依存性バグ（Path Dependency Bug）と Avellaneda-Stoikov の完全喪失

Codexが1.1で指摘した「`per_side_halt` が `balance_forced` によって破られる」問題の深刻さは計り知れない。
Avellaneda-Stoikovモデルにおいて、片側にポジションが偏った際（インベントリ・リスク増大時）は、予約価格（Reservation Price）を極端にシフトさせて非対称なクオートを行う。だが、**「巨額の損失を出して片側Haltが発動しているサイド」**に向かって、単に「そっちのトークンが欲しい（または売りたい）から」という理由だけで発注を強制（Forced）させるのは、インベントリを調整するメリットよりも明らかに逆選択コスト（Adverse Selection）の被害が上回っている証拠である。
制御フロー（Control-flow）上の順番ミスというコード上のバグに留まらず、これは「資金管理（Money Management）」の根幹を否定している状態だ。実装指示通り、`balance_forced` によって `next_side` が反転した直後で必ず再度の `is_side_halted` 検証を行うよう修正せよ。

### 6.4 評価指標の汚染（データ・マージ）：A/Bテストの基本原則違反

Codexが1.3で指摘した「1日の間に5つの異なるコミット（Git SHA）の成果をごちゃ混ぜにして評価している」という事実は、データサイエンスとして致命的だ。
市場の時間的非定常性（Non-stationarity）を扱うシステム開発において、**ロジックの因果関係（Causality）**を正しく推論するためには、エポック（検証期間とSHA）を厳密に分離しなければならない。利益が出ているのか損失が出ているのか、どの変更が効いているのかを「日次集計」で曖昧にする運用は直ちに改め、**「Git SHA × Run ID ベースのマイクロエポック評価」**を標準の分析基盤とせよ。

### 6.5 今後のアクション（妥協なき是正要請）

前回（214#）の反省を踏まえ、純粋な技術と数理・統計ロジックのみを以て要求する。

1. **[P0] Dual-Kill Bypass (Gate 4/5 貫通) の即時廃止**: 両サイドが規定の損失閾値を超えたなら、それは正常な「Halt（全停止）挙動」である。これをデッドロックと呼ぶのをやめ、正しくシステムを休止させろ。
2. **[P0] Balance Forced と Side Halt の順序修復**: `fill_loop_orchestrator.py` の `balance_forced` で `next_side` を上書きした際、必ず `self._daily_drawdown_guard.is_side_halted(next_side)` を評価し、Trueなら取引を完全にSkipさせろ。Haltされているサイドを通すな。
3. **[P1] 状態永続化（State Persistence）の適時保存**: Skipが連続する際にStateが5時間も古いまま放置されるのは、障害復旧のアンチパターンである。Gate BlockでContinueする直前でも、最終保存から一定時間（例: 5分）経過していれば軽量にStateを保存するフックを設けろ。

これらは議論の余地のない、強牢なトレードシステムに必要な絶対要件である。即座の実装反映を推奨する。
