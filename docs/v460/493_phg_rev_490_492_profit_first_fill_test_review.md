# 493# 490#-492# レビュー: Fill Test 不調の profit-first 再整理と代替打ち手

> 種別: review
> 対象: 490# `docs/v460/490_phg_second_opinion_architectural_pivot.md`, 491# `docs/v460/491_session_remediation_deep_dive.md`, 492# `docs/v460/492_codex_review_pipeline_audit.md`
> 日付: 2026-03-20

---

## 0. 総評

490#-492# は、現行 fill test が「単純な tuning 不足」ではなく、
**防御レイヤーの過積載・在庫制約・runtime 運用不整合・slow fill の逆選択**が重なって崩れている、という危機感をかなり正しく捉えている。

その点は支持できる。

一方で、主張の重み付けには補正が必要である。
特に 490# の新機構導入論や 491# の Composite Risk 有効化は、
**中長期の設計論としては理解できても、今すぐ収益を戻す順番としては前のめり**である。

まず止血すべきは、次の 4 点である。

1. `scripts/v460/lib/orchestrator_mid_cycle.py` 系の **runtime drift / 実行コード不整合**
2. `buy insufficient -> switch sell -> sell kill / route-to-kill deadlock` という **在庫・資金制約の連鎖**
3. `queue_wait_sec >= 30s` で急激に悪化する **slow fill の逆選択**
4. same-SHA / same-run に固定されていないまま因果が語られている **観測設計の弱さ**

したがって結論は次である。

- **490#-492# の問題意識は概ね正しい**
- ただし **「新しい大きな機構を先に入れる」のは時期尚早**
- 先にやるべきは **runtime 安定化・在庫制約の短絡化・slow fill 対策・可観測性整備**

---

## 1. 490#-492# で支持できる論点

### 1.1 「今の fill test はかなり悪い」はその通り

このレビュー作成時点での直近 `2026-03-10` から `2026-03-19` の `fill_records` 集計では、

- `rows=4328`
- `fills=1070`
- `fill_rate=24.7%`
- `cum post_fill_30s_pnl=-283.527`
- `avg post_fill_30s_pnl=-0.265`
- `PF=0.891`

であり、profit-first の観点では明確に苦しい。

失敗理由の上位も、単独バグより「防御・制約・流動性不足」の複合像である。

- `ranging_low_vol_skip=718`
- `sell_dynamic_kill=572`
- `skip_gate=447`
- `spread_too_narrow=393`
- `timeout=212`
- `stale_adverse_drift=169`
- `preflight_insufficient=142`
- `route_to_kill_deadlock=140`
- `no_feasible_quote=112`

この並びを見る限り、今の system は「危険なフローだけを避けている」のではなく、
**参加しなさ過ぎて spread capture と inventory recycle 自体を失っている** と見る方が自然である。

### 1.2 slow fill が悪い、は裏付けられる

492# の中核仮説のひとつである「遅い約定ほど悪い」は、実ログでも概ね支持できる。

`2026-03-17` から `2026-03-20` の fill を `queue_wait_sec` で割ると、

- `quick < 10s`: `n=139`, `avg +0.238`
- `mid 10-30s`: `n=120`, `avg -0.684`
- `slow >= 30s`: `n=30`, `avg -3.750`

であった。

この差は無視できない。市場理論的にも、これはかなり素直な結果である。
maker 注文が長く板上に残るほど、

- 情報優位者にだけ拾われる
- 板の stale quote が逆選択される
- もともと取りたかった spread より adverse move が大きくなる

という Glosten-Milgrom 型の adverse selection が強く出る。

よって、**新機構より先に TTL / micro-timeout / stale 再価格調整を詰める価値が高い**。

### 1.3 在庫・残高制約が本丸の一角、はその通り

490# と 491# が強調している `balance insufficient` は、単なる運用ノイズではない。
`results/v460/fill_test/logs/fill_test.log` では 2026-03-19 から 2026-03-20 にかけて、

- `[balance] buy insufficient, switching to sell immediately`
- `Route-to-Kill deadlock: buy insufficient, sell has balance but is kill-gated`

が反復している。

これは市場理論で言えば、
**在庫が片側に偏った状態で、alpha 判定より先に inventory repair が side choice を支配している**
ということである。

この状態では、

- buy alpha があっても buy に立てない
- sell に逃がしても sell kill にかかる
- 結果として participation だけ落ちる

ため、「モデルが悪い」以前に **inventory/funding 制約が live policy を歪めている**。

### 1.4 0.20 ceiling への張り付き自体は実在する

492# の「ceiling への圧縮」という観測自体は誇張ではない。

`results/v460/fill_test/fill_records_20260319.jsonl` の filled だけを見ると、

- `<0.15`: `1`
- `0.15-0.195`: `23`
- `0.195-0.205`: `48`

であり、filled の大半が `0.20` 近辺に集まっている。

さらに `execution_pre_clamp_offset` が記録されているものでは、

- `pre_clamp_count=50`
- `avg pre_clamp_offset=0.2763`
- `max=0.4492`

であり、上流で大きくなった offset が final clamp で 0.20 に戻されるパターンは確かにある。

この意味で、492# の「二重クランプ観測」は diagnostics としては有効である。

---

## 2. 490#-492# で補正が必要な論点

### 2.1 「新機構を足せば直る」は強すぎる

490# は

- Composite Risk Score
- 事前ロック型予算管理
- SAC への状態量移管

を強く推している。

考え方自体は理解できるが、今の順番ではない。
理由は単純で、**現在は runtime と観測設計がまだ不安定**だからである。

2026-03-19 のログには、

- `Cycle execution error: name '_sidecar_signal' is not defined`

が 17:13 から 18:09 まで連続している。
一方で現行 repo の `scripts/v460/lib/orchestrator_mid_cycle.py:141` には
`_sidecar_signal` の代入が存在する。

このズレは、ロジック不備というより **deploy/runtime drift** を疑うべき場面である。
この状態で新機構を追加しても、「何が効いたか」「何が壊したか」が更に見えなくなる。

profit-first の優先順位としては、

1. runtime を clean にする
2. same-SHA / same-run の検証単位を守る
3. 既存防御を減らして slow fill を止める
4. それでも残る問題にだけ新機構を足す

の順である。

### 2.2 `_effective_max_ratio()` の `max()` を単独犯にするのは強すぎる

492# は `scripts/v460/lib/maker_price.py:583` の `_effective_max_ratio()` を強く問題視しているが、
ここは **「設計として気に入らない」ことと「P0 バグである」ことを分けるべき** である。

現行コードでは、

- `scripts/v460/lib/maker_price.py:583` が中間 exploration 幅を確保
- `scripts/v460/lib/fill_config.py:354` が side-aware ceiling を解決
- `scripts/v460/lib/offset_pipeline.py:280` が final clamp を適用

という二段構造になっている。

したがって、正しい整理は次である。

- `max()` 単体が ceiling 無効化の直接原因、とは言い切れない
- 問題は **exploration -> final clamp -> veto/no_feasible_quote** の相互作用である
- ここを same-SHA で切り出さずに「ここが主犯」と断定するのはまだ早い

### 2.3 Composite Risk は発想はよいが、今はまだ実証不足

491# の Composite Risk 実装は、思想としては筋が通っている。
相関した soft gate をそのまま AND で直列に置くと participation を削り過ぎる、という問題意識は正しい。

ただし、492# が認めているように、

- weight は経験的 (`0.4, 0.5, 0.6, 0.7`)
- threshold `1.5` に外部エビデンスがない
- live で clean に有効化された証跡もまだ弱い

という状態である。

このため、現時点での判定は

- **設計案としては採用候補**
- **収益回復の即効薬としては未証明**

が妥当である。

特に今の top cancel は、soft gate 由来だけでなく

- `preflight_insufficient`
- `route_to_kill_deadlock`
- `no_feasible_quote`
- `timeout`

が大きいので、Composite Risk を入れても直らない部分がかなり残る。

### 2.4 492# の定量は、ログスキーマと run 混在の注意書きが必要

492# は意欲的に数値を出しているが、土台はまだ揺れる。

まず、現行 fill_records の PnL フィールドは主に `post_fill_30s_pnl` であり、
`post_fill_30s_pnl_bps` 前提の読み方をそのまま持ち込むと分析コード側でズレやすい。

また `results/v460/fill_test/fill_records_20260319.jsonl` 自体が、

- `git_sha`: `548dda24...` が `181件`, `cc6c9466f47e` が `20件`, `5608149c65e7` が `5件`
- `run_id`: `1773912878_81346ff4` が `201件`, `1773911604_0779a775` が `5件`
- `config_hash`: `None=116`, `c085a484...=74`, `5d47adb...=16`

と混在している。

この状況では、

- `ceiling 0.25` の効き
- `Composite Risk` の有効化効果
- `cross_venue` の寄与
- `sidecar stale` の影響

を 1 日単位で因果断定するのは危険である。

### 2.5 `cross_venue` は要注意だが、まだ「主犯」とは言えない

ログには `cross_venue_lead_lag_veto` を last reason にした `NO_FEASIBLE_QUOTE` が現れているので、
無視はできない。

ただし 2026-03-19 の `fill_records` をそのまま見ると、

- `cross_venue_lead_lag_applied=True`: `43`
- `cross_venue_lead_lag_applied=False`: `47`
- `cross_venue_lead_lag_vetoed=True`: `0`

であり、fill_records 側だけでは veto 連鎖の全体像を十分に再現できていない。

つまり今言えるのは、

- log 上では `cross_venue -> no_feasible_quote` の気配がある
- しかし JSONL 側の first-class attribution はまだ弱い
- よって **論点として重要だが、断定には観測強化が必要**

というところまでである。

---

## 3. 490#-492# がまだ薄く、今回補うべき盲点

### 3.1 一番危ないのは runtime drift である

2026-03-19 の `fill_test.log` では、

- `NameError: name '_sidecar_signal' is not defined`
- `Cycle execution error: name '_sidecar_signal' is not defined`

が 4 分周期で連発している。

これは単なる「sidecar が stale」という話ではない。
**途中で壊れた code path を bot が踏み続けていた** という話である。

収益改善議論より前に、これは P0 である。

- clean restart で現行 SHA と一致しているか
- 途中 hot reload / watchdog restart で旧コードが残っていないか
- `run_id`, `git_sha`, `config_hash` が fill_records に一貫して残るか

を先に締めるべきである。

### 3.2 sidecar は「入っていない」のではなく「安定して信用できない」

492# では stale 問題が触れられているが、もう少し整理した方がよい。

`fill_records_20260319.jsonl` では `sidecar_signal_status` が

- `None=116`
- `fresh=40`
- `stale=31`
- `error=19`

filled に限っても

- `fresh=27`
- `stale=28`
- `error=17`

であり、**fresh 一色ではない**。

つまり今の問題は

- sidecar を導入していない

ではなく、

- **sidecar path があるのに、fresh / stale / error / null が混ざって live 判定の品質が不安定**

ということである。

### 3.3 在庫制約は alpha 問題を覆い隠している

2026-03-19 から 2026-03-20 のログでは、

- `buy insufficient, switching to sell immediately`
- `Route-to-Kill deadlock: buy insufficient, sell has balance but is kill-gated`

が非常に多い。

これは、

- buy 側の不振
- sell kill の多発
- side 切り替えの不自然さ

を「学習器や gate の問題」と誤診しやすくする。

実際には、**在庫不足のせいで本来評価したい side が選ばれていない** ケースが混ざるため、
モデル改善の検証母集団そのものが汚れている。

### 3.4 観測フィールドの命名・整合がまだ弱い

492# のような deep dive を正しく続けるには、

- `post_fill_30s_pnl` と `post_fill_30s_pnl_bps`
- `queue_wait_sec` と `fill_delay_sec`
- `last_reason` と `cancel_reason`
- `offset_stages` と `executor_offset_stages`

のような分析単位の揺れを小さくする必要がある。

今のままだと、正しい仮説であっても analysis script ごとに読んでいる field がずれ、
同じ現象に別の名前を付けてしまいがちである。

---

## 4. 統計学・市場理論から見た本質

### 4.1 本丸は「単一バグ」ではなく participation collapse

統計的には、現在の fill test 不調を 1 つの root cause に還元するのは無理がある。
実際には、

- soft gate
- hard gate
- final clamp
- spread guard
- balance / inventory constraint
- timeout / stale drift
- occasional runtime error

が同時に乗っている。

このため、現在の損失は

- 「危険な取引をし過ぎて負けている」

だけではなく、

- **安全装置を重ね過ぎて、良い局面でも板に残れず、残った注文だけが slow fill で逆選択されている**

という構造で説明しやすい。

### 4.2 maker としては「引っ込み過ぎ」ている

市場理論で言えば、今の stack は uncertainty が高い時に supply を引っ込める方向へ強くバイアスしている。
それ自体は保守的で悪くないが、複数の veto が同時に働くと

- quote continuity が壊れる
- 在庫の自然な往復が止まる
- buy/sell の片側だけが詰まりやすくなる
- participation が減った割に、残った注文は stale 化しやすい

という悪い形になる。

ここでは「より賢い新機構」より、
**既存の防御をどこまで簡素化して continuous participation を回復するか** の方が短期収益に効きやすい。

### 4.3 slow fill の悪化は tactical problem であり、すぐ手を打てる

`slow >= 30s` が `-3.750` まで悪い以上、ここは architecture 議論の前に tactical に触る価値が高い。

具体的には、

- TTL 短縮
- micro-timeout の導入または強化
- stale drift を見た再価格調整の前倒し
- long wait bucket の強制撤退

の方が、Composite Risk を入れるより短期収益改善に直結しやすい。

---

## 5. 新機構以外の解決策

### 5.1 P0: runtime と実験単位を clean にする

最優先はこれである。

1. `NameError _sidecar_signal` を完全に消す
2. watchdog / hot reload 後に `git_sha`, `run_id`, `config_hash` が安定記録されることを確認する
3. 1 実験 1 SHA 1 config_hash を原則にして、日跨ぎ mixed-SHA 比較を避ける

これは地味だが、今の段階では最も ROI が高い。

### 5.2 P0: `buy insufficient -> switch sell` を short-circuit する

今の route-to-kill deadlock は、
**本来ノートレードでよい局面を、無理に反対 side に逃がしている** ことが一因である。

したがって、

- 資金不足 side を即座に opposite side へ逃がす
- 逃がした先が kill / no-feasible に掛かる

という流れは、一度切った方がよい。

profit-first に言えば、
「買えないなら売る」「売れないなら買う」は inventory repair の思想としては分かるが、
**反対 side に alpha がない時まで強制する必要はない**。

### 5.3 P1: micro-timeout / TTL を先にやる

492# の slow fill 観測を真正面から使うなら、次にやるべきは新機構ではなくここである。

推奨は次の順である。

1. `queue_wait_sec >= 10s` 時点で adverse drift を強く見る
2. `20-30s` で reprice or cancel を明確化する
3. `>=30s` をなるべく残さない

これは理屈だけでなく、現行の adverse selection 形状にも合う。

### 5.4 P1: `NO_FEASIBLE_QUOTE` を first-class 集計する

今の top cancel では `no_feasible_quote=112` が見えているが、
その内訳が不足している。

最低でも次は残した方がよい。

- `last_reason`
- `min_spread`
- `cross_venue veto` 起因か
- `final_clamp_hard_skip` 起因か
- `spread_too_narrow` から派生したものか

これがないと、「防御過積載」と言っても、どの防御を減らすべきかが曖昧なままである。

### 5.5 P1: offset 監査を一本化する

492# が言うように、`offset_stages` と `executor_offset_stages` が分断されているのは確かに見づらい。
ここは大改造までは不要だが、少なくとも

- pre-clamp
- post-clamp
- hard-skip threshold
- sidecar / cross_venue / EV の寄与

を 1 レコードで追えるようにした方がよい。

### 5.6 P2: 新機構を入れるなら「狭く」入れる

490# の方向性を完全に否定する必要はない。
ただし入れるなら順番がある。

先に候補になるのは、次の 2 つである。

1. **Pre-flight asset reservation**
   - 490# の中ではこれが最も実利的
   - 新しい意思決定機構というより、無駄な cycle を short-circuit する stop-loss である
2. **Composite Risk の shadow mode**
   - いきなり block に使わず、まず score を記録だけする
   - 既存 AND-chain と比較して「通してよかった局面 / 止めてよかった局面」を後で検証する

逆に、今の時点で後回しでよいのは

- SAC へ gate を丸ごと移す大工事
- 新しい複雑な cross-venue 系 veto 追加
- まだ不安定な sidecar の責務拡大

である。

---

## 6. 490#-492# への是々非々の判定

| 項目 | 判定 | コメント |
|---|---|---|
| 490# の「防御過積載」認識 | ✅ 支持 | 現状把握としてかなり正しい |
| 490# の「新機構を先に入れる」方向 | ❌ 早い | まず runtime / inventory / slow fill |
| 490# の Pre-flight asset control | ✅ 条件付き支持 | 新機構というより無駄 cycle 削減策として有効 |
| 491# の Composite Risk 実装 | 🟡 保留 | 設計案としては良いが実証不足 |
| 491# の Composite Risk 即有効化 | ❌ まだ早い | mixed-SHA と runtime drift 下では評価不能 |
| 492# の slow fill 逆選択論 | ✅ 支持 | 実ログでもかなり再現 |
| 492# の 306/421 二重クランプ観測 | ✅ 支持 | 診断として有効 |
| 492# の `_effective_max_ratio` 主犯論 | ❌ 強すぎる | 相互作用の一部であって単独犯ではない |
| 492# の ceiling 0.25 効果過小論 | 🟡 概ね妥当 | ただし same-SHA 条件で再測定が必要 |
| 492# の `cross_venue` 警戒 | 🟡 妥当 | ただし定量の根拠はまだ弱い |

---

## 7. 結論

現状の fill test は、

- alpha が完全に無い

というより、

- **runtime が汚れ**
- **在庫制約が side choice を歪め**
- **防御レイヤーが participation を削り**
- **残った slow fill が adverse selection で損を出す**

という複合崩れに近い。

このため、今の優先順位は次である。

1. **runtime drift の除去**
   - `_sidecar_signal` NameError 経路を完全停止
   - same-SHA / same-config で実験窓を固定
2. **inventory/funding short-circuit**
   - `buy insufficient -> switch sell -> kill` を切る
3. **slow fill 対策**
   - TTL / micro-timeout / stale reprice を profit-first で詰める
4. **可観測性整備**
   - `NO_FEASIBLE_QUOTE`, clamp, sidecar, cross_venue の attribution を一本化
5. **その後に新機構**
   - Pre-flight reservation を狭く入れる
   - Composite Risk は shadow mode で実証してから

要するに、490#-492# は「方向を誤っている」のではない。 
ただし、**いま必要なのは大きな新機構より、既存システムを clean にして slow fill を止めること**である。
