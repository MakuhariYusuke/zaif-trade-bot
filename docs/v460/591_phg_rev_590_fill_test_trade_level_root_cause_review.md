# 591# 590#レビュー: Fill Test 再起動前の取引レベル根本原因点検

> **Date**: 2026-03-24
> **Scope**: `590_fill_test_log_analysis_5day.md` の検証、`fill_records_20260319-20260323.jsonl` の再集計、既存分析スクリプトによる trade-level 根本原因整理
> **Method**: `scripts/v460/analysis/analyze_fill_logs.py` / `scripts/v460/analysis/tail_loss_analysis.py` / raw `fill_records` spot check

---

## 0. 総括

`590#` の危機感は妥当である。実際、`2026-03-19` から `2026-03-23` の Fill Test は依然として赤字であり、約定率も高くない。

ただし、`590#` をそのまま restart 前の設計根拠に使うのは危険である。理由は単純で、**集計窓が mixed-SHA / mixed-run であり、しかも最も重要な損失経路は「sell 全体崩壊」ではなく `decision_path=ev_offset` に集約した adverse selection だからである。**

今回の再確認では、以下の3点が本質だった。

1. **5日窓の本丸は buy/sell 一般論ではなく、tail loss がほぼ全面的に AS 起因であること**
2. **負け方の共通パターンは side を問わず `ev_offset` 経路で、広め spread・短期トレンド追随・ranging 帯参加が重なること**
3. **現行の比較的新しい `c164d21d` では sell は既に黒字化しており、残課題は主に `buy × ranging` と `CV/sidecar の局所的誤作用` に縮退していること**

従って、restart 前の主眼は「新機構追加」ではなく、**post-589 の clean run を取り、`ev_offset` が toxic fill を通す条件を side 別に削ること**である。

---

## 1. 重要所見

### 1.1 CRITICAL: 590# は restart 判定用の因果資料としてはまだ粗い

`analyze_fill_logs.py` の再集計では、`--date-from 2026-03-19 --date-to 2026-03-24` で `total_records=2507`, `filled=800`, `sum_pnl30_bps=-129.8118` だった。一方、`590#` の本文は `2504 / 799 / -132.2bps` であり、既に数値がずれている。

さらに、現時点で `results/v460/fill_test/fill_records_20260324.jsonl` は存在しない。したがって、`590#` の日次表にある `03-24` 行は現ファイル状態と整合しない。`generated_utc` も `2026-03-23T20:03:39+00:00` であり、**実質的には 03-19〜03-23 の5日窓**として扱うべきである。

この点は細かいようで重要で、`000#` の運用方針が求める **same-SHA / same-YAML / reproducible analysis** から外れる。よって `590#` は「症状把握」には使えるが、「この変更が効いた」と言うための資料にはまだならない。

### 1.2 CRITICAL: tail loss の本丸は `sell` 一般ではなく `ev_offset + adverse selection`

`tail_loss_analysis.py --date-from 2026-03-19 --date-to 2026-03-24` の再実行では、以下が共通していた。

- `sell`: tail 36件、**AS率 100%**、`decision_path=ev_offset` 100%
- `buy`: tail 45件、**AS率 100%**、`decision_path=ev_offset` 100%

これは極めて重い。`590#` は sell 側や `macro_strong_down` を強めに問題視しているが、実際の損失の芯はもっと狭い。

> **「alpha が無い」より先に、「offset を決める経路が toxic fill を通している」ことが問題である。**

side 別の差はあるが、両 side ともテールは AS で説明される。したがって、売り全体を大改造するより、**`ev_offset` 経路の参加条件を削る方が期待値改善に直結する。**

### 1.3 HIGH: 現行 `c164d21d` では sell は既に崩壊していない

`analyze_fill_logs.py --date-from 2026-03-23 --date-to 2026-03-24 --git-sha c164d21d` では以下だった。

- 全体: `73 fills / avg +0.86bps`
- `buy`: `41 fills / avg +0.14bps`
- `sell`: `32 fills / avg +1.79bps`

つまり、`590#` の5日窓で見える sell 悪化のかなりの部分は、古い SHA 群の悪化や mixed window の影響を含んでいる。現在の課題は「sell 全面崩壊」ではなく、**current SHA ではむしろ `buy × ranging` が残課題**である。

この補正は restart の方向を変える。sell 救済策を全面投入するより、まず `buy` の entry 条件を削る方が合理的である。

### 1.4 HIGH: `sidecar error の方が良い` は事実だが、即 `sidecar は逆効果` とは言えない

5日窓では `sidecar_signal_status=error` が `fresh/stale` より良かった。これは `590#` の問題提起として価値がある。

ただし解釈は慎重にすべきである。実際に `c164d21d` では:

- `fresh`: 19 fills, mean -1.0bps
- `stale`: 54 fills, mean +1.5bps

一方で raw record を見ると、`fresh` の負けトレードは low-confidence sidecar が小さく差し込まれている例が多い。例えば `results/v460/fill_test/fill_records_20260323.jsonl:96` では、

- `buy`
- `regime=ranging`
- `post_fill_30s_pnl=-1.5857bps`
- `sidecar_signal_status=fresh`
- `sidecar_bias=-0.7209`
- `sidecar_confidence=0.1068`
- `sidecar_offset_bps=-0.0203`

となっており、**sidecar 自体が強く悪化させたというより、低信頼の補正がほぼ効かず、本体の `ev_offset` 経路がそのまま toxic fill を通した**と読む方が自然である。

結論として、現時点で言えるのは「sidecar を強めるな」までであり、「sidecar を反転させろ」まではまだ言えない。

### 1.5 HIGH: `CV lead-lag は悪い` も一般論にし過ぎない方がよい

5日窓全体では `CV Applied` がやや悪い。しかし内訳を見ると歪みがある。

- 5日窓全体では `CV applied = 255/799 fills`
- `buy` では `236/447 fills` と多く、mean `+0.08bps`
- `sell` では `19/352 fills` と少なく、mean `-0.72bps`
- `c164d21d` の `buy` では `CV applied` が `-3.21bps` と明確に弱い

つまり broad には「CV 全面停止」ではなく、**current SHA の `buy` で CV 適用条件が悪い**という方が近い。

さらに raw では `results/v460/fill_test/fill_records_20260323.jsonl:111` のように、

- `sell`
- `cross_venue_lead_lag_applied=true`
- `cross_venue_lead_lag_cap_hit=true`
- `post_fill_30s_pnl=-2.9642bps`
- `adverse_selected=true`

という例がある。これは CV が働いたというより、**CV を載せても cap に当たって no-op 化し、toxicity だけ残った**形に近い。

したがって、CV を一律 disable するより、

- `buy` の CV 適用条件見直し
- `cap_hit=true` 時の意味論点検
- `sell` で adverse side のとき veto 寄りにするかの再検証

の順が筋である。

### 1.6 MEDIUM: 590# の telemetry 欠損原因説明は弱い

`590#` は `spread_capture_bps` / `adverse_selection_cost_bps` 欠損を「稼働 SHA が古いから」と整理しているが、ここは言い切らない方がよい。

現 HEAD では `scripts/v460/lib/fill_record_builder.py:93` と `scripts/v460/lib/fill_record_builder.py:94` で両フィールドを builder に積んでおり、`ztb/metrics/fill_quality.py:75` と `ztb/metrics/fill_quality.py:76` で `FillRecord` 側も受けている。

それにもかかわらず raw `fill_records` には現行 `c164d21d` 行でも当該キーが見えない。従って安全な結論は、**単なる分析漏れではなく、runtime/deploy 混在または古いプロセス継続の疑いがある**である。

ここを「最新 SHA で自然解決」と軽く書くと、restart 後も同じ可観測性断線を見逃す危険がある。

---

## 2. 590# の支持できる点

### 2.1 `ranging` が主戦場で、そこで負けている

これはその通りである。5日窓では:

- `ranging: 711 fills / avg -0.36bps`
- `trending_up: 49 fills / avg +1.36bps`
- `trending_down: 39 fills / avg +1.55bps`

約定のほとんどが `ranging` に集中している以上、`ranging` で勝てないなら全体も勝てない。ここは `590#` の主張を支持する。

### 2.2 高め offset の方が broad には良い

5日窓 broad では、`high offset (>0.25)` の方が良かったという整理も概ね支持できる。特に sell で high offset が良い傾向はある。

ただし、これをそのまま `base_offset 全面引き上げ` に飛ばすのは危ない。`c164d21d` の負け trade を見ると、`0.35-0.44` まで上げていても toxic fill は起きている。よって真に言えるのは、

- **低すぎる offset は危険**
- しかし **高ければ自動的に安全になるわけではない**

の二点である。

### 2.3 深夜帯 / 特定時間帯の危険性はある

tail analysis でも sell の overrepresented hour は `UTC 19, 13, 0, 14, 15`、buy の overrepresented hour は `UTC 14, 13, 12, 06, 19` だった。よって時間帯差の観察自体は妥当である。

ただしこれは `hour hard skip` を即入れる根拠ではなく、**hour を toxicity proxy として使う補助変数**と捉えるのが良い。

---

## 3. 取引レベルで見た「負け方の型」

### 3.1 Buy の負け方

#### 型A: `ranging` で spread がそこそこ広いのに入って下がる

例: `results/v460/fill_test/fill_records_20260323.jsonl:96`

- `buy`, `ranging`, `post_fill_30s_pnl=-1.5857bps`
- `spread_bps=2.3556`
- `effective_offset_used=0.3578`
- `sidecar_signal_status=fresh`
- `sidecar_bias=-0.7209`, `sidecar_confidence=0.1068`
- `decision_path=ev_offset`

これは「buy の根拠が強い」のではなく、**ranging なのに参加し、30秒後に普通に値を切り下げられている**例である。sidecar は弱く、本体の entry を止められていない。

#### 型B: `trending_up` 追随気味 buy が局所高値掴みになる

例: `results/v460/fill_test/fill_records_20260323.jsonl:92`

- `buy`, `trending_up`, `post_fill_30s_pnl=-3.0378bps`
- `spread_bps=3.2604`
- `queue_wait_sec=60.79`
- `queue_fill_prob_est=0.0113`
- `sidecar_signal_status=stale`
- `executor_offset_stages={"ev":1.1288,"velocity":2.0,...}`

これは短期上昇の勢いを見て攻めたが、**queue priority が低いまま待たされ、局所的な高値で約定して反落を食らった**形である。buy 側では「上がっているから買う」がそのまま利得になっていない。

### 3.2 Sell の負け方

#### 型C: `ranging` sell が AS 化する

例: `results/v460/fill_test/fill_records_20260323.jsonl:111`

- `sell`, `ranging`, `post_fill_30s_pnl=-2.9642bps`
- `adverse_selected=true`
- `effective_offset_used=0.4154`
- `cross_venue_lead_lag_applied=true`
- `cross_venue_lead_lag_cap_hit=true`
- `decision_path=ev_offset`

これは sell 側でかなり象徴的で、**offset を広げても、CV を載せても、結局 toxic flow に捕まる**。しかも `cap_hit=true` なので、追加の防御が価格まで届いていない可能性が高い。

#### 型D: `trending_up` sell がそのまま踏み上げられる

例: `results/v460/fill_test/fill_records_20260323.jsonl:137`

- `sell`, `trending_up`, `post_fill_30s_pnl=-4.6189bps`
- `spread_bps=3.6009`
- `queue_wait_sec=22.34`
- `adverse_selected=true`
- `effective_offset_used=0.4396`

これは売り位置が悪いというより、**上昇継続中に売って、そのまま 30 秒後も上に走られている**。sell は broad には改善していても、こうした continuation への逆張り参加はまだ痛い。

### 3.3 勝ち trade の型

参考として `447b2ec5` の良い約定を見ると、`results/v460/fill_test/fill_records_20260323.jsonl:2` や `:5` のように:

- `ranging` でも勝てている
- `adverse_selected=false`
- `decision_path=ev_offset`
- `CV/sidecar` はほぼ関与していない

つまり、このシステムは `ranging` 自体が絶対悪なのではない。**毒性の薄い ranging を選べれば勝てる**。本質は regime 名ではなく、entry quality の選別である。

---

## 4. restart 前の優先順位

### P0: まず post-589 の clean run を取る

これは絶対に先である。`590#` のままでは mixed-SHA で、かつ `03-24` 行も整合していない。再起動後は少なくとも以下を固定したい。

- `git_sha` 固定
- `run_id` 固定
- `config_hash` 監視
- `fill_records_YYYYMMDD.jsonl` 単位での連続観測

`000#` の運用方針に照らしても、ここをやらずにパラメータ議論を続けると再び attribution が壊れる。

### P1: `buy × ranging × ev_offset` を最優先で削る

current SHA の残課題はここである。具体的には次の条件が重なると危険度が高い。

- `buy`
- `gated_regime=ranging`
- `spread_bps` が 2.3bps 以上
- `queue_fill_prob_est` が低い、または `queue_wait_sec` が長い
- `sidecar fresh` でも confidence が低い
- `decision_path=ev_offset`

この群は broad な base_offset 変更より、**参加抑制 / offset 追加防御 / micro-timeout 的な待ち時間制御**の方が効く。

### P2: sell は全面救済ではなく toxic continuation だけ削る

current `c164d21d` では sell は既に黒字なので、sell 全体に guard を増やすと再び over-defensive になる。狙うべきは以下の局所だけである。

- `sell × adverse_selected=true`
- `sell × ranging`
- `sell × trending_up` で 30s 後も上昇継続する型
- `cross_venue_lead_lag_applied=true && cap_hit=true`

つまり売り全体ではなく、**「防御したつもりなのに毒を食っている sell」** に限定して削るべきである。

### P3: sidecar は強化ではなく fail-soft と attribution 優先

`000# §0.1` の通り SAC は Sidecar であり Driver ではない。今のログを見ても、その立場は変えない方がよい。

やるべきなのは:

- `fresh/stale/error` 別成績を same-SHA で監視
- low-confidence `fresh` を過大評価しない
- `fresh` が悪い時でも base execution が壊れない fail-soft を維持

であり、**bias 反転や倍率拡大をいきなりやる段階ではない。**

### P4: CV は blanket disable ではなく buy 条件を先に詰める

現時点での読みとしては、CV は「市場全体で逆効果」というより、

- buy での適用条件が悪い
- sell では適用後も cap に当たって no-op が出る

の2点が主問題である。ここを分けずに止めると、将来使える alpha も一緒に殺す。

---

## 5. 590# に対する是々非々の判定

### 支持

- `ranging` で負けているという主張
- 広め offset が broad には有利という観察
- sidecar / CV を盲信しない姿勢
- telemetry 欠損を問題視したこと

### 反証・補正

- `03-24` 行は現ファイル状態と整合しない
- mixed-SHA のため、`macro_strong_down` や旧 SHA 由来の悪化を current issue と混同しやすい
- sell は current SHA では黒字であり、全面崩壊とまでは言えない
- `sidecar error > fresh` から即「反転」へ行くのは飛躍
- `CV applied < not applied` から即「CV は無効」も飛躍
- telemetry 欠損は「古い SHA だけ」で片付けない方がよい

---

## 6. 結論

`590#` は「どこが苦しいか」を掴む資料としては有用だった。ただし、本質はもっと絞れる。

> **今の赤字の芯は、`ev_offset` 経路が adverse selection を通していることにある。**

しかも current SHA では sell 全体より `buy × ranging` の方が重い。したがって restart 前の重点は、

1. post-589 clean run の確保
2. `buy × ranging × ev_offset` の参加抑制
3. `sell` は toxic continuation のみ局所補修
4. sidecar / CV は全停止ではなく attribution を取りながら弱めに扱う

の順が良い。

新機構を足すより、まずこの4点を守った restart の方が、`000#` の profit-first 方針にも合致する。
