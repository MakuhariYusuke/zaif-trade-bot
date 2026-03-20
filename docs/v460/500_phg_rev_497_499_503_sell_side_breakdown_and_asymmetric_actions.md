# 500# 497#-499# / 503# レビュー: Buy 改善後の Sell 崩壊をどう扱うか

> 種別: review
> 対象: 497# `docs/v460/497_config_change_impact_deep_dive.md`, 498# `docs/v460/498_hot_reload_lateral_expansion.md`, 499# `docs/v460/499_hard_loss_cap_crash_loop_fix.md`, 503# `docs/v460/503_sell_buy_pnl_factor_analysis.md`
> 日付: 2026-03-20

---

## 0. 総評

結論から言うと、今回の整理で一番大事なのは次の 3 点である。

1. **503# の問題意識はかなり良い**
   - 損失の大半が `ranging` に集中していること
   - `buy` と `sell` で逆選択の形が違うこと
   - `sell_dynamic_kill` が機会損失を大きくしていること
   は、実データでも概ね支持できる。

2. **ただし、503# の処方箋の一部は危ない**
   特に「`cross_venue` guard を sell 側にもそのまま常時適用する」は、現行ロジックの意味論とも、実データとも整合しない。

3. **sell 側は恒久的に壊れているわけではない**
   7 日集計では sell が大きく負けているが、`2026-03-20` の same-run / same-SHA では `sell +25.16` まで戻っている。
   つまり本丸は「sell が構造的に不可能」ではなく、**特定条件下で sell が slow fill と wide offset に引っ張られて崩れる**ことにある。

profit-first に整理すると、次にやるべきは

- `sell` を `buy` と対称に扱うことではなく、
- **sell 専用の stale / offset / kill 取り扱いを詰めること**

である。

---

## 1. 主要所見

| # | 重要度 | 所見 | 判定 |
|---|---|---|---|
| 1 | CRITICAL | `503#` の「cross_venue を sell にも常時適用」は根拠が弱く、現行意味論ともズレる | 強く補正 |
| 2 | HIGH | `sell` 崩壊は 7 日全期間で均一に起きているのではなく、主に `3/19 ranging + slow fill` に集中 | 補正必須 |
| 3 | HIGH | `buy` と `sell` は負け方が対称ではない。execution policy を分けるべき | 強く支持 |
| 4 | HIGH | `recovery_skew` は liveness 回復には効いても、現時点では収益改善は未証明 | 要監視 |
| 5 | MEDIUM | `sell_dynamic_kill` は重いが、単純緩和は危険。先に deadlock と slow fill を切るべき | 是々非々 |
| 6 | MEDIUM | `498#` の hot-reload 横展開は運用改善だが、収益評価の因果を汚しやすい | 注意 |
| 7 | LOW | `499#` の hard_loss_cap 修正は正しく必要だが、sell alpha 問題とは別物 | 支持 |

---

## 2. まず強く補正すべき点

### 2.1 `cross_venue` を sell 側へそのまま広げる案は危ない

503# の最大の補正点はここである。

`scripts/v460/lib/cross_venue_lead_lag.py:287` では、

- `direction="up" -> adverse_side="sell"`
- `direction="down" -> adverse_side="buy"`

と定義されている。

そして `scripts/v460/lib/maker_risk_guards.py:236`-`243` では、

- `hint.adverse_side != side` の場合は **no-op**

である。

これは単なる実装漏れではなく、**「reference venue がどちらの side に adverse か」だけを守る** という設計意図である。

実データでも、503# の処方箋はそのまま支持しにくい。
`2026-03-14` から `2026-03-20` の fill では、sell 側の `cross_venue_lead_lag_applied` は

- `True`: `22 fills`, `PnL=-39.01`, `avg=-1.773`
- `False`: `255 fills`, `PnL=-54.44`, `avg=-0.213`
- `None`: `101 fills`, `PnL=-88.73`, `avg=-0.879`

であり、**sell で適用された少数ケースはむしろかなり悪い**。

したがって、今言えるのは次までである。

- `cross_venue` が buy 側の保護として有効そうなのは事実
- しかしそれを根拠に「sell にも同じ guard を常時適用すべき」とは言えない
- やるなら **sell 専用の別ロジック** として設計すべき

つまり、503# の P0 提案はそのまま採用しない方がよい。

### 2.2 sell 崩壊は「常時」ではなく「条件付き」で強い

503# の 7 日集計は有益だが、そのまま current 状態の断定には使い過ぎない方がよい。

同じ `2026-03-14` から `2026-03-20` の fill では、

- `buy`: `385 fills`, `PnL=-110.10`, `avg=-0.286`, `WR=46.8%`
- `sell`: `378 fills`, `PnL=-182.18`, `avg=-0.482`, `WR=50.0%`

で、確かに sell の方が悪い。

ただし日別に切ると様子が違う。

- `20260316`: `sell +55.43`
- `20260317`: `sell -83.40`
- `20260318`: `sell +49.52`
- `20260319`: `sell -143.53`
- `20260320`: `sell +25.16`

特に `20260320` は `run_id=1773953473_cd76a319`, `git_sha=dfbe3b539eaa`, `config_hash=09d62816a95e3089` の単一 run で、sell がプラスである。

したがって current の整理は、

- `sell は壊れている`

よりも、

- **sell は特定の ranging / stale 条件で大きく崩れるが、fresh な run では回復し得る**

の方が正確である。

### 2.3 `recovery_skew` は「死なない」には効いても「儲かる」はまだ言えない

496# の `recovery_skew` は発想として理解できる。
`route_to_kill_deadlock` で常に休むより、wide offset で inventory repair を試みる方が maker 的には自然だからである。

しかし `20260320` の same-run だけを見ると、filled の `resolved_side_reason` は

- `recovery_skew|sell`: `19 fills`, `PnL=-11.99`, `avg=-0.631`
- `balance_switch|sell`: `23 fills`, `PnL=+25.23`, `avg=+1.097`
- `None|sell`: `21 fills`, `PnL=+11.91`, `avg=+0.567`

であり、**現時点では recovery_skew は収益面で弱い**。

つまり判定は次である。

- liveness 改善としては意味がある
- しかし収益改善策としては未証明
- `offset_mult=2.0` はかなり荒く、現状では profit-first の本命ではない

---

## 3. 503# の中で特に支持できる論点

### 3.1 `buy` と `sell` は対称でない

ここは 503# の最も良い点である。

7 日集計では、

- `buy` の fast fill `<10s` は `-64.56`
- `buy` の slow fill `>=30s` は `+11.02`
- `sell` の fast fill `<10s` は `+58.56`
- `sell` の slow fill `>=30s` は `-164.55`

だった。

この非対称性は重要で、売買を同じ execution policy で処理してはいけないことを示している。

市場理論的には、

- `buy`: 即約定が toxic になりやすい
- `sell`: 長時間滞留が toxic になりやすい

という別々の adverse selection パターンである。

したがって、次の方針はかなり筋が良い。

- `buy` は「少し深め、すぐ食われない」方へ寄せる
- `sell` は「取り残されない、古くならない」方へ寄せる

### 3.2 損失の大半が `ranging` に集中している

これも概ね正しい。

7 日集計では、

- `buy ranging`: `335 fills`, `PnL=-83.98`
- `sell ranging`: `329 fills`, `PnL=-196.55`

であり、両サイド損失の大半が `ranging` に乗っている。

特に sell の worst day である `20260319` では、

- `sell ranging`: `41 fills`, `PnL=-128.13`, `avg=-3.125`
- `sell >=30s`: `20 fills`, `PnL=-144.62`, `avg=-7.231`

で、`ranging + slow sell` が本丸だった。

一方で `20260320` は、

- `sell ranging`: `57 fills`, `PnL=+14.81`, `avg=+0.260`
- `sell >=30s`: `20 fills`, `PnL=-19.88`, `avg=-0.994`

まで改善している。

この差を見ると、本当に潰すべきは

- `ranging` そのもの

ではなく、

- **`ranging` の中で stale 化した sell**

である。

### 3.3 sell は wide offset より narrow-to-mid の方が良い

ここもかなり重要である。

7 日集計の sell 側 offset bucket は、

- `0.10-0.19`: `43 fills`, `+138.58`, `avg=+3.223`
- `0.19-0.25`: `129 fills`, `-207.43`, `avg=-1.608`
- `>=0.25`: `206 fills`, `-113.33`, `avg=-0.550`

だった。

`20260319` と `20260320` を切っても、

- `20260319 sell 0.10-0.19`: `+36.95`
- `20260319 sell 0.19-0.25`: `-180.48`
- `20260320 sell 0.10-0.19`: `+55.62`
- `20260320 sell 0.19-0.25`: `-30.46`

である。

このため、sell 側に対しては

- 「危険だからもっと wide にする」

一辺倒は合っていない。
むしろ、**狭めだが鮮度の高い quote** の方が良い可能性が高い。

---

## 4. 497#-499# の評価

### 4.1 497# の観測は有益だが、原因は単独ではない

497# のうち、次は支持できる。

- `454# micro-timeout ON` がプラスだった
- `438# ranging hard skip` は明確にマイナスだった
- `3/19` の regression は slow fill / sdk / deadlock が絡んだ

一方で、

- `458# slope_threshold=0.5` が sell 悪化の主因

という読みは、まだ強すぎる。
現行 config は今も `configs/v460/fill_test.yaml:194` で `slope_threshold: 0.5` だが、`20260320` sell はプラスである。

よって正しい整理は、

- macro sell 感度は **寄与因子のひとつ** ではありうる
- しかし **3/19 sell 崩壊の単独主因とは言えない**

である。

### 4.2 498# は運用改善としては良いが、収益評価を難しくする

`498#` の hot-reload 横展開は、運用面では前進である。
ただし experiment discipline という観点では副作用がある。

変更可能フィールドが増えるほど、

- 同じ日内に意味論が変わる
- `run_id` と `config_hash` の対応が追いづらくなる
- 因果推定が更に難しくなる

ためである。

今のフェーズでは、`498#` は

- **ops 改善としては支持**
- **profit study の主経路としては restart ベース優先**

くらいの位置づけが妥当である。

### 4.3 499# は正しい修正であり、入れてよい

`499#` は必要な liveness fix である。
コード上も、

- `scripts/v460/lib/orchestrator_lifecycle.py:360`-`367`
- `scripts/v460/lib/orchestrator_pre_cycle.py:104`
- `scripts/v460/lib/fill_loop_orchestrator.py:373`

で、`cumulative_pnl_jpy` を UTC 当日スコープ + 日替わりリセットにしている。

これは正しい。

ただし、これは

- crash loop を止める
- hard_loss_cap の誤発火を止める

ための修正であって、sell alpha 改善そのものではない。
ここを混同しない方がよい。

---

## 5. 今の本丸は何か

profit-first に一番効く整理は、次である。

### 5.1 sell は「広げる」より「古くしない」

今の sell は、

- `sell_dynamic_kill=312`
- `final_clamp_hard_skip=74`
- `route_to_kill_deadlock=152`

といった抑制に苦しんでいる。

しかも `20260320` の sell cancel 上位は

- `sell_dynamic_kill=33`
- `skip_gate=26`
- `final_clamp_hard_skip=24`

だった。

この状態で更に sell offset を広げると、

- stale 化
- final clamp / hard skip
- slow fill

の悪い方向へ行きやすい。

したがって sell の最優先は、

- **広げることではなく、累積待機時間を短くすること**
- **狙う offset 帯を 0.10-0.19 付近へ寄せること**

である。

### 5.2 sell_dynamic_kill は「悪」ではなく「副作用つきの防御」

503# は `sell_dynamic_kill` の機会損失を正しく見ている。
しかし、そこからすぐ

- `sell threshold -0.5 -> -0.8`

へ行くのは危険である。

理由は 2 つある。

1. 現行 runtime では regime threshold と inventory relaxation が入っているため、単純な base 値比較ではない
2. `20260320` の same-run では、`sell_dynamic_kill=33` を抱えながら sell はプラスだった

つまり、sell kill は重いが、**全部外せば良くなるわけではない**。

先にやるべきは、

- kill そのものの緩和

よりも、

- kill が `route_to_kill_deadlock` と結びついて participation を潰す経路の整理
- kill 前に stale sell を減らすこと

である。

### 5.3 `recovery_skew` は縮めて試すべき

現在の `configs/v460/fill_test.yaml:736` は `recovery_skew_offset_mult: 2.0` である。
しかし same-run 実績では `recovery_skew|sell` が負けている。

したがって、もし継続するなら次のように扱う方がよい。

- 本採用ではなく **評価継続中** とみなす
- `2.0` をいきなり使わず、`1.2-1.5` の狭い範囲で試す
- `recovery_skew` を `ranging` の時だけ止める、または severe inventory の時だけ許可する

今のまま「deadlock 解消 = 収益改善」とはまだ言えない。

---

## 6. 推奨アクション

### P0

1. **sell の cumulative order age cap を入れる**
   - `wait_sec` ではなく、requote を含めた総待機時間で `20-25s` を上限にする
   - sell は `>=30s` が明確に悪いので、ここを物理的に残さない

2. **sell offset を side-specific に狭める**
   - `buy` は wide がまだ許される
   - `sell` は `0.10-0.19` が良く、`0.19+` が悪い
   - 対称 ceiling / 対称 boost をやめる

3. **`cross_venue` は sell へ横流ししない**
   - 現行 adverse-side semantics を維持
   - sell でやるなら新規の sell-specific toxicity / stale 防御として別設計にする

### P1

4. **`recovery_skew` を shadow に近い扱いで再評価する**
   - `resolved_side_reason="recovery_skew"` を KPI として独立監視
   - 現状マイナスなので multiplier 縮小 or 条件限定を検討

5. **`sell_dynamic_kill` は base threshold ではなく副作用経路を調整する**
   - cooldown / deadlock / balance_switch 連動を点検
   - blunt な threshold 緩和は後回し

6. **`498#` 以降の実験は restart 前提で区切る**
   - hot-reload は運用用
   - 収益評価は same-run / same-config / same-SHA で切る

### P2

7. **`ranging_low_vol_skip=718` の soft 化を検討する**
   - これは sell 専用ではないが、参加率回復の大きなレバーである
   - ただし sell stale 問題と分離して測ること

---

## 7. 結論

今回の対象で一番良かったのは、`503#` が

- `buy` と `sell` の負け方の違い
- `ranging` 集中
- `sell_dynamic_kill` の重さ

をかなり正確に捉えていたことである。

一方で、一番危ないのは

- **`cross_venue` を sell にもそのまま適用する**
- **sell_dynamic_kill を buy と同じに緩める**

という「対称化」である。

今のデータが示しているのは対称化ではなく、**非対称化の必要性**である。

- `buy` は fast fill を避けたい
- `sell` は slow fill を避けたい
- `buy` は wide offset がまだ機能する
- `sell` は narrow-to-mid offset の方が良い

したがって、次の一手は

1. sell stale age を切る
2. sell offset 帯を狭める
3. recovery_skew を再評価する
4. cross_venue sell 常時化はやらない

の順が良い。

要するに、**buy が少し良くなった今、sell を同じやり方で直そうとしないこと**が最重要である。
