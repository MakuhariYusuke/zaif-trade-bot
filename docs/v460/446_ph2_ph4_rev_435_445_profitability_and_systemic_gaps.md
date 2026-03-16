# 446# 435#-445# 横断レビュー: 進展評価・実走上の盲点・既存システム起因の残課題

> **種別**: rev
> **対象**: 435#-445#
> **関連**: 432# / 434# / 437# / 440# / 441# / 445#
> **レビュー日**: 2026-03-16

---

## §0 Executive Summary

435#-445# の流れは、全体としては **正しい方向への収束** である。

- `440#` の **regime × side 非対称化** は妥当
- `445#` の **EMA + confidence** は 120s cycle 前提では 442# より自然
- `441#` の **A/B 比較基盤** は今後必須

ただし、現時点で strongest claim を置けるのは「方向性が良い」までであり、**「もう効いた」まではまだ言えない**。

今回の主結論は次の6点。

| # | 重要度 | 結論 |
|---|---|---|
| F1 | HIGH | `3/16` の改善値は **mixed-SHA / mixed-runtime** で汚れており、438#-445# の純粋効果としてはまだ読めない |
| F2 | HIGH | `cross_venue` は live presence が出始めたが、**発火は薄く、0 veto、適用時も no-op が多い** |
| F3 | HIGH | 445# confidence mode は前進だが、**EMA 方向と point spread を同居させているため、veto/観測の意味がずれる** |
| F4 | MEDIUM | 438# の `ranging_buy_low_vol_skip` は効いている可能性がある一方、**参加率を落としすぎるリスク** が高い |
| F5 | MEDIUM | 440# の「toxicity ML 棄却」は current design には妥当だが、**既存 heuristic toxicity 系の再校正を飛ばして cross-venue へ寄り過ぎ** |
| F6 | HIGH | `route_to_kill_deadlock` / `sell_dynamic_kill` / `buy_dynamic_kill` が依然として主要 cancel reason で、**既存システムの liveness 問題が残っている** |

profit-first に言えば、次の優先順位は

1. **same-SHA で効果測定できる状態を作る**
2. **cross_venue の「効いたふり」を潰す**
3. **既存 toxicity / deadlock 系の改善余地を先に回収する**

である。

---

## §1 裏取りで確認できた事実

### §1.1 440# regime-side offset 非対称化は実装されている

以下はコードと YAML の両方で確認できた。

- `configs/v460/fill_test.yaml:154` `ranging_offset_discount_buy: 1.15`
- `configs/v460/fill_test.yaml:155` `ranging_offset_discount_sell: 0.85`
- `configs/v460/fill_test.yaml:404` `unknown_buy_offset_boost: 2.0`
- `configs/v460/fill_test.yaml:406` `unknown_sell_offset_boost: 1.3`
- `scripts/v460/lib/maker_regime_boost.py:159`
- `scripts/v460/lib/maker_regime_boost.py:255`

この変更自体は後方互換を維持しており、設計として無理がない。

### §1.2 441# A/B 比較基盤は有用だが、split-date だけで比較している

`ab_offset_comparison.py` は現状、`split_date` で日付分割するだけであり、`git_sha` / `run_id` を使っていない。

- `scripts/v460/analysis/ab_offset_comparison.py:318`
- `scripts/v460/analysis/ab_offset_comparison.py:327`

したがって、同日中に複数 SHA が混在すると、比較結果はそのままでは因果に使えない。

### §1.3 3/16 の After データは mixed-SHA

`results/v460/fill_test/fill_records_20260316.jsonl` を見ると、`After` 157 records の中に **6個の git SHA** が混在していた。

- `f34467b5c2f8`: 79 rows, 6 fills, avg `+7.742bps`, cross_venue fields `0`
- `a9714ad9af85`: 39 rows, 8 fills, avg `+1.223bps`, cross_venue fields `5`
- `e23a063923ee`: 23 rows, 7 fills, avg `+1.340bps`, cross_venue fields `3`
- その他 3 SHA も混在

つまり、`3/16` の総合 `+1.391bps` は、**cross_venue 未搭載断片の寄与を含む**。

### §1.4 cross_venue の live presence はまだ薄い

`results/v460/fill_test/fill_records_20260316.jsonl` では:

- rows: `157`
- filled: `26`
- cross_venue fields non-null: `9`
- `cross_venue_lead_lag_applied = true`: `4`
- `cross_venue_lead_lag_veto`: `0`

これは「配線された」ことは示すが、「主要ドライバーになった」ことは示さない。

### §1.5 既存 toxicity 系は今も live path に存在する

435#-445# では `ML toxicity veto` の失敗が強調されているが、heuristic toxicity 系は既に live path へ入っている。

- `scripts/v460/lib/orchestrator_mid_cycle.py:127`
- `scripts/v460/lib/cycle_gate_aggregator.py:419`

また実ログでは、guard restore に以下が見える。

- `toxic_veto_set: 367`
- `toxic_veto_block: 53`

つまり、「toxicity を捨てた」のではなく、**ML 版だけが失敗した** と読むべきである。

---

## §2 Findings

### F1. `3/16` の改善値は mixed-SHA のため、438#-445# の純粋効果として扱えない

`441#` の A/B 比較をそのまま走らせると、確かに target bucket は良く見える。

- `ranging:buy`: `-0.422 → +4.299bps`, `p=0.0319`
- overall: `-0.274 → +1.391bps`

しかし、この `After` 側 157 records は日付単位で切っただけで、`git_sha` が混在している。しかも最も成績が良い `f34467b5c2f8` は **cross_venue fields が 0 件** だった。

したがって、現時点で言えるのは

> `3/16 は改善して見えるが、どの変更が効いたかは未分離`

までである。

`441#` のツール自体は良いが、**same-SHA / same-run_id fence を持たない限り、強い結論の根拠には使えない**。

### F2. cross_venue は live で動き始めたが、まだ「効いている」証拠は弱い

`445#` 後の live 痕跡として、`cross_venue` はゼロではなくなった。これは前進である。

ただし、現時点では次の限界がある。

1. `9/157` rows にしか field が出ていない
2. `applied=true` は `4` 件のみ
3. `veto=0`
4. `fill_test.log` で確認できた buy-adverse 3件はすべて `offset 0.3000->0.3000` の **no-op**

つまり、`cross_venue` は

- 観測としては出始めた
- しかし実行効果としてはまだ薄い

という段階である。

特に `maker_risk_guards.py` 側で `max_ratio=self._effective_max_ratio(side)` による cap を受けるため、**既に offset が上限に張り付いている局面では boost を掛けても変わらない**。

- `scripts/v460/lib/maker_risk_guards.py:255`
- `scripts/v460/lib/maker_risk_guards.py:277`

これは「理論上は良いが live では no-op」という典型的な盲点である。

### F3. 445# confidence mode は direction と magnitude の意味がずれている

445# で一番気になるのはここである。

`compute_cross_venue_lead_lag_hint()` は confidence mode では

- **direction** を `ema_spread_bps` の符号で決める
- しかし返す `hint.spread_bps` は **point spread** のまま持つ

- `scripts/v460/lib/cross_venue_lead_lag.py:234`
- `scripts/v460/lib/cross_venue_lead_lag.py:239`
- `scripts/v460/lib/cross_venue_lead_lag.py:316`

さらに veto 判定は `abs(hint.spread_bps)`、つまり point spread を使う。

- `scripts/v460/lib/maker_risk_guards.py:238`

このため、実ログで実際に

- `direction=down`
- `adverse_side=buy`
- `spread=+0.15bps`
- `ema_spread=-2.26bps`

という **意味のねじれた hint** が出ている。

これは即バグ断定まではしないが、少なくとも

- 観測指標として分かりづらい
- veto と boost の意味が一致しない
- 後分析で誤読しやすい

という問題を持つ。

設計としては、`point_spread_bps` と `ema_spread_bps` を明示的に分け、

- direction / adverse_side / confidence: EMA 系
- veto: EMA か point のどちらを使うかを明記

とした方がよい。

### F4. 438# の buy+ranging 防御は promising だが、かなり強い suppression になっている

`438#` の hard skip 復帰は、実走ログでも強く出ている。

- `ranging_low_vol_skip`: `57` 件 / `157` rows (`3/16`)
- `fill_test.log` でも同 reason が大量発火

A/B 比較上も `ranging:buy` の fill rate は

- `31.4% → 10.5%`

まで低下している。

これは「毒を避けている」と読むこともできるが、同時に

- side bias
- 在庫補正負荷
- trade starvation
- 数件の勝ちで平均が跳ねる不安定性

を生みやすい。

現状は **改善の芽はあるが、休みすぎの危険も高い** という評価が妥当である。

### F5. 440# の「toxicity ML 棄却」は current design には妥当だが、toxicity 路線全体を捨てる理由にはならない

440# の実験結果自体は納得できる。

- ROC-AUC ≈ `0.50`
- skip 改善 ≈ `0bps`
- current feature / label design では per-trade AS prediction が立たない

ただし、そこから

> toxicity veto は無理なので cross_venue へ進む

と一直線に寄るのはやや危うい。

理由は2つある。

1. 既存の heuristic toxicity budget / veto が live path でまだ十分に再評価されていない
2. current ML failure は「toxicity という概念の失敗」ではなく、「今のラベル・特徴量・切り方の失敗」かもしれない

特に 440# 自身も認めている通り、個別取引レベルは難しくても **regime × side レベル** では差が出ている。ならば、heuristic toxicity と regime-side asymmetry を再統合する余地はまだ大きい。

### F6. 元々ある liveness 問題が依然として重い

3/16 の `cancel_reason` 上位を見ると、次がまだ大きい。

- `ranging_low_vol_skip`: `57`
- `route_to_kill_deadlock`: `24`
- `sell_dynamic_kill`: `15`
- `skip_gate`: `10`

特に `route_to_kill_deadlock` と `sell_dynamic_kill` は、cross_venue 以前の既存系ブロッカーである。

また `fill_test.log` でも

- `sell_dynamic_kill` の連続 block
- `buy_dynamic_kill` の time limit release

が残っている。

これはかなり率直に言えば、**今の live では alpha 改善と liveness 修正が同時に競合している** 状態であり、後者が片付いていない限り前者の真価は見えにくい。

---

## §3 評価できる点

厳しめの指摘が多いが、前進もはっきりある。

### §3.1 440# の実装姿勢は良い

- backward compatible
- config SSOT 維持
- code change が局所的
- A/B しやすい

この意味で、`regime-side asymmetry` は 435#-445# 群の中で最も「堅くて儲けに近い」実装である。

### §3.2 445# の EMA 発想は正しい

`sign_disagree` 問題に対して、120s cycle で raw velocity を hard gate にするのをやめ、EMA + confidence へ寄せたのは理にかなっている。

ここは **方向修正として正しい**。

### §3.3 比較・可観測性への意識は前進している

- `ab_offset_comparison.py`
- `cross_venue` FillRecord fields
- `cross_venue_hint` event log

いずれも「効いたかどうかを見よう」という方向で、今後の検証速度を上げる資産になる。

---

## §4 追加で拾うべき改善余地

### P0: 効果測定の clean room 化

1. `ab_offset_comparison.py` に `--git-sha` と `--run-id` filter を追加する
2. 比較結果 JSON に `git_sha_set` / `run_id_set` / `date_range` を埋める
3. `3/16` のような mixed-SHA 日は、A/B の headline から除外する

### P0: cross_venue の no-op 可視化

1. FillRecord へ `cross_venue_pre_offset` / `cross_venue_post_offset` / `cross_venue_effective_delta` を出す
2. `mult=1.00` ではなく、`cap_hit=true/false` を明示する
3. veto 判定に使った量が `point_spread` なのか `ema_spread` なのかを出す

### P1: toxicity 路線の再監査

1. ML toxicity は一旦凍結でよい
2. その代わり、既存 `toxicity budget / veto` の発火帯と PnL を再集計する
3. `regime-side asymmetry` と `toxicity level` を掛け合わせて再キャリブレーションする

### P1: suppression の効き過ぎ対策

1. `ranging_buy_low_vol_skip` に participation floor を設ける
2. `buy+ranging` を全閉鎖せず、microprice / imbalance / spread 条件を満たす一部だけ通す controlled lane を作る
3. fill rate 低下だけでなく `inventory imbalance` と `one-sidedness` も毎日監視する

### P1: 既存 liveness 問題の削減

1. `route_to_kill_deadlock` の再発条件を切り分ける
2. `sell_dynamic_kill` の ranging 帯発火を再監査する
3. cross_venue の前に、kill / deadlock 系の second-order impact を減らす

### P2: retrain は attribution が落ち着いてから

`441#` の scheduler 準備は悪くないが、今のように same-day mixed-SHA で attribution が濁る環境では、オンライン retrain を live へ重ねると原因追跡がさらに難しくなる。

現時点では

- 先に execution 側の clean attribution
- その後に retrain

の順が安全である。

---

## §5 最終結論

435#-445# でやっていることは、全体としては「正しい方向に寄っている」。

ただし、現時点で最も危ないのは

> **良い改善を入れた後に、mixed-SHA と suppression の副作用で“効いたように見えているだけ”の状態を本改善と誤認すること**

である。

今回のレビューとしては、以下の順で進めるのが最も堅い。

1. `440` 系の regime-side asymmetry を same-SHA で再評価する
2. `445` 系 cross_venue の意味論と no-op を潰す
3. 既存 toxicity heuristic と deadlock 系を先に詰める
4. その後に retrain を live へ強く乗せる

一言で言えば、**今は「新しい賢さ」を足す段階というより、「効いている改善だけを clean に残す段階」** である。
