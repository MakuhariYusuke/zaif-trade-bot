# 403# 402レビュー: Fill Test 不調の多角的点検と Profit-First 補正

| 項目 | 値 |
|---|---|
| 文書番号 | 403# |
| 対象 | `docs/v460/402_time_guard_root_cause_and_397_review.md` |
| 観点 | fill test 収益性 / システム工学 / 市場理論 / 実装整合 / 再現性 |
| 判定 | 部分採用 |

---

## 1. 総評

`402#` は、時間帯を「悪い時間だから止める」から「なぜ悪いのか」に掘り直した点で前進です。
特に以下は確認できました。

- `UTC 00 / JST 09` が悪いこと
- `confidence >= 0.9` が全体で最悪帯であること
- `bff652e7df07` の再起動後 12h が明確に悪化していること

ただし、そこから直接

- `sell_hour_boost[0] 1.5 -> 2.5`
- `hard_skip_utc_hours[21]` の除外
- `sell ceiling を上げれば 397# guard が効く`

へ進むのは早いです。`402#` は観測として有益ですが、**修正対象の特定が一部ずれており、現行 fill test の不調を「時間帯レイヤー」へ寄せすぎています**。

---

## 2. 主要指摘

### 1. HIGH: `402#` の全体分析は mixed-SHA 集計であり、現行設定変更の直接根拠には弱い

`docs/v460/402_time_guard_root_cause_and_397_review.md:56` は `5,078 records / 1,249 fills` を全体分析の母集団に置いています。
しかし `results/v460/fill_test/fill_records_202603*.jsonl` をそのまま集計すると、現在手元の raw は **5,084 records / 1,250 fills** でした。

また、この期間は多数の `git_sha` が混在しています。つまり `402#` の第2章は、**現行 `bff652e7df07` の root cause というより、3月の broad forensic analysis** です。

この broad 分析自体は有用です。ただし、ここから直接

- `sell_hour_boost[0] 1.5 -> 2.5`
- `hard_skip UTC 21 の解除`
- `397# confidence guard の評価`

のような **現行設定の変更** に飛ぶと、因果が混ざります。

同じ問題は第3章にもあります。`docs/v460/402_time_guard_root_cause_and_397_review.md:119` は `fills=99` としていますが、raw の `bff652e7df07` は現時点で **100 fills / -46.2412bps** でした。差は小さいですが、再現条件が書かれていません。

**推奨対応**:

1. `402#` を「全体構造」と「current SHA 対応」に分ける
2. 3月通期の話は `broad pattern` と明記する
3. current SHA で触るパラメータは `bff652e7df07` 限定 deep dive から決める

### 2. HIGH: `AS率最大ドライバー` は正しいが、それは `adverse_selected_raw` の話であり live で直接使える量ではない

`docs/v460/402_time_guard_root_cause_and_397_review.md:58`-`docs/v460/402_time_guard_root_cause_and_397_review.md:70` の `AS%` は、raw 集計と一致しました。
ただし一致したのは `adverse_selected` ではなく **`adverse_selected_raw`** です。

根拠:

- `scripts/v460/lib/pnl_measurer.py:136`-`scripts/v460/lib/pnl_measurer.py:142`
  - `adverse_selected_raw` は「30秒後 mid が不利方向に動いたか」の **生判定**
  - `adverse_selected` は deadzone 適用後の運用指標
- `ztb/metrics/fill_quality.py:60`-`ztb/metrics/fill_quality.py:61`
  - dataclass 上も `raw` と `final` は別物として定義されています

ここで重要なのは、`adverse_selected_raw` は **post-fill / future-dependent label** だという点です。
診断には非常に有効ですが、live でそのまま使えるシグナルではありません。

したがって、`docs/v460/402_time_guard_root_cause_and_397_review.md:222`-`docs/v460/402_time_guard_root_cause_and_397_review.md:223` の

- `VPIN + OBI + vol_ratio -> AS probability`
- `AS-conditional offset`

という方向性自体は正しい一方、**raw AS をそのまま policy driver に昇格させるのは leakage の危険**があります。

正しい順序は次です。

1. `adverse_selected_raw` を offline teacher label として使う
2. pre-trade で観測可能な特徴量だけで proxy model を学習する
3. OOS で `Brier / PR-AUC / post_fill_30s_pnl uplift` を確認してから live に入れる

### 3. HIGH: `397#` の mid-confidence guard 不発は「sell ceiling 0.30」が原因とは言い切れない

`docs/v460/402_time_guard_root_cause_and_397_review.md:126`-`docs/v460/402_time_guard_root_cause_and_397_review.md:132` は、sell 側で confidence boost が効かない理由を「offset ceiling 0.30」と説明しています。

ここは実装上の整理が必要です。

実コードと設定を見ると:

- `configs/v460/fill_test.yaml:68`
  - `adaptation.max_offset_ratio = 0.30`
- `configs/v460/fill_test.yaml:473`
  - `sell_guard.offset_floor = 0.30`
- `configs/v460/fill_test.yaml:582`
  - `offset_ceiling_ratio_sell = 0.50`
- `scripts/v460/lib/maker_regime_boost.py:243`-`scripts/v460/lib/maker_regime_boost.py:272`
  - 397# の mid-confidence boost は regime boost 段で適用
- `scripts/v460/lib/maker_price.py:994`-`scripts/v460/lib/maker_price.py:1009`
  - side-specific ceiling は最後に別途適用

つまり、sell 側の真のボトルネックは単純な `offset_ceiling_ratio_sell` ではなく、

- 手前の `max_offset_ratio = 0.30`
- `sell_offset_floor = 0.30`
- パイプライン順序

の組み合わせである可能性が高いです。

実際、`bff652e7df07` の `sell + confidence[0.7,0.9)` 記録をみると、`offset_stages.final` は **21/21 件で 0.30** でした。一方で最終 `effective_offset_used` は `0.30` を超えるケースもあり、単純な「sell ceiling 0.30」説明では収まりません。

さらに重要なのは、**offset を多少上げても成績が改善していない** ことです。

`bff652e7df07` の filled だけを見ると:

- `sell mid-confidence`: `n=16`, `mean_pnl=-1.3265`, `mean_offset=0.3181`
- `sell other confidence`: `n=34`, `mean_pnl=+0.0576`, `mean_offset=0.3242`

offset 平均はほぼ同じなのに、mid-confidence sell だけが悪い。したがって、`sell ceiling を上げれば解決` という読みは弱いです。

**推奨対応**:

1. `sell ceiling` ではなく `global 0.30 cap + floor + pipeline order` を監査する
2. 同時に `confidence` 自体の calibration 問題として扱う
3. 397# の対策を続けるなら、offset 増加より `confidence band veto / side participation quality` を優先する

### 4. HIGH: `hard_skip_utc_hours[21]` を外す提言は、現データでは選択バイアスを含む

`docs/v460/402_time_guard_root_cause_and_397_review.md:215`-`docs/v460/402_time_guard_root_cause_and_397_review.md:216` は、`UTC 21` を hard skip から外す候補にしています。

ここは現行データでは判定できません。

確認結果:

- `UTC 16`: `records=10`, `fills=0`, すべて `hard_skip_utc_hour`
- `UTC 21`: `records=9`, `fills=0`, すべて `hard_skip_utc_hour`

つまり current dataset では、その時間は **そもそも取引していません**。取引していない時間について「AS が低いから安全」と言うのは、現行ポリシーのもとでは observational に評価できません。

`402#` の broad history では `UTC 21` が比較的ましに見えても、それは別 SHA / 別 guard 構成の観測です。`205#` の hard skip を現行で外す根拠には足りません。

**推奨対応**:

- `hard skip` の撤去はしない
- やるなら `shadow mode` または `soft gate` で 1 段階だけ緩める
- その時も `sell only`, `spread`, `VPIN`, `confidence` を guardrail にする

### 5. HIGH: 今回の fill test 不調は `JST 09h sell` だけではなく、両サイド劣化と過剰 suppression の複合問題

`402#` は `JST 09h sell tail` を強く押していますが、`bff652e7df07` の current SHA はもっと広く悪いです。

確認結果:

- raw `bff652e7df07`: `351 records / 100 fills / -46.2412bps`
- `buy`: `n=50`, `mean=-0.5395`
- `sell`: `n=50`, `mean=-0.3854`

つまり **sell だけが悪い状態ではありません**。

加えて cancel reason を見ると、すでに suppression はかなり強いです。

- `sell_dynamic_kill`: 130
- `skip_gate`: 47
- `spread_too_narrow`: 32
- `stale_adverse_drift`: 18

この状態で `docs/v460/402_time_guard_root_cause_and_397_review.md:215` のように `sell_hour_boost[0] 1.5 -> 2.5` を入れると、改善よりも **さらなる liveness 低下** になりやすいです。

市場理論的にも、毒性が高い時間帯で offset を広げ過ぎると「より不利な時だけ刺さる」構図になりやすいです。今の bff は既にその気配があります。

**推奨対応**:

1. `JST 09h sell` は監視継続する
2. ただし P0 を `boost 1.5 -> 2.5` には置かない
3. current SHA では `buy` 側も含めて `confidence`, `skip_gate`, `stale_adverse_drift`, `spread_too_narrow` の交絡を見る

### 6. MEDIUM: `confidence >= 0.9` は本当に危険帯だが、ここは時間帯より calibration 問題として扱うべき

`docs/v460/402_time_guard_root_cause_and_397_review.md:97`-`docs/v460/402_time_guard_root_cause_and_397_review.md:99` の
`confidence >= 0.9` が最悪という観測は再現できました。

raw では:

- `confidence [0.7,0.9)`: `n=440`, `mean=+0.1231`
- `confidence >= 0.9`: `n=52`, `mean=-1.6897`

ここから見えるのは、「confidence は単調に信頼度を表していない」ということです。
時間帯問題として読むより、**Bayesian regime filter の過信 / transition lag / calibration 崩れ** として読む方が筋が良いです。

したがって提言も、追加の時間帯レイヤーではなく以下が優先です。

- `confidence bucket` ごとの reliability / calibration 点検
- `confidence >= 0.9` 時の `regime`, `spread`, `microprice_bias_bps`, `VPIN` 分解
- 必要なら `mid_confidence boost` ではなく `high_confidence caution` を既存 boost 系へ統合

### 7. MEDIUM: `JST 09h = 機関投資家の大口注文` は plausible だが、文書では inference と明記した方がよい

`docs/v460/402_time_guard_root_cause_and_397_review.md:85`-`docs/v460/402_time_guard_root_cause_and_397_review.md:87` は、JST 09h の悪化を「機関投資家の大口注文」と説明しています。

市場理論としては十分あり得ます。ただし、リポジトリ内のログは参加者属性を持っていません。従って、これは **説明仮説** であって **実証済み事実** ではありません。

ここは表現を一段弱めて、

- opening-session informed flow
- queue imbalance / spread widening cluster
- macro/opening auction style liquidity shock

程度の記述に留めるのが堅いです。

---

## 3. 検証して残すべき論点

`402#` の中で、次はそのまま採用してよいです。

1. `adverse_selected_raw` ベースでは、時間帯差はかなり大きい
2. `JST 09h / UTC 00` は broad history でも current SHA でも危険
3. `confidence >= 0.9` は全体で危険帯
4. 時間帯レイヤーが増え過ぎており、整理対象である

つまり `402#` は「時間帯防御を増やす根拠」ではなく、**時間帯を proxy にしてきた設計を AS proxy / confidence calibration ベースに置き換えるべきだと示す文書** として読むのが正しいです。

---

## 4. Profit-First の次アクション

### P0

1. `402#` の全体分析と current-SHA 分析を分離する
2. `sell_hour_boost[0] 1.5 -> 2.5` は保留する
3. `hard_skip_utc_hours[21]` は外さず、必要なら shadow mode で再評価する
4. `confidence >= 0.9` と `mid-confidence` の両方を current SHA で再分解する

### P1

1. `adverse_selected_raw` を teacher label とする leak-free AS proxy を作る
2. 既存 7 レイヤーを増やさず、`skip_gate / risk guard / regime boost` の既存経路へ統合する
3. `global 0.30 cap`, `sell_floor 0.30`, `offset_ceiling_ratio_sell 0.50` の三者関係を整理する

### P2

1. 時間帯ではなく `state-conditioned participation score` に寄せる
2. `buy` の不調も主対象に入れる
3. 追加の時間帯レイヤーは原則禁止し、既存レイヤー統合を優先する

---

## 5. 結論

`402#` の一番価値ある発見は、**「時間帯が悪い」のではなく、AS raw と confidence calibration が悪い** という点です。

ただし現時点では、そこから

- `sell boost をさらに強くする`
- `hard skip を外す`
- `sell ceiling を上げる`

へ進むのは筋が悪いです。

今の fill test 不調は、**JST 09h sell tail だけではなく、両サイド劣化・過剰 suppression・confidence miscalibration・time-layer 過積載** の複合問題です。

profit-first で進めるなら、次にやるべきは 8 個目の時間帯ガードではありません。
**AS raw を teacher にした pre-trade proxy 化と、既存レイヤーの統合・再較正** です。
