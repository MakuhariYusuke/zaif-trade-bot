# 393# 391レビュー: Fill Test Root Cause 分析の妥当性監査

| 項目 | 値 |
|---|---|
| 文書番号 | 393# |
| 対象 | `docs/v460/391_fill_test_deep_dive_root_cause.md` |
| 観点 | profit-first / システム工学 / 市場理論 / 再現性 |
| 判定 | 部分採用 |

---

## 1. 総評

`391#` は、利益を削っている箇所を具体的に掘ろうとしている点で方向は良いです。
特に、`sell` 側 tail risk、wide spread、rapid-fire 約定の悪化、`ranging` 優勢局面の薄利という観測自体は、maker の逆選択という市場理論とも整合します。

ただし、そのまま実装判断に使うには危険な点が 4 つあります。

1. **4日窓が mixed-SHA で、現行実装の原因分析としては汚れている**
2. **headline 数値が raw JSONL から再現できず、集計条件が不明**
3. **`EV>0 = 価格上昇予測` という解釈がコード実装と一致していない**
4. **`+100bps 改善余地` の見積りが強すぎ、HFT 目的とも衝突する**

従って、`391#` は「即実装計画」ではなく、**有効仮説を残しつつ、危険な因果推論を補正する監査メモ**として扱うのが妥当です。

---

## 2. 主要指摘

### 1. CRITICAL: `391#` は mixed-SHA 集計であり、実装原因の断定に使い過ぎない方がよい

`391#` は `2026-03-09 ~ 2026-03-12` を単一の root cause 窓として扱っていますが、raw fill record を見ると日別に複数 `git_sha` が混在しています。

確認結果:

| ファイル | records | filled | 30s PnL sum | 主な SHA |
|---|---:|---:|---:|---|
| `results/v460/fill_test/fill_records_20260309.jsonl` | 603 | 188 | +32.2805 | `819ec73b2081`, `0d22298c5e7e`, `22a4fc583078` |
| `results/v460/fill_test/fill_records_20260310.jsonl` | 190 | 51 | +21.7662 | `27d6acd90c2b`, `22a4fc583078` |
| `results/v460/fill_test/fill_records_20260311.jsonl` | 85 | 28 | +13.5242 | `b2a902cc3f854e9322bfabf27cfb16400c80e665`, `b359a5ba5be9` |
| `results/v460/fill_test/fill_records_20260312.jsonl` | 397 | 148 | -12.6239 | `92c588e535de`, `66165ee5c12f` |

この状態で `docs/v460/391_fill_test_deep_dive_root_cause.md:7` のように「381# YAML 変更適用済」の前提を置いても、**実際には複数世代のコード/設定が同じ集計に混ざっています**。

したがって、以下のような断定は強すぎます。

- `381#` の `offset_ceiling_ratio_buy` が buy 劣化を招いた
- 現行 offset pipeline の本質が 4 日窓で証明された
- 現行実装の `EV` ロジックが主要因である

**推奨対応**:

1. `391#` の分析を `git_sha` ごとに切り直す
2. 最低でも `same-SHA / same-config / same-date-range` の表を追加する
3. 実装修正の根拠に使うのは current SHA 限定 deep dive のみにする

### 2. HIGH: headline 数値が raw JSONL から再現できず、`391#` の集計条件が不明

`docs/v460/391_fill_test_deep_dive_root_cause.md:4`-`docs/v460/391_fill_test_deep_dive_root_cause.md:5` は以下を headline に置いています。

- `総レコード数 1,270`
- `約定数 411`
- `累計 30s PnL +67.7 bps`

しかし raw 4ファイルをそのまま集計すると、確認できたのは次です。

- `records = 1275`
- `filled = 415`
- `30s PnL sum = +54.9471 bps`

また `docs/v460/391_fill_test_deep_dive_root_cause.md:70` の「TOP 15 損失は全て sell」も、raw 上では **14 sell + 1 buy** でした。

この差分自体は「391 の結論が全て誤り」という意味ではありません。ただし、**フィルタ条件が書かれていない以上、第三者が再現できない** という点が問題です。

レビューとしてはここを軽視できません。再現不能な深掘りは、後から都合のよい切り出しに見えやすく、以後の A/B 判定も壊します。

**推奨対応**:

1. `391#` 末尾に「除外条件」を明記する
2. 可能なら `analysis/*.py` に固定し、`analysis_results/*.json` を残す
3. `top loss`, `gap bucket`, `EV bucket` も同じフィルタから再計算する

### 3. HIGH: `EV` 解釈がコードとズレており、`sell + EV>0` を「逆方向売り」と読むのは誤り

`docs/v460/391_fill_test_deep_dive_root_cause.md:160`-`docs/v460/391_fill_test_deep_dive_root_cause.md:172` は、

- `EV > 0 -> 価格が上がる予測`
- `sell + EV>0 -> 上がるのに売っている`

という読みをしています。

ここはコード整合上、そのままでは通りません。

根拠:

- `ztb/ml/skip_gate.py:344`-`ztb/ml/skip_gate.py:455`
  - `SkipGate.evaluate()` の `predicted_pnl_bps` は **その注文の predicted PnL** です
  - `side` を受け取る side-aware な判定です
- `scripts/v460/lib/skip_gate_evaluator.py:592`-`scripts/v460/lib/skip_gate_evaluator.py:670`
  - `ev_weighted` は `ev_score` を計算しますが、`skip_gate_ev_as_offset_enabled=true` では **side 決定ではなく offset 修飾子** として使われます
- `configs/v460/fill_test.yaml:318` と `configs/v460/fill_test.yaml:324`
  - 現在 `ev_weighted_enabled: true`, `ev_as_offset_enabled: true`
- `scripts/v460/lib/side_selector.py:76`-`scripts/v460/lib/side_selector.py:188`
  - side は交互ロジック、microprice、smart-side、inventory-aware などで決まります

つまり、現行系では `sell + EV>0` だからといって、直ちに
「モデルは上昇を予測したのに、システムが逆方向の sell を出した」
とは言えません。

ここから言えるのはもっと限定的です。

- **sell 側の EV-positive bucket は、現行の side selection + offset execution + market state の組み合わせで underperform している**

です。これは重要ですが、`391#` の解釈より狭く、かつ実装修正も変わります。

**推奨対応**:

- `sell_ev_positive gate` を即実装しない
- 先に `ev_score_pretrade × side × microprice_bias_bps × inventory_escape × smart_side` の交絡を分離する
- 修正対象は「方向ガード」ではなく **sell-side EV calibration** または **EV を使う既存 offset ロジックの side別再較正** に置く

### 4. HIGH: `P0-2` の `min_fill_interval_seconds=300` は期待値見積りが強すぎ、HFT 目的とも衝突する

`docs/v460/391_fill_test_deep_dive_root_cause.md:364`-`docs/v460/391_fill_test_deep_dive_root_cause.md:373` は、

- `gap < 5min` の 203 件が `-0.427bps`
- よって `5分制限` で `+87.3bps` 改善余地

という整理ですが、これは **反実仮想が粗い** です。

理由は 2 つです。

1. **skip した fill の対照 PnL を 0bps とみなしている**
   - 実際には、その fill を飛ばした結果として次の有利 fill を失う可能性もあります
2. **HFT を目指す目的と正面衝突する**
   - `300秒` の一律 pause は high-frequency participation を自ら捨てる設計です

しかも現行スタックには、既に rapid-fire/連鎖損失を吸収する部品があります。

- `configs/v460/fill_test.yaml:550` `dynamic_cycle_interval`
- `configs/v460/fill_test.yaml:596` `loss_cooldown_threshold_bps`
- `configs/v460/fill_test.yaml:605` `toxic_fill_veto_threshold_bps`
- `configs/v460/fill_test.yaml:647` `sell_dynamic_kill`
- `configs/v460/fill_test.yaml:667` `buy_dynamic_kill`
- `configs/v460/fill_test.yaml:700` `narrow_spread_pause`

したがって profit-first の順序は、`新しい 5 分ハード制限` ではなく **既存 rapid-fire 防御の side条件付き再較正** です。

**推奨対応**:

1. 一律 `300秒` は採用しない
2. まず `sell` 限定、かつ `直前 adverse_selected / wide spread / loss streak` 条件付きで試す
3. `pause` より先に `offset boost` や `threshold strict化` の soft guard から入る

### 5. MEDIUM: `offset_ceiling_ratio_buy 0.20` を主犯に置くには因果が弱い

`docs/v460/391_fill_test_deep_dive_root_cause.md:238`-`docs/v460/391_fill_test_deep_dive_root_cause.md:249` は、`3/12` buy 劣化を `offset_ceiling_ratio_buy` 変更に結びつけています。

ただし、ここも mixed-SHA と one-day compare の制約が大きいです。

`results/v460/fill_test/fill_records_20260312.jsonl` の buy filled を SHA 別に見ると、少なくとも以下でした。

| SHA | buy fills | mean 30s PnL | mean offset |
|---|---:|---:|---:|
| `92c588e535de` | 48 | -0.7594 | 0.2005 |
| `66165ee5c12f` | 26 | -0.7038 | 0.1986 |

両方で buy は悪く、offset 平均も近いです。つまり「0.20 ceiling が怪しい」仮説自体は残る一方、**3/12 単日観測から ceiling だけを主犯認定するには材料不足** です。

市場理論的には、buy 劣化は ceiling だけでなく、以下でも説明できます。

- ranging 終盤での stale quote
- low-information 帯での passive participation 過多
- regime confidence 高値帯の誤参加
- side alternation / microprice safety による参加位置の悪化

**推奨対応**:

- `0.15 / 0.18 / 0.20` を same-SHA ladder で比較する
- その際、`buy_ranging`, `spread bucket`, `microprice_bias bucket` を固定して見る
- 1日観測だけで rollback しない

---

## 3. 採用してよい論点

`391#` の中でも、以下はそのまま次の仮説として残してよいです。

1. **sell tail asymmetry は実在する**
   - raw 上でも worst single fill と tail loss 累計は明確に sell 側が重い
2. **wide spread は toxic participation の候補**
   - 市場理論上も `spread >= 3bps` は低流動性/情報流入のシグナルとして自然
3. **rapid-fire deterioration は plausible**
   - ただし hard pause ではなく conditional guard で扱うべき
4. **confidence [0.7,0.9) は再点検価値が高い**
   - Bayesian regime の過信帯や、transition lag の可能性がある

要するに、`391#` は「観測された異常セグメント」は残し、**その説明理論だけ補正する** のが正しい扱いです。

---

## 4. 利益優先での次アクション

### P0

1. `391#` を same-SHA で再集計し、filter manifest を明記する
2. `sell + EV>0` は「方向ミス」ではなく「sell-side EV miscalibration」として再定義する
3. `rapid-fire` は一律 5 分禁止ではなく、既存 `sell_dynamic_kill` / `toxic_fill_veto` / `dynamic_cycle_interval` の sell 側再較正で先に試す

### P1

1. `spread >= 3bps` の hard skip / offset boost を same-SHA で A/B する
2. `confidence [0.7,0.9)` を `regime`, `macro_trend`, `microprice_bias_bps` で再分解する
3. `buy ceiling` は `0.15 -> 0.18 -> 0.20` ladder で比較する

### P2

1. `391#` の deep dive を `analysis/*.py` に昇格し、再利用可能にする
2. `EV`, `side`, `offset_stage`, `decision_path`, `inventory_escape` を横断した監査表を残す
3. 市場理論システムの追加より先に、既存ガードの重複・干渉を整理する

---

## 5. 結論

`391#` は「どこで損しているか」を嗅ぎ分ける文書としては有用です。
しかし、**原因の断定と改善余地の見積りは、そのまま採用しない方がよい**です。

最重要の補正は次の 3 点です。

1. **mixed-SHA を分離する**
2. **`EV` を price-direction と誤読しない**
3. **rapid-fire を一律 pause にせず、既存ガードの再較正で扱う**

profit-first で見ると、今やるべきは新しい大きな仕組みの追加ではありません。
**`391#` の有効仮説を current-SHA で再検証し、既存ガードを最小修正で正しく効かせること**です。
