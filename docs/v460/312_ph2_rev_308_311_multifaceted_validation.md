# 312# 308#-311# 再レビュー: 多角的検証と優先順位の再整理

> **対象**: [308#](308_ph2_gemini_31_pro_review_306_307_inverted_microstructure.md), [309#](309_ph2_review_response_307_308_fixes.md), [310#](310_ph2_impl_design_improvements.md), [311#](311_ph2_rpt_observational_comparison_rerun.md)  
> **観点**: システム工学 / 市場微視構造 / 統計解釈 / 実験設計 / 運用優先度  
> **日付**: 2026-03-07  
> **立場**: 実装者ではなくレビュワーとして、結論の強弱と次の一手を整理する

---

## 1. Findings

### F1. 311# は有用な「回顧分析」だが、現行システム評価としては mixed-SHA 汚染が強い【HIGH】

`analysis/311_observational_rerun.py` は `results/v460/fill_test` を丸ごと読み、`git_sha` / `run_id` / `date_from,to` でのフィルタを一切使っていない。  
一方で 311# 本文は、自身で

- データの大部分は `pre-310#`
- `310#` デプロイ直後で新データがほぼない
- 168h 計測は `dcc3064a8` で再スタート

と認めている。

つまり 311# の数値は、

> 「いま動いている実装の評価」

ではなく、

> 「複数 SHA を跨いだ22日間の構造分析」

として読むべきである。

実際、2026-03-07 の再実行でも `311#` 記載値 (`filled=2564`) と今日の再実行値 (`filled=2568`) に軽微な差があり、データ母集団はまだ動いている。ただし、`sell p10≈-6.87`, `trending_up sell 3/3 FAIL`, `none≈10%`, `危険時間帯の偏在` という構造自体は維持されている。

**推奨**:

1. `ztb/metrics/fill_quality.py` の既存 `apply_fill_record_filters()` を使い、`git_sha=dcc3064a8` または `date_from=2026-03-07` 以降で再集計する
2. 311# の改善案は「現状への即適用」ではなく「仮説の棚卸し」として扱う
3. Gate 判断と構造分析を同一レポートで混ぜない

### F2. 310#/311# の spread capture / AS cost 分解式は、現状の price 定義と整合していない【CRITICAL】

310# と 311# は次を採用している。

- `spread_capture = spread_bps × effective_offset_used`
- `as_cost = spread_capture - realized_pnl`

しかし `maker_price.py` の価格式は、

- buy: `price = best_bid + spread × ratio`
- sell: `price = best_ask - spread × ratio`

である。  
この定義では、ratio を大きくするほど quote は **より内側・より攻撃的** になり、mid に対する理論上の受取 spread はむしろ減る。

コードからの推論として、即時の mark-to-mid 的な capture は

`(0.5 - ratio) × spread_bps`

方向で効くはずで、`ratio × spread_bps` ではない。

したがって現在の

- `spread_capture`
- `as_cost`
- `efficiency`
- `D-buy` を P0 に上げる優先順位ロジック

は、そのまま信用してはいけない。

311# 自身も「buy efficiency は分母が小さいため誤読しやすい」と気付いているが、根本問題は **効率指標の解釈** ではなく **分解式そのものの妥当性** にある。

**推奨**:

1. 分解式を `maker_price.py` の quote 定義に合わせて再導出する
2. `spread_capture` は mid 基準での理論 capture に置き換える
3. `311` の `D-buy P0` など efficiency ベースの優先度付けは一旦無効化する

### F3. 310# A の「sell_hour_offset_boost 効果検証」は、介入効果の測定になっていない【HIGH】

`analysis/311_observational_rerun.py` の `sell_hour_boost_analysis()` は、UTC `8/13/14/16` の sell を「boost対象」、それ以外を「非boost」として比較している。

これは

- 高 AS の時間帯
- それ以外の時間帯

を比較しているだけであり、

> boost を入れたから良くなったか

を測っていない。

しかも 311# 本文が認める通り、データの大部分は `pre-310#` である。よって「Boost対象の方が悪い」という結果はほぼ自明で、施策失敗の証拠にはならない。

市場理論的にも、高毒性時間帯は元々悪い。必要なのは

- 同一時間帯内の pre/post 比較
- 同一 SHA 内の with/without 比較
- 可能なら matched comparison

である。

**推奨**:

1. `sell_hour_offset_boost` は hour-based selection bias を除いた評価に切り替える
2. 最低でも `UTC 8/13/14/16` のみを抽出し、`pre-310 vs post-310` で比較する
3. 現段階で `312-B` を強く進めるのは早い

### F4. 311# の「動的フロア割引が主因」仮説は有望だが、まだ因果とまでは言えない【HIGH】

311# の sell quintile はかなり示唆的である。

| Quintile | Offset帯 | PnL | AS |
|---|---|---:|---:|
| Q1 | 0.136–0.268 | -0.64 | 33.9% |
| Q2 | 0.268–0.300 | -0.71 | 30.7% |
| Q3 | 0.300固定 | -0.24 | 19.7% |
| Q4 | 0.300–0.482 | +0.60 | 32.1% |

ただし、これは causal ではなく endogenous である。低 offset 群には以下が混ざる。

- inventory skew
- regime
- 時間帯
- none/trending/ranging
- 他ステージの相互作用

したがって `sell_offset_floor_inv_discount=0.5 -> 0.7` を今すぐやれば改善するとまでは言えない。  
ただし、**「Q1/Q2 が明確に劣後しているので、floor discount を最優先仮説に置く」** こと自体は妥当である。

**推奨**:

1. `ranging × 危険時間帯 × buy偏重在庫` に絞った層別分析を行う
2. floor discount は hot-reload 可能な YAML 変更なので、post-310 データを48–72h見た後に最小変更で試す
3. Q1→Q3 移動を「fill消失込み」で評価する。PnL 改善だけでなく fill_rate 低下も同時に見る

### F5. 308# の L2 指摘は鋭いが、L1 指摘はやや一般化しすぎている【MEDIUM】

308# が指摘した L2 microprice 反転は妥当だった。これは 309# で安全側へ修正された。

一方、L1 について 308# は

> 高σ時は Maker は常に休むべき

という強い言い方をしているが、これは一般理論としては言い過ぎである。  
継続気配更新型の MM では、高ボラ時に **参加頻度を下げる** のと **観測・取消更新の頻度を上げる** のは両立しうる。

ただし、この bot の `dynamic_cycle_interval` は「独立した発注サイクルの間隔」を直接伸縮させる設計なので、現実装に限れば 309# の cooldown 方向は妥当である。

結論として、

- 308# の L2 批判は強い
- 308# の L1 批判は「この bot では妥当」だが「一般論として絶対」ではない

と整理するのが正確である。

### F6. `none` レジーム追加対策は、303# の passive MM を踏まえた post-filter 再評価が先である【MEDIUM】

`configs/v460/fill_test.yaml` では既に

- `none_regime.passive_mm_enabled: true`
- `none_regime.fixed_offset_bps: 2.0`

が有効化されている。

それにもかかわらず 311# が `312-C: none × 1.3 conservative multiplier` を候補に置いているのは、mixed-SHA データを現在の none 問題として読んでいる可能性が高い。

もし `303#` 以降、特に `310#` 以降の `none` がまだ悪いなら、次に検討すべきは単なる multiplier 追加ではなく、

1. `sell none` の hard veto
2. `sell none` の passive-only 徹底
3. `none` の warmup / detector-miss / stale-signal 分解

である。

**推奨**:

1. `none` は post-303 / post-310 限定で再集計する
2. その上でなお `sell none` が悪ければ multiplier より veto を優先する

### F7. 309#/310# は「安全化」としては正しいが、「収益改善が確認できた」とはまだ言えない【MEDIUM】

今日の再実行でも、

- None除外: sell `p10=-6.8683`
- trending_up sell: `3/3 FAIL`
- sell 危険時間帯の偏在

はそのまま残っている。

したがって 309#/310# は、

- 逆方向ロジックの除去
- 観測基盤の整備
- 防御レイヤーの追加

としては評価できるが、まだ **profitability reconstruction** には到達していない。

特に L2 は現状 `enabled: false` であり、利益への寄与は未検証である。ここを「直したから良くなるはず」と先走らない方がよい。

### F8. 308# 文書には制御文字混入があり、監査・検索性を落としている【LOW】

`308#` には文字化け・制御文字混入があり、例えば `analysis_results` や `fill_timestamp` の記述が壊れている箇所がある。内容理解は可能だが、

- 検索性
- diff 可読性
- 再レビュー時の信頼性

を下げる。地味だが直した方がよい。

---

## 2. 妥当だった点

308#-311# の流れで良かった点は以下である。

1. 309# で `306_deep_dive` のスキーマ齟齬を止血したこと
2. 309# で L2 の逆方向ロジックを修正し、同時に無効化したこと
3. 310# で `decision_path`, `none_regime` 観測、sell 時間帯ブーストなどを分離実装したこと
4. 311# で sell tail が `ranging` と危険時間帯に偏ることをかなり明瞭に示したこと
5. `tests/unit/v460/test_306_proposals.py` が 2026-03-07 再実行でも **67 passed** で壊れていないこと

---

## 3. 優先順位

### P0

1. `311` 系分析を `git_sha` / `run_id` / `date_from` フィルタ付きで再実行する
2. spread capture / AS cost 分解式を修正し、`310# E` と `311# §6` を再解釈する
3. `sell_hour_offset_boost` は hour 内 pre/post 比較へ作り直す

### P1

4. floor discount は `ranging × 危険時間帯 × buy偏重在庫` で層別検証する
5. `none` は post-303/post-310 限定で見直し、必要なら multiplier ではなく veto/passive-only を検討する
6. L2 safety mode は再有効化しても shadow か限定条件で始める

### P2

7. `311` の改善提案自動導出ロジックから `efficiency` 依存を外す
8. 308# の文書ノイズを整形し、後続レビューの入力品質を上げる

---

## 4. 総評

308#-311# は、理論倒錯の修正と観測強化という意味では前進している。  
ただし、結論の強さにはかなり差がある。

現時点で強く言えるのは次の4点である。

1. **sell のテール損失が主問題であり、特に ranging と危険時間帯が重い**
2. **309#/310# は安全化として妥当だが、収益改善の実証はまだない**
3. **311# は mixed-SHA 回顧分析として有益だが、現行構成の Gate 根拠には弱い**
4. **310#/311# の spread capture 分解は再計算が必要で、このままでは優先順位を誤る**

---

## 5. 確認メモ

今回のレビューでは以下を照合した。

- `docs/v460/308_ph2_gemini_31_pro_review_306_307_inverted_microstructure.md`
- `docs/v460/309_ph2_review_response_307_308_fixes.md`
- `docs/v460/310_ph2_impl_design_improvements.md`
- `docs/v460/311_ph2_rpt_observational_comparison_rerun.md`
- `analysis/311_observational_rerun.py`
- `analysis/306_deep_dive.py`
- `scripts/v460/lib/side_selector.py`
- `scripts/v460/lib/maker_price.py`
- `scripts/v460/lib/fill_loop_orchestrator.py`
- `scripts/v460/lib/param_adapter.py`
- `configs/v460/fill_test.yaml`

また、2026-03-07 時点で以下を実行した。

1. `./.venv/Scripts/python.exe -m pytest tests/unit/v460/test_306_proposals.py --no-cov`  
   → **67 passed**
2. `./.venv/Scripts/python.exe analysis/311_observational_rerun.py`  
   → `filled=2568`, `sell p10=-6.8683`, `trending_up sell 3/3 FAIL`, `none=10.4%` を再確認
