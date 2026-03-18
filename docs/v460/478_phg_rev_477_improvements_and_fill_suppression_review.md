# 478# 475-477 改善レビュー: fill 抑制構造の再点検と見落とし論点

**種別**: rev  
**対象**: 475#, 476#, 477# ならびに関連実装  
**日付**: 2026-03-18

---

## §0 結論

475-477 で行われた改善には、**数式修正として正しいもの** と **問題の切り方がまだ粗いもの** が混在している。

まず評価すべき点は明確である。

1. 474# 由来の `_recalc_price_with_new_offset()` 修正は数式として妥当
2. micro-timeout の再クオート式修正も妥当
3. dust sweep と lot scale の衝突修正は必要だった
4. retrain_scheduler 多重起動対策の方向も妥当

一方で、477# の「20層 suppression が fill 17.7% を作っている」という総括は、**観測としては有益だが、因果整理としてはまだ不十分** である。主な理由は次の4つである。

1. **分析対象データが pre/post 修正を跨いでいる** ため、現行系の主犯をそのまま断定できない
2. **hard skip / soft reduction / 発注後 timeout / 資金不足** を同じ suppression として足し合わせている
3. **新設 Gate 2b は soft mode では独立した sell filter ではなく、既存 low-vol boost への委譲に過ぎない**
4. **fill 低下の一部はガード過多ではなく、資金制約と price feasibility 制約** である

したがって reviewer としての最終結論は次である。

> 475-476 のコード修正自体は概ね筋が良い。しかし 477 の root cause 整理は「抑制が多い」までは正しくても、「今どの抑制が本当に fill を殺しているか」の優先順位付けがまだ甘い。現時点の主因は一枚岩の suppression ではなく、**混在コホート、資金制約、price feasibility、買い側 hard gate の偏り** の4系統である。

---

## §1 先に総評

### §1.1 良かった点

- 475# のうち、mmap を却下し shadow order を defer に置いた判断は妥当
- 476# の dust sweep 修正は、実際に `dust_sweep_active` 時に lot scale chain をバイパスしており、問題の芯に触っている
- 474#/475# の `_recalc` 修正と micro-timeout 修正は、以前の意味論破綻を一段減らしている
- 477# は cancel reason 集計だけでなく、soft reduction まで棚卸しした点に価値がある

### §1.2 まだ危うい点

- 477# は「現在の fill 低下」と「修正前を含む履歴上の suppression」を半ば同一視している
- 475# は Gate 2b を buy 側対称の新防御として書いているが、soft mode 実装はそこまで強くない
- 476# は「0.001 単位切り捨て廃止」と書くが、preflight failure path にはなお `min_order_btc` 単位の丸めが残っている
- 低 fill rate を suppression 一色で説明すると、資本制約と実現不能価格の問題を過小評価する

---

## §2 HIGH: 477# の最大の弱点はコホート混在である

477# 自身が末尾で書いている通り、対象ログは

- `git_sha=0dd7bacaa`
- しかも **476# 適用前後を跨ぐ**

というデータである。

これは reviewer としてかなり重要で、477# の上位論点のうち少なくとも次はそのまま現行系へ外挿できない。

1. `ranging_low_vol_skip 217件`  
   現在の config では `skip_ranging_buy_low_vol: true` だが `ranging_buy_low_vol_as_offset: true` であり、buy 側は hard skip ではなく soft mode 委譲である。

2. sell 側の残高不足 71件  
   476# で dust sweep 周りを触っているため、同じ頻度で続くとは限らない。

3. cooldown lot scale の効き方  
   476# 後は `_min_lot = min_order_btc` に変わっており、旧 cohort と同じ意味で「実質 skip」とは言いづらい。

よって 477# の集計は**履歴上の抑制地図**としては使えるが、**現行系の優先順位表**としてはそのまま使えない。ここは 461#/462# で指摘した schema/run/config drift と同種の問題である。

### §2.1 実測補強: 3/17-3/18 の 915 レコードは failure mode が SHA ごとに分断されている

今回、`.venv` 上で補助スクリプト `temp/analyze_478_fill_funnel.py` を追加して 3/17-3/18 の fill_records を再集計した。結果は、478# の指摘をかなり強く裏付ける。

- 総件数: **915 records / 163 fills**
- `git_sha=f840d0e0aa5e`: **321件**, fills=93, feasibility=115, post_submit=54, gate=59
- `git_sha=d0769f283da3`: **217件すべて gate**, しかも **`ranging_low_vol_skip` 217件**
- `git_sha=b70365d4d4c9`: **135件中 134件が preflight**, 主因は `preflight_insufficient`

つまり 477# の主要 cancel reason 上位は、単一実装の同時多発ではなく、**別 SHA・別運転状態の failure mode を縦に積んだ合成結果** である。

この一点だけでも、

> 「ranging_low_vol_skip が現行 fill 低下の最大主犯」

という 477# の読みに、そのままは乗れない。

---

## §3 HIGH: 477# は suppression を同列に数えすぎている

数学的に見ると、477# の「20層」は存在するが、**同じ確率空間に並ぶ20個の独立障壁ではない**。

少なくとも以下は分ける必要がある。

1. **Hard pre-trade gate**  
   例: `ranging_low_vol_skip`, `skip_gate`, `spread_too_narrow`

2. **Balance / system constraint**  
   例: `preflight_insufficient`, `route_to_kill_deadlock`, `status_unknown_fast`

3. **Soft modifier**  
   例: offset boost, lot scale, cross-venue retreat

4. **Post-submission outcome**  
   例: `timeout`, `post_only_reject`, stale cancel

これらを横並びにして「20層が 82.3% を潰す」と言うと、直感的には強いが、確率論としては粗い。なぜなら

$$
P(\text{fill}) = P(\text{pass preflight}) \times P(\text{pass gates} \mid \text{preflight}) \times P(\text{quote feasible} \mid \cdots) \times P(\text{fill before timeout} \mid \cdots)
$$

であり、cancel reason の単純合算はこの条件付き構造を潰してしまうからである。

たとえば `timeout` は suppression ではなく、**既に注文を出せた後の fill failure** である。`preflight_insufficient` と同列に数えると、設計改善と資本制約と市場到達失敗の境界が消える。

したがって 477# の集計は useful だが、次の段階では

- 発注前 block 率
- 発注到達率
- 到達後 fill 率
- fill quality

に分解し直すべきである。

### §3.1 実測補強: funnel 再分類では `gate 334 / preflight 162 / feasibility 154 / post_submit 102 / filled 163`

同じ 915 レコードを、reviewer 観点で次の5群に再分類した。

- `gate`: 334
- `preflight`: 162
- `feasibility`: 154
- `post_submit`: 102
- `filled`: 163

ここから分かるのは、477# の「20層 suppression」が誤りということではない。誤りなのは、**その 20 層が同じ性質の壁として並んでいるかのように読める点** である。

特に `post_submit=102` は、すでに発注に到達した後の friction であり、`preflight=162` や `gate=334` とは設計上の意味が全く異なる。

したがって fill 改善の優先順位付けは、単純な cancel reason 順位表ではなく、**どの段で落ちているのか** を first-class な指標として扱うべきである。

---

## §4 HIGH: 475# Gate 2b は「売り対称の新保護層」と言うには弱い

475# は `skip_ranging_sell_low_vol` を Gate 2b として追加し、buy 側と対称のフィルタだと説明している。しかし実装上、現在の config は

- `skip_ranging_sell_low_vol: true`
- `ranging_sell_low_vol_as_offset: true`

であり、soft mode 時の `CycleGateAggregator._check_ranging_sell_low_vol()` は **blocked=False を返すだけ** である。

その後に実際に起こるのは、売り専用の新ロジックではなく、`maker_regime_boost._regime_boost_low_vol()` による **既存の全 side 共通 low-vol boost** である。

したがって reviewer の表現としては、これは

> 「売り対称 hard gate の実装」

ではなく、

> **「sell 側にも low-vol 条件を audit trail 上で明示し、既存 generic boost に接続した」**

が正確である。

この違いは重要で、もし 475# を「sell suppression を大きく増やした変更」と読んでいるなら過剰解釈である。逆に「sell の危険帯を audit trail に載せ、既存 boost へ接続した変更」と読むなら妥当である。

---

## §5 HIGH: 476# は正しいが、「0.001 切り捨て完全廃止」はまだ言い過ぎである

476# の中心修正、すなわち

- `balance_checker` 側の satoshi 精度 round
- `dust_sweep_active` 時の lot-scale chain バイパス
- `order_monitor` 側の切り捨て除去

は正しい。

ただしレビュー上は次を補足すべきである。

`orchestrator_balance._handle_preflight_failure()` では、連続 preflight failure 時の縮小ロットがなお

```text
int(raw_shrunk / min_order_btc) * min_order_btc
```

で量子化されている。

今の `min_order_btc=0.001` では、ここは依然として 0.001 刻みである。したがって、

> 「0.001 単位切り捨てを廃止した」

は主要 hot path については概ね正しいが、**全経路で完全撤廃ではない**。

もっとも、これはただちに 476# の価値を損なうものではない。問題は主張の精度である。レビュー文書としては

> 「主要発注経路の切り捨てを除去したが、preflight recovery path には min_order 単位丸めが残る」

まで書くべきである。

---

## §6 HIGH: 現在の fill 低下は suppression だけでなく、資本制約そのものが強い

477# は `preflight_insufficient` を 19.1% と正しく拾っているが、その重みづけがまだ弱い。

現行 config は

- `order_quantity = 0.001`
- `min_order_btc = 0.001`
- `balance_margin_ratio = 1.01`

であり、しかも 477# 自身の記述では JPY 残高は約 15,600 JPY 水準である。これは「1注文は出せるが、ほぼ遊びがない」状態である。

この状況では、fill rate を低下させる構造は suppression 以前に

- 片側約定で反対側が枯れる
- route-to-kill deadlock が出やすい
- buy/sell を対称運用する資本余裕がない

という **inventory financing 問題** になる。

市場理論的にも、マーケットメイクの期待利益は単に約定率ではなく

$$
E[\Pi] \approx \lambda \cdot (\text{spread capture} - \text{AS loss} - \text{inventory cost})
$$

で効く。ここで資本制約が厳しいと inventory cost が実質的に跳ね上がり、片側停止と opportunity loss が増える。よって reviewer としては、477# の

> suppression が主因

という整理に、

> **小さすぎる運転資本が suppression を増幅している**

を追加する必要がある。

---

## §7 MEDIUM: `no_feasible_quote` は「抑制しすぎ」だけではない

477# は `no_feasible_quote` を「防御を重ねるほど発注不可能になる」という物語で捉えている。これは半分正しいが、半分はミスリードである。

現実には `fill_cycle_executor` で `NO_FEASIBLE_QUOTE` になるのは、主として `maker_price.compute()` が連続で `InfeasibleQuoteError` を返した後である。つまりこれは

- spread 制約
- sell max spread 制約
- cross-venue veto
- post-only 成立不能に近い価格幾何

など、**制約集合そのものが空になった** ことの表現である。

したがって、これは suppression の心理的問題というより

> **制約設計の feasibility 問題**

である。対策も「緩めればよい」では足りず、どの制約同士が空集合を作っているかを分解しないと再発する。

この点で 477# の P1 は方向として悪くないが、なお「multiplier 累積のせい」と単純化しすぎている。

---

## §8 MEDIUM: cooldown lot scale の読みは current state では少し古い

477# は `Cooldown lot scale (0.3x)` を強い suppression として挙げている。履歴上そうだった局面はあるが、現在の executor では

- `_min_lot = config.min_order_btc`
- base `order_quantity = min_order_btc = 0.001`

なので、基準ロットが最小値のままなら `0.001 × 0.3` は結局 `0.001` に床上げされる。

つまり current state の cooldown scale は

- baseline lot ではほぼ no-op
- balance-linked expansion や recovery 後の拡大ロットには効く

という位置づけであり、少なくとも「現在の fill 17.7% の主犯」の表現は強すぎる。

---

## §9 市場理論面での補強

### §9.1 fill rate は目的変数ではなく制約つき媒介変数である

477# も末尾で「fill を上げるだけでは黒字化しない」と書いており、これは重要である。レビューとしてさらに強めるなら、maker の最適化は

- fill rate 最大化
- fill quality 最大化

のどちらでもなく、**期待利得の最大化** である。

とくに Coincheck のような相対的低流動性市場では、low-vol ranging が「安全」ではなく、

- spread は広いが flow が薄い
- queue で待たされる
- たまに来る informed flow にだけ当たる

という悪い組合せもありうる。したがって `ranging_low_vol_skip` を全面解除すれば solve するという発想は危険である。

### §9.2 ただし現在の buy 側 hard gate は偏りすぎている

一方で、現在の buy 側は

- ranging buy hard gate の履歴負債
- no_feasible_quote の buy 偏重
- preflight insufficiency の buy 側頻発

が重なっており、**探索機会そのものが痩せている**。これは学習・検証の統計効率を悪化させる。

よって reviewer の立場では、

> PnL を守る suppression は必要だが、buy 側だけ機会集合を過剰に削る現状は、検証系としてもよくない

と整理する。

---

## §10 個別改善レビュー

### §10.1 475# の提案評価レビュー

- mmap 却下: 妥当
- XGBoost micro-prediction defer: 妥当
- Shadow Order defer: 妥当。ただし将来性は高い
- Gate 2b 追加: 低リスクの観測強化としては妥当
- GC 追加: fill rate 改善には直結しない。運用安定化策として読むべき

### §10.2 476# の改修レビュー

- dust sweep が lot scale chain に潰される問題を直した点は非常に良い
- buy 側 balance-linked lot 拡大も資本効率の観点では合理的
- ただし preflight recovery path の量子化残りはドキュメント上明記した方がよい

### §10.3 477# の分析レビュー

- cancel reason 棚卸しは有益
- ただし現行因果としては oversell 気味
- 「20層」という見せ方は問題提起には強いが、実装優先順位決定にはまだ粗い

---

## §11 著者が見落としている点

1. **コホート混在**  
   476# 前後を跨ぐログから current bottleneck を直接決めている。

2. **sell Gate 2b の実体**  
   soft mode では新しい sell 専用保護ではなく、既存 generic low-vol boost への接続である。

3. **preflight recovery path の丸め残り**  
   0.001 切り捨ては主要経路では減ったが、完全には消えていない。

4. **資本制約の過小評価**  
   fill 低下の相当部分は suppression というより運転資本不足である。

5. **条件付き確率構造の欠落**  
   timeout, skip, insufficient, infeasible を同じ suppression として扱っている。

6. **feasibility と policy の混同**  
   `no_feasible_quote` は単なる保守化ではなく、制約集合崩壊のシグナルである。

7. **micro-timeout の痕跡の読み落とし**  
   `micro_timeout` という cancel reason は 0 件でも、`requote_attempts > 0` のレコードは同期間に **106件** ある。つまり post-submit friction は cancel reason だけでは見切れず、再クオートの発生自体を別軸で見る必要がある。

---

## §12 最終判定

475-476 の改善は、**コード修正としてはかなり健全** である。特に価格再計算と dust sweep の修正は、以前の構造的不整合を減らしている。

ただし 477 の総括は、レビューとして次のように言い換えるべきである。

> fill 17.7% は「20層 suppression」だけで説明されるわけではない。実際には、修正前後が混ざった分析対象の上で、buy 側 hard gate の履歴負債、資金制約、制約集合崩壊、発注後 timeout が重なって見えている。

したがって次の実務優先順位は、私ならこう置く。

1. **同一 config / 同一 SHA / 同一運転資本で fill funnel を再計測**
2. **preflight / gate / quote feasible / post-submit timeout を分離して再集計**
3. **buy 側 hard gate と no_feasible_quote の現在寄与を再評価**
4. **その後に suppression 緩和を行う**

特に 477# の P0 として `ranging_low_vol_skip` 緩和を直ちに置くのは、同一 cohort で再計測する前だと危うい。現行 config では既に buy 側が soft mode に寄っているため、まず確認すべきは「今も本当にそこが最大の壁か」である。

今の 477# は、問題提起としては十分に価値がある。しかし reviewer としては、現段階ではまだ

> 「suppression 過多が fill を殺す」

よりも、

> **「suppression・資本制約・feasibility 制約が混ざって fill funnel を詰まらせている」**

の方が正確である。
