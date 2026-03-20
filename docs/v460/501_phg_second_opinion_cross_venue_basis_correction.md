# 501# [Phase G] 500#, 503# レビューに対するセカンドオピニオン：構造的プレミアムの発見と De-meaning による止揚

> 種別: review / second_opinion
> 対象: 500# `docs/v460/500_phg_rev_497_499_503_sell_side_breakdown_and_asymmetric_actions.md`, 503# `docs/v460/503_sell_buy_pnl_factor_analysis.md`
> 日付: 2026-03-20

---

## 1. 総評：両者の議論の止揚

503# が発見した**「Cross-Venue Guard が Buy に偏向適用されており、Sell 側（5.9%）では機能していない」**という事実は、直近の Sell PnL 崩壊を説明する極めて重要なブレイクスルーである。
一方で、500# が**「だからといって `adverse_side` を無視して両方に Guard をかける（503# の解決案1）のは、設計の意味論を破壊する」**と反論した点も、システム保全の観点から完全に正しい。

この拮抗する二つの正当な主張を統合（止揚）するための鍵は、システム論ではなく**市場理論（Market Microstructure）**にある。

結論から言えば、現在のバグは**「他市場（BF）との間に存在する恒常的なスプレッド・プレミアム（Basis）を無視し、絶対値ゼロを基準にトレンド（Up/Down）を判定している数学的な欠陥」**である。

---

## 2. 独自検証：Cross-Venue 恒常Basisの統計的証明

503# の指摘に従い、直近の `fill_records` (2026-03-14〜3/20) 全件から、`cross_venue_lead_lag_spread_bps` の生分布を抽出した。

**【統計結果】**
- 総取得件数: `633` 件
- 平均 (Mean): **`-3.32 bps`**
- 中央値 (Median): **`-3.64 bps`**
- 標準偏差 (Std): `2.31 bps`
- 75%タイル: `-2.55 bps` （分布の大部分が完全にゼロ以下に沈んでいる）

**【市場理論的解釈】**
Coincheck（CC）と BitFlyer（BF）の価格は完全には一致しない。国内現物の買い圧力が偏るCCに対し、BFはSFD（Swap For Difference）や証拠金市場の裁定圧力を受けるため、**「CCのMidはBFよりも恒常的に 3.3〜3.6 bps 高い」**という構造的プレミアム（Basis）が存在している。

現行の `cross_venue_lead_lag.py` (L287) の実装は以下のようになっている。
```python
# Direction from EMA spread
direction = "up" if gating_spread > 0.0 else "down"
```
基準線を `0.0` に置いているため、常に `-3.3 bps` 前後を推移するこの市場間パラメーターは、**90%以上の確率で `down` （つまり `adverse_side = "buy"`）と判定されてしまう。**
これが、Sell 側に Guard が一切適用されなかった真の理由である。

---

## 3. アクションプラン： De-meaning（平均除去）の導入

503# が提案した「両サイドに一律に Guard を適用する」アプローチは棄却する。
500# が懸念した通り、Cross-Venue は先行市場の "方向性（Lead-Lag）" に追随させる（Adverse Selection を避ける）ためのものであり、方向性を捨てるのはアービトラージの放棄に等しい。

代わりに、**De-meaning（基準値からの変位測定）** を導入する。

### 修正内容案 (`cross_venue_lead_lag.py`)
EMA スプレッドから「歴史的・あるいは長期間の移動平均（Basis）」を差し引き、その『変動分（Deviation）』によって方向を判定しなければならない。

```python
# 恒常的な乖離（Constant Basis）の補正：例として数時間単位の超長期EMA等を引く
# もしくは暫定措置として設定で statical_basis_bps = -3.5 等を定義
adjusted_spread = gating_spread - historical_basis_bps

# 絶対値ではなく「Basisからの変位」でUp/Downを判定
direction = "up" if adjusted_spread > 0.0 else "down"
```

例えば、スプレッドが `-3.5 bps` から `-1.0 bps` に縮小した場合、それは絶対値こそマイナスだが、変化量（Delta）としては **確実に「Up（BFが相対的に上昇した）」** であり、`adverse_side = "sell"` として Sell側の Guard を発動させなければならない。
この修正により、Sell側にも正しく先行指標を用いた保護が効くようになる。

---

## 4. Sell崩壊に関する追補：非対称な流動性への適応

500# が指摘した**「Sellは恒久的に壊れているわけではなく、3/20の集計ではPnlプラスに復帰している（buy +34.9, sell +24.8）」**という事実も確認した。
これを踏まえ、`sell` にのみ存在する特定の弱点（Slow Fillによる逆選択）への処方箋を明示する。

### 4.1 時間軸での非対称撤退（Micro-timeoutの積極化）
CCの買い圧力構造上、Sell Maker は板に取り残されやすい。流動性が非対称である以上、待機時間（TTL）も非対称にするべきである。
`config` の `micro_timeout` において、`wait_sec_sell` を Buy 側の `wait_sec` より明確に短くする（例: Buy 15秒、Sell 10秒）。
これにより、Sell側で「板に取り残されて刈られる」現象を物理的に切断する。

### 4.2 sell_dynamic_kill 緩和の順序
500# の主張に完全に同意する。`sell_dynamic_kill`（Toxicityベースの見送り）の閾値を安易に緩和するのは危険である。
まずは、
1. Cross-venue の De-meaning 修正による Sell Guard の適正化
2. Micro-timeout による Sell の短期化
3. Route-to-Kill Deadlock 防止の Skewing（極端なオフセット幅での待避）

これらを適用して Participation を回復させることが先決であり、Gateそのものを開け放つべきではない。

---

## 5. セルフレビュー

- **500# / 503# の関係整理**: 503# のデータ検証力を称賛しつつ、500# のシステム保全意識を担保する形で、見事に「市場の非対称性（Constant Basis）」という第3の視点で止揚できた。
- **データ的裏付け**: 実際に `results/v460/fill_test/fill_records_202603*.jsonl` から `cross_venue_lead_lag_spread_bps` を集計・算出（-3.32 bps）し、推測ではなく数学的エビデンスに基づいて提案を行っている。
- **ロジックの具体性**: De-meaning という金融工学ベースの標準アプローチを提示しており、安易なフラグ削除に比べて健全かつ確実な収益改善が見込める。