# 300# A/B テスト深堀り — システム工学 × 市場微視構造理論からの多角的考察

> **文書番号**: 300#  
> **種別**: `rev` (レビュー/考察)  
> **作成日**: 2026-03-06  
> **前提**: [299_ph2_rpt_ab_test_f4_validation.md](299_ph2_rpt_ab_test_f4_validation.md)  
> **外部レビュー前提**: 本文書は Gemini / Codex / GPT 等による第三者 AI レビューを想定し、
> 反論・補足・代替仮説の提示を求める構成とする。

---

## §0 エグゼクティブ・サマリー

299# A/B テスト (6,952 レコード / 22 日間) は「sell vs buy に統計的有意差なし」という
帰無仮説不棄却の結論を出した。本文書はその結果を**鵜呑みにせず**、
システム工学と市場微視構造理論の二重レンズで解体する。

**核心的発見**:
1. 「有意差なし」は「問題なし」ではない — 両サイドとも downside_p10 が基準超過
2. レジーム条件付き分析では sell の構造的劣位が明確だが、サンプル不足で検出力不足
3. 現行システムに **5 つの構造的矛盾** が存在し、収益性改善のボトルネックとなっている
4. 「約定しない注文」と「約定した注文」で異なるメカニズムが作用しており、
   filled-only 分析は **生存者バイアス** を含む

---

## §1 市場微視構造理論からの分析

### 1.1 Adverse Selection の非対称性 — Glosten-Milgrom (1985) の実証

| レジーム | buy AS率 | sell AS率 | 差分 | 含意 |
|---|---|---|---|---|
| overall | 28.0% | 30.4% | +2.4pp | 軽微 |
| ranging | 24.8% | 27.5% | +2.7pp | 軽微 |
| trending | 27.9% | 27.9% | ±0.0pp | 対称 |
| trending_down | **24.7%** | **34.2%** | **+9.5pp** | sell が顕著に逆選択 |
| trending_up | 28.0% | **41.2%** | **+13.2pp** | sell が壊滅的に逆選択 |
| none (warm-up) | 43.2% | 42.2% | −1.0pp | 両方高い → レジーム未確定期 |

**理論的解釈 (Glosten-Milgrom 1985)**:

Glosten-Milgrom モデルでは、マーケットメイカーの期待損失は
$E[AS] = \alpha \cdot \mu \cdot \sigma$ で近似される。
ここで $\alpha$ = 情報取引者の割合、$\mu$ = 情報優位の度合い、$\sigma$ = 価格変動性。

- **trending_up で sell する** = 上昇トレンド中に sell 指値を板に載せる行為
- 約定が成立するのは「価格が下方に突き抜ける」瞬間のみ
- 上昇トレンド中の急落は、大量の informed sell flow (トレンド終了の先行者) に起因
- → 約定は高確率で **toxic flow** (情報優位者の反対ポジション解消) によって起こる
- **fill_rate 18.1% + AS率 41.2%** は「約定 ≒ 逆選択」を意味する

### 1.2 As-Loss の深度分析 — 「AS を食らったときにどれだけ痛いか」

| レジーム | buy avg_AS_loss | sell avg_AS_loss | sell/buy 比 |
|---|---|---|---|
| ranging | −5.98 bps | −6.80 bps | 1.14× |
| trending | −6.34 bps | −6.97 bps | 1.10× |
| trending_down | −7.73 bps | −7.08 bps | 0.92× |
| trending_up | −6.36 bps | **−8.18 bps** | **1.29×** |

**発見**: AS 発生率だけでなく、**AS 1 回あたりの損失深度も sell が大きい** (trending_up)。
これは Kyle (1985) の「価格インパクトは情報量に比例する」モデルと整合する。
上昇トレンド中の急落は情報量が大きい (レジーム転換の兆候) ため、
食らった際の深度が深くなる。

trending_down buy の avg_AS_loss = −7.73 bps が全セル中最大なのも注目に値する。
下降トレンド中の buy AS は「落ちるナイフを掴む」パターンであり、
AS 発生率は低い (24.7%) が、食らった際の傷が深い。

### 1.3 VG 発動率 vs テールリスクの非整合

VolatilityGuard (VG) は高ボラティリティ時に offset を拡大して逆選択を防御する機構だが：

| レジーム Side | VG 発動率 | p10 | VG 発動後も p10 悪化? |
|---|---|---|---|
| ranging buy | 19.1% | −5.36 | — (baseline) |
| ranging sell | 18.7% | −6.55 | — (baseline) |
| trending_down buy | **26.0%** | −5.79 | やや悪化 |
| trending_down sell | **36.7%** | **−7.41** | ❌ VG 不十分 |
| trending_up buy | **38.7%** | −5.72 | VG が効いている |
| trending_up sell | **37.5%** | **−9.93** | ❌❌ VG 全く不十分 |

**構造的矛盾 #1: VG はフィルタ率を高めるが、通過した約定の毒性は緩和しない**

VG が 37.5% 発動しているにもかかわらず p10 = −9.93 bps ということは：
- VG が防いだ約定は「そこそこ悪い」レベルだった
- VG をすり抜けた約定は「壊滅的」だった
- → **VG は量のフィルタであり、質のフィルタではない**

Avellaneda-Stoikov (2008) の最適スプレッド理論では、
$\delta^* = \gamma \sigma^2 \tau + \frac{2}{\gamma} \ln(1 + \gamma/k)$
ここで $\gamma$ = リスク回避度、$\sigma$ = ボラティリティ、$\tau$ = 残り時間。
VG は $\sigma$ の増加に対して $\delta$ を拡大するが、
**$\gamma$ (リスク回避度) をレジーム条件付きで動的調整していない**。

### 1.4 Fill Rate と Spread 分布の非対称性 — 約定選択バイアス

**生存者バイアス仮説**: A/B テストは **filled orders のみ** を比較している。

| レジーム | sell n_total | sell n_filled | **fillされなかった sell** |
|---|---|---|---|
| trending_up | 442 | 80 | **362 (81.9%)** |
| trending | 360 | 118 | 242 (67.2%) |
| trending_down | 196 | 79 | 117 (59.7%) |
| ranging | 1,690 | 785 | 905 (53.6%) |

trending_up sell で 442 回試行して 80 回しか約定しないのは、
大半の sell 指値が「mid に到達しない安全な位置」に置かれている（VG/offset 拡大により）
ことを意味する。約定する 18.1% は、mid が急落して指値を貫通した場合に限られる。

**これは Glosten (1994) の「市場崩壊限界」理論と一致**する。
thin な板で指値が遠い位置に集中すると、わずかな order flow で価格が
指値まで到達し、到達した瞬間にさらなる売りフローが発生する（レジーム崩壊）。

→ **「約定しないことが多い」は安全ではなく「約定するときは常に最悪」を意味する**

### 1.5 Queue Priority と Cancel-Replace の力学

293# BS-1 が既に指摘した cancel-before-deadband-check の問題に加え、
reprice_rate の分析が構造的示唆を与える：

| レジーム Side | reprice率 | avg_reprice_drift (bps) |
|---|---|---|
| ranging buy | 8.9% | 5.94 |
| ranging sell | 8.3% | 6.74 |
| trending_up buy | 5.4% | 10.62 |
| trending_up sell | **1.3%** | 10.47 |

trending_up sell の reprice 率 1.3% は「ほぼ reprice しない」ことを示す。
これは SkipGate / VG が reprice 前にサイクルを遮断しているためであり、
reprice 自体が問題ではない。**問題は初回発注時の価格決定にある**。

---

## §2 システム工学からの分析

### 2.1 Offset パイプラインの単調拡大バイアス

MakerPriceCalculator.compute() パイプライン (10 ステージ):

```
① base_offset_ratio (side 別)
  ↓ ② inventory_skewing (trending 時は disabled)
  ↓ ③ sell_offset_floor (sell のみ)
  ↓ ④ _apply_as_reservation_shift (Avellaneda-Stoikov)
  ↓ ⑤ _apply_regime_boosts (trending_down → boost)
  ↓ ⑥ _apply_spread_adaptive
  ↓ ⑦ _apply_kyle_lambda (Kyle 価格インパクト)
  ↓ ⑧ _apply_amihud_illiq (Amihud 非流動性)
  ↓ ⑨ _apply_volatility_guard (VG)
  ↓ ⑩ _apply_imbalance_risk (板不均衡)
  ↓ ⑪ _apply_buy_as_guard (283# P1-6)
  ↓ ⑫ _apply_loss_boost (損失後拡大)
  ↓ ⑬ _apply_ffd_boost (Fast Fill Defense)
```

**構造的矛盾 #2: 全ステージが offset を拡大する方向にのみ作用**

- `min_offset_ratio` はフロアとして存在するが、**天井 (max_offset_ratio) がない**
- 各ステージの拡大は加算的/乗算的に累積する
- trending_up sell で ④⑤⑦⑧⑨⑩ が同時発動する可能性がある
- → offset が過度に拡大 → 約定は mid 急落時のみ → **toxic fill only trap**

**代替仮説（レビュアーへの問い）**: offset の過剰拡大は「損失を防ぐ」のか、
「良い約定機会も消して、残った約定を毒性のみにする」のか？
後者が正しければ、**offset 天井の導入** が必要。

### 2.2 Inventory Skewing のレジーム門番 — 論理矛盾

```python
# 249# regime gate: trending 時は inv_skew を無効化
if _r.is_trending:
    _inv_skew_regime_blocked = True
```

**目的**: トレンド方向のポジション蓄積を inv_skew が阻害しないため（コメント記載）。
**問題**: trending_up で sell を実行するケースでは inv_skew が sell offset を
狭めてくれる可能性があるのに、それが無効化されている。

**構造的矛盾 #3**: inv_skew_regime_gate は「トレンド方向のアルファ保護」を意図しているが、
「トレンド逆方向の自殺行為」に対する防御も同時に無効化している。

**修正案**: `inv_skew_regime_gate` をトレンド方向依存にする：
- trending_up → buy の inv_skew は無効 (alpha 保護)、sell の inv_skew は有効 (AS 防御)
- trending_down → sell の inv_skew は無効 (alpha 保護)、buy の inv_skew は有効 (AS 防御)

### 2.3 Side Selector のレジーム不感症

SideSelector は以下のロジックで side を決定する：
1. 基本: buy/sell 交互
2. Smart Side: imbalance_threshold 超過時に suppress/follow
3. rapid_exit: Early Exit 後の反対 side 強制
4. frozen_side: 残高不足時の凍結

**構造的矛盾 #4**: Smart Side は **板不均衡 (orderbook imbalance)** を入力とするが、
**レジーム情報を参照しない**。

trending_up レジームで板不均衡が中立 (|imbalance| < threshold) の場合、
alternation ルールにより sell が選択される。しかし §1.1 で示した通り、
trending_up sell は構造的に非実行的 (fill 18.1%, AS 41.2%) である。

**着目点**: trending_up sell の n_total = 442 に対し buy の n_total = 171。
sell の試行回数が buy の **2.6 倍** あるのは、トレンド方向 (up) で
buy が約定しやすく (fill 54.4%)、サイクルが短くなる一方、
sell は約定せずタイムアウトしてサイクルが長引くためと推定される。

**しかし**: n_total の比率は sell:buy = 2.6:1 だが、
filled の比率は sell:buy = 80:93 = 0.86:1。
つまり **売りは買いの 2.6 倍試行して、約定数は少ない**。
計算リソースと API 負荷の浪費。

### 2.4 Daily Drawdown Guard の side 不分離

daily_drawdown_guard は side 引数を受け取る (コード確認済) が、
soft lot reduction は **全体** に適用される（side 別のロット縮小ではない）。

trending_up sell が連続損失を出した場合：
1. DD guard が soft lot reduction を発動
2. **buy のロットまで縮小される** → trending_up buy の正の収益機会も減殺
3. 結果: 「sell の損失を抑えるために buy の利益も削る」

**構造的矛盾 #5**: DD guard のロット縮小は side-agnostic だが、
損失源は regime × side 条件付き → 無差別縮小は本来の意図と乖離。

### 2.5 "none" レジームの隠れた出血

A/B テストは `exclude_regimes=["none"]` で none を除外しているが、
実際のシステムは none レジームでも執行を続けている：

| Side | none n_total | none n_filled | fill_rate | AS率 | pnl30 |
|---|---|---|---|---|---|
| buy | 413 | 139 | 33.7% | **43.2%** | −0.15 bps |
| sell | 915 | 128 | **14.0%** | **42.2%** | **−0.80 bps** |

none レジームは **全レジーム中 AS 率が最悪** (43% / 42%)。
これは warm-up 期間中の未安定なレジーム検知により、
offset 調整が機能しないまま発注が行われていることを示す。

sell n_total = 915 は none レジーム中で最大 → レジーム未確定時に
大量の sell 試行が行われている。fill_rate 14.0% は trending_up sell (18.1%) よりさらに悪い。

---

## §3 統計的手法の批判的検討

### 3.1 検出力 (Statistical Power) の不足

F-4 ノンパラメトリック検定の結論「有意差なし」は、**帰無仮説を積極的に支持するものではない**。
「差がないことを証明した」のではなく「差を検出するには証拠が足りなかった」可能性がある。

Post-hoc 検出力分析:

trending_up では n_variant = 80, n_control = 93。Cohen's d = −0.152 を
有意水準 α = 0.05 で検出するために必要なサンプル数は：
$$n \approx \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot 2}{d^2} = \frac{(1.96 + 0.84)^2 \cdot 2}{0.152^2} \approx 680$$
per group — 現在の n ≈ 85 の **8 倍** 必要。

→ **trending_up の検定は検出力 ≈ 12% しかない**。
Type II error (偽陰性) のリスクが極めて高い。

### 3.2 多重比較の階層構造

現在の Holm-Bonferroni は k = 2 (Welch + MW) の補正のみ。
しかし実際には 5 レベル (overall + 4 regimes) × 2 検定 = 10 の検定を
暗黙的に実施している。

レジーム横断で同時検定する場合、familywise error rate (FWER) は：
$$FWER = 1 - (1 - 0.05)^{10} \approx 0.40$$
現在の Holm は各レジーム内のみで補正 → 全体としては補正不十分。

**提案**: BH (Benjamini-Hochberg) 法による FDR 制御を全 10 検定に適用する。
FWER よりも検出力を維持しつつ偽発見率を制御できる。

### 3.3 条件付き分布の非独立性

A/B テストは sell vs buy を独立標本として扱っているが、
**同一市場環境で交互に実行されるため、時系列自己相関が存在する**。

連続する buy → sell サイクルでは：
- 同一レジーム内で実行される確率が高い
- 市場の方向性が共有される (同じ VPIN / volatility 環境)
- buy の約定が sell の発注条件に影響する (inventory skewing)

→ 独立標本の仮定は厳密には成立しない。
**提案**: Mann-Whitney U に代えて Wilcoxon signed-rank test (paired) を
時間的に隣接する buy-sell ペアに適用する、あるいは
クラスターロバスト標準誤差で独立性仮定の違反に対処する。

---

## §4 自己批判と代替仮説

### 4.1 本分析の限界

1. **量的分析の不在**: offset パイプラインの各ステージが実際に何 bps 寄与しているかの
   定量的分解ができていない。`ev_offset_mult_applied` の分布分析が必要。

2. **反実仮想の曖昧さ**: 「trending_up sell をスキップしたら」の定量的反事実は
   trending_eval の CF gain (+0.25 bps for trending_down) しかなく、
   trending_up sell の CF gain は未計算。

3. **regime 遷移の時間構造**: レジーム判定にはヒステリシス (連続 N サイクル一致) があり、
   **レジーム遷移直後の fill records** が実際にはどのレジームの市場環境で
   実行されたのかが曖昧。遷移期の misclassification による汚染がありうる。

4. **外的要因の未統制**: 22 日間の BTC/JPY 市場環境は非定常。
   特定の日（例: 0228 の p10 = −14.30）が outlier として全体を歪めている可能性がある。

### 4.2 代替仮説（レビュアーへの問い）

**H1 (Offset 過剰仮説)**: 現行の offset 累積パイプラインが sell 側で過剰に拡大し、
「約定 = toxic flow のみ」の選択バイアスを生んでいる。offset 天井を設けることで
fill_rate が上昇し、毒性の低い約定が混ざることで平均 PnL が改善する。

**H2 (レジーム遅延仮説)**: RegimeDetector のヒステリシスによりレジーム判定が遅延し、
trending_up 初期に ranging パラメータで sell が実行される。この期間の sell が AS を
大量に食らい、trending_up 統計を悪化させている。

**H3 (市場構造仮説)**: Coincheck BTC/JPY の板構造が非対称であり、
buy 側の流動性が sell 側より構造的に厚い。これにより sell の約定は
常に thin side からの supply 枯渇を伴い、テールリスクが大きくなる。

**H4 (Smart Side 失敗仮説)**: Smart Side の imbalance_threshold がレジーム非依存であるため、
trending_up で sell を抑制する感度が不足。trending_up での sell 試行 442 回は
本来 100 回以下に抑制されるべきだった。

---

## §5 アクション優先度の再評価

299# §6.3 の推奨を、本分析に基づいて再構造化する。

### Tier 0: 計測インフラ（修正なくして改善判定なし）

| ID | アクション | 根拠 |
|---|---|---|
| T0-1 | offset パイプライン各ステージの寄与量を FillRecord に記録 | §2.1: 定量評価なしに最適化不可 |
| T0-2 | none レジームの執行を conditional halt (一定期間 skip) | §2.5: AS 43% は許容外 |
| T0-3 | BH (FDR) 法の全レジーム横断検定への適用 | §3.2: FWER 40% は検定として無効 |

### Tier 1: 構造的欠陥の修正（直接的な出血止め）

| ID | アクション | 根拠 | 構造的矛盾 |
|---|---|---|---|
| T1-1 | trending_up sell のレジーム条件付きスキップ | §1.1: AS 41.2%, fill 18.1% | — |
| T1-2 | inv_skew_regime_gate をトレンド方向依存にする | §2.2: 反方向の防御も無効化中 | #3 |
| T1-3 | offset パイプラインに max_offset_ratio 天井を追加 | §2.1: 単調拡大の構造的欠陥 | #2 |
| T1-4 | DD guard の lot 縮小を side 別に分離 | §2.4: sell 損失で buy 利益も削減 | #5 |

### Tier 2: 精度改善（構造修正後のファインチューニング）

| ID | アクション | 根拠 |
|---|---|---|
| T2-1 | VG のレジーム条件付き $\gamma$ (リスク回避度) 動的調整 | §1.3: VG は量だが質のフィルタにならず |
| T2-2 | Smart Side にレジーム入力を追加 | §2.3: 板不均衡のみで trending_up sell を抑制できず |
| T2-3 | Wilcoxon signed-rank test for paired buy-sell cycles | §3.3: 独立標本仮定の違反 |
| T2-4 | ev_offset sensitivity を side 別に分離 (BS-2) | 293# : buy/sell で最適感度が異なるはず |

### Tier 3: 中長期的改善

| ID | アクション | 根拠 |
|---|---|---|
| T3-1 | レジーム遷移期の transition buffer (§4.2 H2 検証) | ヒステリシス遅延の定量化 |
| T3-2 | Coincheck 板構造の bid/ask 非対称性定量分析 (H3 検証) | 構造的な sell 不利が存在するか |
| T3-3 | サンプル蓄積後の trending_up 再検定 (n ≥ 680 目標) | §3.1: 現在の検出力 12% → 80% |

---

## §6 レビュアーへの質問事項

本文書の結論に対し、以下の観点での反論・補完を求める:

### Q1: Offset 天井 (max_offset_ratio) の導入リスク
- 天井を設けると ranging レジームでの AS 防御が弱化するか？
- trending vs ranging で異なる天井を設けるべきか？
- 天井の最適値はどのように決定すべきか (バックテスト? ベイズ最適化?)

### Q2: レジーム条件付き sell スキップの副作用
- trending_up で sell を全面スキップすると、在庫バランスが崩れるか？
- 代替案: スキップではなくスプレッド大幅拡大 (e.g. offset ×3) は妥当か？
- レジーム判定の精度 (ヒステリシス遅延) を考慮した場合のリスクは？

### Q3: 統計手法の妥当性
- paired test (Wilcoxon) と unpaired test (Mann-Whitney) のどちらが本ケースに適切か？
- BH 法の FDR 制御水準は q = 0.05 で十分か、それとも q = 0.10 が適切か？
- ブートストラップ信頼区間による補完は有用か？

### Q4: 短期収益性 vs 長期安定性のトレードオフ
- trending_up sell スキップは短期的に PnL を改善するが、
  上昇トレンド終了時 (反転) の sell 機会を逃すリスクをどう評価するか？
- 「高収益性システム」の大義に照らし、テールリスク許容度をどこに設定すべきか？

### Q5: 本分析で見落としている盲点
- 市場微視構造 / システム工学の両面で、本分析が検討していない重要な観点は何か？
- 特に、Coincheck 固有の取引所特性（手数料構造、API 制約、板の更新頻度）に
  起因する問題は本分析で十分に考慮されているか？

---

## §7 データ付録

### 7.1 全レジーム × Side 完全メトリクス

| Regime | Side | n_total | n_filled | fill% | pnl30 | std | p10 | p05 | prof% | AS% | AS_loss | repr% | repr_drift | VG% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | buy | 413 | 139 | 33.7 | −0.15 | 5.17 | −4.63 | −9.69 | 48.9 | 43.2 | −3.89 | 0.0 | 0.0 | 0.0 |
| none | sell | 915 | 128 | 14.0 | −0.80 | 5.99 | −5.86 | −11.19 | 48.4 | 42.2 | −5.22 | 0.0 | 0.0 | 0.0 |
| ranging | buy | 2169 | 785 | 36.2 | −0.47 | 4.93 | −5.36 | −7.11 | 46.8 | 24.8 | −5.98 | 8.9 | 5.94 | 19.1 |
| ranging | sell | 1690 | 785 | 46.5 | −0.14 | 6.61 | −6.55 | −10.16 | 44.6 | 27.5 | −6.80 | 8.3 | 6.74 | 18.7 |
| trending | buy | 161 | 118 | 73.3 | +0.57 | 6.62 | −6.96 | −7.67 | 51.7 | 28.0 | −6.34 | 11.0 | 0.0 | 0.0 |
| trending | sell | 360 | 118 | 32.8 | −0.66 | 5.70 | −6.56 | −7.68 | 44.9 | 28.0 | −6.97 | 7.6 | 0.0 | 0.0 |
| t_down | buy | 191 | 77 | 40.3 | +0.92 | 8.69 | −5.79 | −7.33 | 53.3 | 24.7 | −7.73 | 11.7 | 5.34 | 26.0 |
| t_down | sell | 196 | 79 | 40.3 | −0.41 | 6.76 | −7.41 | −8.67 | 43.0 | 34.2 | −7.08 | 5.1 | 10.61 | 36.7 |
| t_up | buy | 171 | 93 | 54.4 | −0.33 | 5.57 | −5.72 | −7.55 | 46.2 | 28.0 | −6.36 | 5.4 | 10.62 | 38.7 |
| t_up | sell | 442 | 80 | 18.1 | −1.36 | 7.69 | −9.93 | −10.46 | 42.5 | 41.2 | −8.18 | 1.3 | 10.47 | 37.5 |
| unknown | buy | 58 | 47 | 81.0 | −1.38 | 3.75 | −5.61 | −7.49 | 31.9 | 38.3 | −5.03 | 2.1 | 0.0 | 0.0 |
| unknown | sell | 58 | 46 | 79.3 | −0.39 | 4.27 | −3.94 | −6.32 | 47.8 | 26.1 | −5.02 | 2.2 | 0.0 | 0.0 |

### 7.2 F-4 検定結果サマリ (全レベル)

| Level | Welch p | MW p | Cohen's d | Cliff's δ | δ 解釈 | Holm W | Holm MW |
|---|---|---|---|---|---|---|---|
| overall | 0.893 | 0.491 | −0.006 | 0.017 | negligible | ✗ | ✗ |
| ranging | 0.287 | 0.814 | 0.054 | −0.007 | negligible | ✗ | ✗ |
| trending | 0.129 | 0.236 | −0.199 | 0.089 | negligible | ✗ | ✗ |
| t_down | 0.290 | 0.124 | −0.170 | 0.143 | negligible | ✗ | ✗ |
| t_up | 0.327 | 0.382 | −0.152 | 0.077 | negligible | ✗ | ✗ |

### 7.3 メタ情報

| 項目 | 値 |
|---|---|
| 分析者 | Copilot (Claude Opus 4.6) |
| 参照理論 | Glosten-Milgrom (1985), Kyle (1985), Glosten (1994), Avellaneda-Stoikov (2008), Hamilton (1989), Lo (2004) |
| 分析対象コード | maker_price.py, side_selector.py, fill_cycle_executor.py, order_monitor.py, regime_detector.py, fill_quality.py |
| 前提文書 | 291# (Gemini review), 293# (BS analysis), 298# (F-4 impl), 299# (AB test report) |
