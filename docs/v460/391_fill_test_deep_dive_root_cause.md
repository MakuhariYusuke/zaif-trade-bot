# 391# Fill Test Deep Dive — 根本原因分析

> **対象期間**: 2026-03-09 ~ 2026-03-12 (4 日間)
> **総レコード数**: 1,270件 / **約定数**: 411件 (fill rate 32.4%)
> **累計 30s PnL**: +67.7 bps (buy: +28.3, sell: +39.3)
> **作成日**: 2026-03-12
> **前提**: 381# YAML 変更 (sell_hour_offset_boost 拡張, offset_ceiling_ratio_buy 0.15→0.20) 適用済

---

## 1. エグゼクティブサマリー

4 日間の Fill Test は全体としてプラスだが、**sell サイドに構造的な tail risk** が存在する。
TOP 15 最大損失は **全て sell 約定**であり、sell tail loss (−483 bps) は buy (−178 bps) の **2.7 倍**。
この非対称性は複数の構造的要因が複合して発生している。

### 最重要発見

| # | root cause | 影響度 | 修正容易性 |
|---|-----------|--------|-----------|
| RC-1 | Sell tail risk 非対称 (buy の 2.7x) | **CRITICAL** | 中 |
| RC-2 | Offset pipeline 59% が ceiling 0.30 に張り付き | **HIGH** | 易 |
| RC-3 | EV model の "confident-wrong" ゾーン (EV [1,2) sell WR=29%) | **HIGH** | 中 |
| RC-4 | 高頻度約定 (gap<5min) が net negative (−0.43 bps) | **MEDIUM** | 易 |
| RC-5 | Regime confidence [0.7,0.9) が paradoxically worst | **MEDIUM** | 中 |
| RC-6 | Wide spread (3.0-5.0 bps) での約定が kill zone | **MEDIUM** | 易 |
| RC-7 | 381# offset_ceiling_ratio_buy 0.20 が buy 劣化の可能性 | **LOW** | 易 |

---

## 2. データ全体像

### 2.1 日次サマリー

| 日付 | 約定数 | 累計 30s | 平均 30s | WR | big loss (>5bps) |
|------|--------|---------|---------|-----|-----------------|
| 03/09 | 188 | +32.3 | +0.172 | 49% | 33 |
| 03/10 | 51 | +21.8 | +0.427 | 57% | 10 |
| 03/11 | 28 | +13.5 | +0.483 | 54% | 5 |
| 03/12 | 144 | +0.1 | +0.001 | 49% | 18 |

### 2.2 サイド別日次内訳

| 日付 | buy n | buy 累計 | buy 平均 | sell n | sell 累計 | sell 平均 |
|------|-------|---------|---------|--------|----------|----------|
| 03/09 | 91 | +18.7 | +0.206 | 97 | +13.5 | +0.140 |
| 03/10 | 26 | +50.1 | +1.927 | 25 | −28.3 | −1.133 |
| 03/11 | 15 | +2.4 | +0.157 | 13 | +11.2 | +0.859 |
| 03/12 | 72 | −42.9 | −0.595 | 72 | +42.9 | +0.596 |

**注目**: 03/12 は buy/sell が完全に反転 — buy が −42.9 bps に転落。381# で ceiling_buy を 0.15→0.20 に変更後の影響の可能性あり。

---

## 3. Root Cause 分析

### RC-1: Sell Tail Risk 非対称 (CRITICAL)

**現象**: sell side の tail loss が buy の 2.7 倍

| 指標 | buy | sell |
|------|-----|------|
| worst single fill | −13.3 bps | −41.8 bps |
| 3rd worst | −10.0 bps | −19.8 bps |
| tail loss (< −5 bps) 累計 | −178.3 bps | −483.3 bps |
| tail win (> +5 bps) 累計 | +181.8 bps | +516.4 bps |

sell の tail は win/loss 両方とも大きいが、loss の方が win より大きい(**非対称分布**)。

**TOP 15 損失 (全て sell)**:

| 時刻(JST) | EV | regime | conf | spread | macro | 30s PnL |
|-----------|-----|--------|------|--------|-------|---------|
| 03/10 11:29 | +0.71 | ranging | 0.49 | 3.35 | neutral | −41.76 |
| 03/09 21:36 | −0.71 | ranging | 0.52 | 1.26 | weak_up | −35.65 |
| 03/12 22:47 | −3.59 | ranging | 0.89 | 2.50 | neutral | −19.81 |
| 03/12 03:19 | +1.47 | ranging | 0.79 | 1.87 | neutral | −18.80 |
| 03/09 17:20 | −0.35 | ranging | 0.78 | 2.54 | neutral | −17.88 |
| 03/11 02:30 | −2.55 | ranging | 0.77 | 3.57 | insufficient | −17.72 |
| 03/11 04:13 | +0.91 | ranging | 0.75 | strong_down | −15.97 |
| 03/11 04:21 | −1.65 | ranging | 0.87 | weak_down | −15.56 |
| 03/09 15:16 | +0.32 | ranging | 0.74 | 2.83 | weak_down | −15.06 |
| 03/10 05:00 | +2.47 | ranging | 0.67 | 1.07 | neutral | −15.05 |

**根本メカニズム**: sell 約定後に価格が急上昇 (adverse selection)。sell order の ask 板に乗せた注文が、**informed buyer の sweep** により約定 → 30s 後に価格上昇。

**sell 30s PnL パーセンタイル**:
- P01: −19.81 / P05: −14.73 / P10: −9.31
- P25: −3.59 / P50: +0.14 / P75: +4.14
- P90: +11.07 / P95: +14.73 / P99: +21.36

P01 (−19.8) 対 P99 (+21.4) — ほぼ対称だが、extreme tail で非対称性が発生。

---

### RC-2: Offset Pipeline Capping 問題 (HIGH)

**現象**: 全約定の **59% で final offset が 0.30 に clamp** されている。

#### Offset Pipeline ステージ平均値:

| stage | mean | min | max |
|-------|------|-----|-----|
| base | 0.1155 | 0.0500 | 0.1800 |
| as_shift | 0.1923 | 0.0461 | 0.3000 |
| regime | 0.1807 | 0.0328 | 0.5400 |
| spread_adapt | 0.2086 | 0.0204 | 0.5400 |
| kyle | 0.2057 | 0.0204 | 0.3000 |
| amihud | 0.2061 | 0.0204 | 0.3000 |
| vol_guard | 0.2330 | 0.0350 | 0.3000 |
| imb_risk | 0.2330 | 0.0350 | 0.3000 |
| buy_as_guard | 0.2363 | 0.0350 | 0.4500 |
| sell_hour | 0.2363 | 0.0350 | 0.4500 |
| loss_boost | 0.2366 | 0.0350 | 0.4500 |
| ffd | 0.2413 | 0.0350 | 0.4500 |
| **final** | **0.2413** | 0.0350 | 0.4500 |

60/411 fills (15%) では **base 以降の全ステージが同一値** — パイプラインの多段構成が無効化。

#### 現在の ceiling 設定:
- `offset_ceiling_ratio`: 0.15 (共通デフォルト)
- `offset_ceiling_ratio_buy`: **0.20** (381# で 0.15→0.20)
- `offset_ceiling_ratio_sell`: **0.50** (320# で 0.15→0.50)

**問題**: sell の ceiling は 0.50 だが、sell floor が 0.30 なので実効レンジは **[0.30, 0.50]** の狭い帯域。
buy は **[min, 0.20]** — 381# の変更で上限拡大したが、floor がないためほぼ全て 0.20 以下に分布。

#### サイド別 effective offset:

| side | mean | median | P10 | P90 |
|------|------|--------|-----|-----|
| buy | 0.1702 | 0.1553 | 0.0798 | 0.2495 |
| sell | 0.3444 | 0.2996 | 0.2506 | 0.5300 |

sell は buy の **2.0 倍**のオフセットだが、それでも tail loss は 2.7 倍。
→ **sell のオフセットはまだ不十分**、もしくはオフセット以外の防御が必要。

---

### RC-3: EV Model "Confident-Wrong" ゾーン (HIGH)

#### EV バケット別精度 (全約定):

| EV bucket | n | 30s mean | WR |
|-----------|---|----------|-----|
| [−10,−3) | 26 | +1.09 | 54% |
| [−3,−2) | 26 | −0.19 | 46% |
| [−2,−1) | 33 | −0.16 | 48% |
| [−1,0) | 126 | +0.47 | 56% |
| [0,1) | 134 | +0.04 | 49% |
| **[1,2)** | **45** | **−0.64** | **40%** |
| [2,10) | 21 | +0.62 | 57% |

**EV [1,2) は WR=40% で最悪** — 中程度に confident な予測が一番外れる。

#### EV [1,2) サイド内訳:
- buy: n=28, 30s=+0.158, WR=46% (やや悪い)
- **sell: n=17, 30s=−1.945, WR=29%** (致命的)

#### EV 方向性の意味:
- EV > 0 → 「価格が上がる」予測
- buy + EV>0 = **正しい alignment** (上がるなら買い有利)
- **sell + EV>0 = 逆方向** (上がるのに売り)

| side | EV>0 | 30s | WR | EV<0 | 30s | WR |
|------|------|-----|-----|------|-----|-----|
| buy | 124 (61%) | +0.213 | 48% | 80 (39%) | +0.024 | 52% |
| sell | 76 (37%) | **−0.481** | 46% | 131 (63%) | +0.579 | 53% |

**重大発見**: sell + EV>0 の 76 件は −0.481 bps/fill。EV が「上がる」と予測しているのに sell 注文を出しているケースが、sell 損失の主要因。

**根本仮説**: EV model は価格方向の予測をしているが、**発注サイドの決定に EV が十分に反映されていない**。EV がプラスなのに sell 注文を出す — これは maker として板の片側に座る際に「方向リスクを無視」していることを意味する。

---

### RC-4: 高頻度約定の Net Negative (MEDIUM)

#### Inter-fill gap vs PnL:

| gap | n | 30s mean |
|-----|---|----------|
| [0,5) min | 203 | **−0.427** |
| [5,10) min | 131 | +0.374 |
| [10,20) min | 50 | +0.461 |
| [20,40) min | 19 | +3.446 |
| [40,120) min | 4 | +1.434 |

約定の **49% が 5 分未満の間隔** — この高頻度約定群が **唯一の net negative セグメント**。

低頻度（20-40 分間隔）は +3.446 bps/fill — **8 倍以上のリターン差**。

**rapid sell (gap<5min) の adverse selection**:
- rapid sell + trend_down (5s): n=18, 30s=**−3.011**
- rapid sell + trend_up (5s): n=19, 30s=−1.303

→ rapid fill 時の sell は方向に関わらず損失。informed flow に捕まっている。

---

### RC-5: Regime Confidence の逆説 (MEDIUM)

| confidence | n | 30s mean | WR |
|-----------|---|----------|-----|
| [0.3,0.5) | 37 | +0.126 | 54% |
| [0.5,0.7) | 203 | +0.556 | 52% |
| **[0.7,0.9)** | **158** | **−0.336** | **46%** |
| [0.9,1.0) | 13 | +0.239 | 62% |

confidence [0.7,0.9) は **全 4 バケットで最悪**。

サイド別 ([0.7,0.9) vs <0.7):
| side | conf<0.7 | 30s | WR | conf[0.7,0.9) | 30s | WR |
|------|----------|-----|-----|--------------|-----|-----|
| buy | 120 | +0.398 | 52% | 76 | **−0.591** | 43% |
| sell | 120 | +0.582 | 52% | 82 | −0.100 | 49% |

特に **buy + high confidence** の組み合わせが −0.591 bps。Bayesian Regime Filter が「自信を持って ranging と判定」している時に、実際にはトレンド転換が起きている可能性。

---

### RC-6: Wide Spread Kill Zone (MEDIUM)

| spread (bps) | n | big move率 | 30s mean | abs mean |
|-------------|---|-----------|----------|----------|
| [0,1.0) | 12 | 33% | +1.499 | 4.646 |
| [1.0,1.5) | 79 | 28% | +0.433 | 4.765 |
| [1.5,2.0) | 112 | 26% | +0.976 | 4.323 |
| [2.0,2.5) | 99 | 31% | +0.042 | 4.491 |
| [2.5,3.0) | 75 | 36% | −0.601 | 4.910 |
| **[3.0,5.0)** | **34** | **50%** | **−1.557** | **6.862** |

spread ≥ 3.0 bps では **half の約定が ±5bps 以上のビッグムーブ** に巻き込まれ、net −1.557 bps。

wide spread = 低流動性 or informed flow のシグナルであり、ここでの約定は本質的に不利。

---

### RC-7: 381# offset_ceiling_ratio_buy 変更の影響 (LOW)

| 期間 | n | 30s mean | WR | offset avg | spread avg |
|------|---|----------|-----|-----------|-----------|
| pre-3/12 (buy) | 132 | +0.539 | 52% | 0.1537 | 2.08 |
| 3/12 (buy) | 72 | **−0.595** | 47% | **0.2005** | 1.93 |

381# で ceiling_buy を 0.15→0.20 に変更後:
- offset が 0.15→0.20 に上昇 (設計通り)
- しかし PnL が +0.54→−0.60 に大幅悪化

**注意**: サンプル数が少なく (1 日分)、市場環境の変化も考えられるため、causal attribution は困難。ただし **offset を広げすぎると adverse selection に弱い (unfavorable) 約定ばかりが残る**可能性がある。

---

## 4. Loss Clustering 分析

### 4.1 連敗ストリーク

| streak 長 | 回数 |
|----------|------|
| 1 | 39 |
| 2 | 36 |
| 3 | 10 |
| 4 | 8 |
| 5 | 3 |
| 6 | 1 |
| **10** | **1** |

**最大 10 連敗** — cascade 的な損失が存在。

### 4.2 同一サイド連続約定

| パターン | n | 30s mean |
|---------|---|----------|
| 同一サイド連続 | 69 (17%) | +0.078 |
| サイド切替 | 341 (83%) | +0.184 |

同一サイド連続は全体の 17% — 大半はサイド交互。パフォーマンス差は小さい。

---

## 5. Fill Slippage 分析

| side | n | mean slip (bps) | median slip (bps) |
|------|---|----------------|-------------------|
| buy | 204 | −0.527 | −0.448 |
| sell | 207 | −0.685 | −0.486 |

**両サイドとも negative slippage** — fill price が mid price より不利な方向にある。
sell の slippage (−0.685) は buy (−0.527) より **30% 大きい** — sell 側の adverse selection を裏付け。

---

## 6. Macro Trend と Regime

### 6.1 Macro Trend

| macro_trend | n | 30s | WR |
|------------|---|-----|-----|
| macro_strong_down | 2 | −7.152 | 50% |
| macro_insufficient | 57 | −0.254 | 51% |
| macro_weak_down | 63 | +0.150 | 46% |
| macro_neutral | 199 | +0.053 | 50% |
| macro_weak_up | 74 | +0.819 | 55% |
| macro_strong_up | 16 | +0.995 | 44% |

**macro_insufficient (57件, −0.254)** — データ不足時に約定すべきではない。

### 6.2 Regime

| regime | n | 30s | WR |
|--------|---|-----|-----|
| ranging | 352 | −0.072 | 50% |
| trending_up | 35 | +1.393 | 54% |
| trending_down | 24 | +1.843 | 54% |

**ranging (86%) が圧倒的多数** — ranging 時の PnL がほぼゼロ (−0.07)。
trending 時は有利 (+1.4~+1.8 bps) — **trending detection 精度が上がれば利益拡大の余地あり**。

### 6.3 mid_price_trend_5s と PnL

buy side:
| trend_5s | n | 30s | WR |
|----------|---|-----|-----|
| [−20,−3) (下落中) | 28 | +0.797 | 46% |
| [−3,−1) | 29 | +0.662 | 59% |
| [−1,0) | 25 | +0.561 | 56% |
| **[0,+1)** (微上昇) | **20** | **−1.027** | **40%** |
| [+1,+3) | 35 | −0.068 | 54% |
| [+3,+20) (急上昇中) | 44 | +0.050 | 45% |

**buy + 微上昇 (0~1 bps/5s)**: WR=40% — 「ちょっとだけ上がった後の buy」が最悪。

sell side:
| trend_5s | n | 30s | WR |
|----------|---|-----|-----|
| **[−20,−3) (急下落中)** | **34** | **−1.929** | **47%** |
| [−3,−1) | 41 | +1.563 | 54% |
| [−1,0) | 23 | −0.995 | 39% |
| [0,+1) | 24 | −0.118 | 46% |
| [+1,+3) | 27 | +1.368 | 52% |
| [+3,+20) (急上昇中) | 36 | +0.913 | 61% |

**sell + 急下落 (−20~−3)**: −1.93 bps — **下落中に sell が約定 = 追随下落で損失**。
これは逆直感的: 下落中に売れたら有利に見えるが、実際は momentum に巻き込まれて不利。

---

## 7. 提言 (Priority 順)

### P0: CRITICAL — 即時対応

#### P0-1: Sell EV 方向ガード導入
**sell + EV>0 の 76 件が −0.481 bps/fill** — EV が上昇を予測しているのに sell 注文を出している。

```yaml
# 提案: ev_direction_gate
sell_ev_positive_threshold: 0.5  # EV>0.5 の sell を抑制
sell_ev_positive_action: skip    # or "widen_offset"
```

代替: EV>0 の場合に sell offset を追加拡大 (例: ×1.5)

**期待効果**: 76 件 × −0.481 → 0 bps minimum = **+36.6 bps 改善ポテンシャル**

#### P0-2: Rapid-fire 約定抑制
**gap<5min の 203 件が −0.427 bps** — 高頻度約定は net negative。

```yaml
# 提案: min_inter_fill_gap
min_fill_interval_seconds: 300   # 最低 5 分間隔
# or: rapid_fire_offset_boost: 1.5  # 5 分以内は offset ×1.5
```

**期待効果**: 203 件 × +0.43 → **+87.3 bps 改善ポテンシャル** (5 分制限) or offset 拡大で損失軽減

### P1: HIGH — 次回デプロイ

#### P1-1: Wide spread ガード強化
**spread ≥ 3.0 bps で −1.557 bps, 50% が big move** — kill zone。

```yaml
# 提案: spread_hard_gate
spread_skip_threshold_bps: 3.0   # spread ≥ 3.0 bps のとき skip
# or: spread_boost_threshold: 3.0; spread_boost_factor: 2.0
```

**期待効果**: 34 件 × +1.557 → **+52.9 bps 改善ポテンシャル**

#### P1-2: Regime confidence [0.7,0.9) 補正
**confidence 0.7-0.9 が最悪ゾーン** — "confidently wrong" 状態。

```yaml
# 提案: confidence_penalty
regime_confidence_caution_range: [0.7, 0.9]
regime_confidence_penalty_offset: 1.3  # offset ×1.3
```

#### P1-3: offset_ceiling_ratio_buy 再考
381# で buy ceiling を 0.20 に引き上げたが、3/12 で buy が −42.9 bps に劣化。
**0.15 に戻すか、少なくとも 0.17-0.18 程度に縮小** を検討。

### P2: MEDIUM — 計画的対応

#### P2-1: sell mid_price_trend_5s ガード
sell + 急下落 (trend_5s < −3) = −1.93 bps。

```yaml
# sell 急下落時にオフセット拡大
sell_downtrend_guard_threshold: -3.0  # bps/5s
sell_downtrend_guard_boost: 1.5       # offset ×1.5
```

#### P2-2: Loss cascade breaker
最大 10 連敗。連続損失後のクールダウン機構。

```yaml
# 提案: loss_cascade_breaker
consecutive_loss_threshold: 3  # 3 連敗で pause
loss_pause_seconds: 600       # 10 分間 pause
```

#### P2-3: macro_insufficient 時の Skip
macro data 不足時 (57 件) が net negative (−0.254)。

```yaml
# 提案: macro_insufficient_gate
macro_insufficient_action: skip  # or "widen_offset"
```

---

## 8. 構造的考察

### 8.1 Offset Pipeline の設計課題

現在の 12 段パイプラインは:
1. **60/411 (15%) で全ステージ同一値** — パイプライン段数の割に実効性が低い
2. **59% が ceiling (0.30) に張り付き** — ceiling が暗黙の "default" になっている
3. **spread_offset_ratio = effective_offset_used** — spread_offset_ratio がそのまま使われており、中間変換が無い

これは「多段に見えるが実際は ceiling が支配的」という **擬似的複雑性** (pseudo-complexity) を示唆。

### 8.2 Maker の構造的不利

Maker (limit order) は定義上、**informed flow に対して逆選択される**。
これは解消困難な本質的問題であり、対策は:
1. **adverse selection の事前検知** (EV, trend, spread signal)
2. **十分な offset による protection** (但し広すぎると fill rate 低下)
3. **事後的な cascade 防止** (loss breaker)

の三重構造が必要。現在は (2) のみに依存しすぎており、(1)(3) が不十分。

### 8.3 EV Model の本質的限界

EV [1,2) (中確信帯) が最悪パフォーマンスという事実は、**model uncertainty のキャリブレーション失敗** を示唆。
- EV 極小 (< −3) / 極大 (> 2): 予測が当たる (WR 54-57%)
- EV 中間 (1-2): 予測が外れる (WR 40%)

これは典型的な **overconfidence bias** — model が「少し良い」と思った状況で実際は neutral or negative。

---

## 9. 次ステップ

1. **P0-1 (EV 方向ガード)**: sell + EV>0 の offset 拡大を実装・テスト
2. **P0-2 (rapid-fire 抑制)**: min_fill_interval or rapid offset boost の実装
3. **P1-3 (buy ceiling)**: offset_ceiling_ratio_buy を 0.18 に微調整
4. 3/13 以降の fill データで 381# 変更の統計的有意性を継続検証
5. EV model の recalibration — 特に [1,2) ゾーンの feature 分析

---

> **結論**: 全体 PnL はプラスだが、sell tail risk と rapid-fire の構造的問題が利益を削っている。
> P0 施策 (EV 方向ガード + rapid-fire 抑制) で **+100 bps 以上の改善ポテンシャル** がある。
> これは Pipeline 改修ではなく Gate/Guard ロジックの追加であり、既存構造を破壊しない。
