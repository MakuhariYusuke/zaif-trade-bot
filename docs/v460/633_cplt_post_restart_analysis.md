# 633# SHA別性能分析: Era-D (CalibrationMap) の黒字構造と Era-E (630# 閾値) の clamp 飽和問題

## 概要
632# (ATR floor mult=2.0→1.2 + cap_bps=3.0) 適用検証に加え、入金不可制約下で 3/16-3/25 の全 fill records を **SHA (コミット) 単位で分析**。
25 SHA を 5 つの Era に分類し、config/code 変更の各々がどう性能に影響したかを定量的に判定した。

**最重要発見**: Era-D (554#-594#, CalibrationMap 導入期) だけが **+2.0 JPY/fill** で黒字。
Era-E (620#-632#, 630# 閾値チューニング適用後) は **-1.5 JPY/fill** に悪化し、clamp 飽和が 0%→59-64% に爆発。
630# の「閾値引き下げ」は AS 防御を強化したが、オフセット膨張→clamp 天井衝突→保護不足という **副作用が主効果を上回った**。

## 1. 632# ATR Floor 検証結果 (成功)

| 指標 | 旧(mult=2.0) | 新(mult=1.2, cap=3.0) |
|------|-------------|---------------------|
| Floor値 (σ=0.000244) | 5,534 JPY (4.88 bps) | 3,320 JPY (2.93 bps) |
| spread_too_narrow/h | ~38回/h | ~4回/h |
| 削減率 | — | **89%** |

---

## 2. Era 定義と全体比較

25 SHA を config/code マイルストーンで 5 Era に分類:

| Era | 期間 | 主要変更 | Cycles | Fills | Fill% | PnL/fill | Clamp Sell% | Clamp Buy% |
|-----|------|---------|--------|-------|-------|----------|-------------|------------|
| **A** | 3/16-17 | 454#-459# micro-timeout, lot=0.001 | 1,106 | 164 | 15% | **-0.3** | 1% | 0% |
| **B** | 3/19-21 | 499# loss_cap fix, lot→0.002 | 1,013 | 431 | 43% | **-0.8** | 0% | 0% |
| **C** | 3/21-23 | 527# JPY prec, CV kill, min_spread=500 | 846 | 167 | 20% | **-1.1** | 0% | 0% |
| **D** | 3/23-24 | 554# CalibrationMap, eDRC, ev_toxic | 537 | 147 | 27% | **+2.0** | 15% | 8% |
| **E** | 3/25 | 620# fix, 630# vel=4/trend=0.20/VG=6, 632# ATR | 695 | 162 | 23% | **-1.5** | 59% | 64% |

**Era-D が唯一の黒字 (+2.0/fill)**。Era-E は 630# 閾値チューニング適用後だが、clamp 飽和が激増し per-fill edge が悪化。

### 2.1 Era 間の構造変化

| 指標 | Era-B | Era-C | Era-D | Era-E | 根拠 |
|------|-------|-------|-------|-------|------|
| Sell AS率 | 39% | 32% | 29% | 36% | Era-D 最良 |
| Buy AS率 | 25% | 16% | 25% | 18% | Era-C/E 最良 |
| VG trigger率 | 99% | 99% | 97% | 95% | 全 Era で過大 |
| VG avg boost | 1.418 | 1.133 | 1.214 | 1.213 | Era-B 最重 |
| Preflight block | 1% | 40% | 45% | 35% | Era-C/D で Death Spiral |
| Sell avg offset | 0.195 | 0.255 | 0.395 | 0.382 | Era-D/E で CalibMap 効果 |
| Buy avg offset | 0.227 | 0.255 | 0.341 | 0.315 | 同上 |
| Sell fast p30 | +1.30bps | -0.80bps | +1.57bps | -2.03bps | **Era-E 最悪** |

---

## 3. SHA 別詳細判定

### Era-A: 454#-459# (3/16-17) — Baseline, lot=0.001

| SHA (短) | コミット | n | fills | fill% | net JPY | /fill | 判定 |
|----------|---------|---|-------|-------|---------|-------|------|
| c7ebd8c | 454# micro-timeout有効化 | 192 | 30 | 16% | +80 | **+2.7** | ✅ 良好 |
| f840d0e | 459# hot-reload deadlock fix | 321 | 93 | 29% | -152 | **-1.6** | ❌ 損失 |
| d0769f2 | 458# self-review hot-reload | 317 | 12 | 4% | -25 | -2.1 | ⚠ fill率壊滅 |
| 52627ff | 450# DRY test helpers | 117 | 14 | 12% | -6 | -0.4 | — 中立 |
| a9714ad | 445# cross-venue EMA | 80 | 9 | 11% | +5 | +0.5 | — 中立 (少量) |
| f34467b | 442# Cross-Venue有効化 | 79 | 6 | 8% | +54 | **+9.0** | ✅ (n=6, 信頼性低) |

**Era-A 判定**: `c7ebd8c` (454#) が安定黒字。`f840d0e` (459#) は fill率改善 (29%) したが per-fill -1.6 で損失拡大 — hot-reload 有効化でサイクル増加 → AS 被害増。`d0769f2` は self-review 中で 4% fill (82% narrow block)。

### Era-B: 499#-session037 (3/19-21) — lot 増量, 最高 fill 率

| SHA (短) | コミット | n | fills | fill% | net JPY | /fill | S_ceil | B_ceil | 判定 |
|----------|---------|---|-------|-------|---------|-------|--------|--------|------|
| **dfbe3b5** | **499# hard_loss_cap fix** | 332 | 169 | **51%** | **+78** | **+0.5** | 0% | 0% | **✅ BEST** |
| 548dda2 | session037 test align | 181 | 58 | 32% | -138 | **-2.4** | 0% | 0% | ❌ S=-152 |
| 5a546923 | session037 SAC/Bayesian promote | 160 | 89 | 56% | -65 | -0.7 | 0% | 0% | ⚠ B=-200 |
| 20d4f77 | session037 test import sweep | 340 | 115 | 34% | -201 | **-1.7** | 0% | 0% | ❌ 損失 |

**Era-B 判定**: `dfbe3b5` (499#) が **全25 SHA中最良の総合スコア** — 51% fill率 × +0.5/fill × clamp 0%。
499# の hard_loss_cap 修正自体はクラッシュループ修正で性能と直接無関係 → **この時期の market condition (3/19-20) が好適だった可能性**。
session037 系 SHA は test refactor が主体で config 変更なし。`548dda2` は sell -152 JPY の大損が特徴的。

💡 **626# (sell loss root cause) との対照**: 626# は sell AS 率 69% を報告したが、Era-B sell AS 率は 39% — 時期による AS 率の変動が大きい。

### Era-C: 527#-549# (3/21-23) — min_spread 引き下げ, Death Spiral 開始

| SHA (短) | コミット | n | fills | fill% | net JPY | /fill | preflight% | 判定 |
|----------|---------|---|-------|-------|---------|-------|------------|------|
| d93b9a5 | 527# JPY精度改善 | 583 | 106 | 18% | -113 | **-1.1** | 42% | ❌ |
| 8a63d95 | 535# CV kill + min_spread 700→500 | 38 | 8 | 21% | -14 | -1.7 | n=38 | ⚠ (少量) |
| d79e669 | 539# 風水渙棚卸し | 123 | 22 | 18% | -55 | **-2.5** | 41% | ❌ |
| 99ca951 | 549# EWMA Winsorize | 102 | 31 | 30% | -4 | **-0.1** | 26% | ✅ 改善 |

**Era-C 判定**: preflight 40% 突入で fill 率が Era-B の半分以下に激減。`d93b9a5` (527#) は最多サイクル SHA だが -1.1/fill。
`99ca951` (549# EWMA Winsorize) で sell_dynamic_kill スパイラルを修正 → preflight 26% に改善、-0.1/fill でほぼ break-even。
**min_spread 700→500 (535#)** の影響は n=38 で統計不足。ただし Era-C 全体で narrow block が 4% (Era-A の 7% より低下) → 適切な緩和。

### Era-D: 554#-594# (3/23-24) — CalibrationMap, **唯一の黒字 Era**

| SHA (短) | コミット | n | fills | fill% | net JPY | /fill | S_ceil | B_ceil | 判定 |
|----------|---------|---|-------|-------|---------|-------|--------|--------|------|
| **447b2ec** | **554# CalibrationMap** | 86 | 41 | **48%** | **+152** | **+3.7** | 30% | 14% | **✅ 最良** |
| **c164d21** | **573# telemetry fix** | 179 | 73 | **41%** | **+149** | **+2.0** | 3% | 7% | **✅ 良好** |
| 8e37cf9 | test: coverage extend | 72 | 33 | 46% | -11 | -0.3 | 19% | 0% | — ほぼ中立 |
| 29fe26e | 594# ev_toxic_skip | 149 | 0 | 0% | 0 | — | — | — | ❌ 全block |
| cb1cb85 | 575# eDRC有効化 | 51 | 0 | 0% | 0 | — | — | — | ❌ 全block |

**Era-D 判定 — 極めて重要**:

1. **`447b2ec` (554# CalibrationMap) と `c164d21` (573# telemetry fix) が連続黒字** — CalibMap がオフセットを市場状態に応じて適切に設定し、clamp 飽和率も穏健 (15-30%)
2. **`29fe26e` (594# ev_toxic_skip) と `cb1cb85` (575# eDRC) は填充ゼロ** — preflight 83% / 53% で完全ブロック。Balance 不足が致命的に。eDRC 有効化 (`α=0.020, β=0.40`) 自体の影響は不明 (fill 0 で評価不能)
3. **Sell avg offset 0.395** — Era-B の 0.195 から倍増。CalibMap が sell 保護を強化した結果、per-fill edge が正転

💡 **630# (threshold tuning) との対照**: 630# P1 は Era-D の成功を **前提に** vel/trend/VG 閾値を下げた。しかし Era-E ではその追加 boost がオフセットをさらに押し上げ → clamp 天井衝突。

### Era-E: 620#-632# (3/25) — 630# 閾値チューニング適用後

| SHA (短) | コミット | n | fills | fill% | net JPY | /fill | S_ceil | B_ceil | 判定 |
|----------|---------|---|-------|-------|---------|-------|--------|--------|------|
| ce31662 | 620# skip_gate regime bypass fix | 156 | 70 | **45%** | -39 | **-0.6** | 45% | 66% | ⚠ |
| **2ac4d05** | **refactor: analysis results-dir** | 176 | 68 | 39% | **-197** | **-2.9** | **73%** | **60%** | **❌ 最悪** |
| 88274227 | 632# numpy import fix | 40 | 13 | 32% | -16 | -1.2 | 57% | 83% | ❌ (少量) |
| a8a9d2e | 631# venv launcher docs | 213 | 10 | 5% | 0 | 0.0 | 60% | 40% | ⚠ fill壊滅 |
| a8275cc | 631# orphan cleanup fix | 33 | 1 | 3% | +1 | +0.6 | — | — | — (n=1) |
| 15ee9e0 | 630# P1 threshold tuning | 77 | 0 | 0% | 0 | — | — | — | ❌ 全block |

**Era-E 判定 — 630# 閾値チューニングの副作用が主因**:

1. **Clamp 飽和が 0%→59-64% に爆発**: Era-B/C は sell/buy とも clamp ceiling hit 0%。Era-D で CalibMap 導入後に 15% 出現。Era-E で 630# 適用後に 59-64% に急増
2. **`2ac4d05` が全 SHA 中ワースト (-2.9/fill)** — コードは "refactor: analysis results-dir" で config 無変更。73% sell clamp 飽和が示すとおり、モデルが必要とするオフセット (avg pre-clamp ~0.60) を 0.40 で切り詰め → 保護不足で AS 被害
3. **`15ee9e0` (630# P1 threshold tuning 自体) は fill=0** — preflight block で評価不能。閾値変更の影響は後続 SHA で現出
4. **630# が作った構造**: vel_skip 4bps + trend 0.20% + VG 6bps → 3 段すべてがオフセット膨張方向 → CalibMap ベースオフセット + vel boost + trend boost + VG boost → 合計が clamp ceiling 0.40 を大幅超過 → 切り詰め → 保護不足
5. **Sell fast fill p30 = -2.03bps** (Era 全体) — 全 Era 中最悪。即座約定 = informed trader による picking off

💡 **630# セルフレビューの予言**: 630# 自身が「clamp 飽和との相互作用 ⚠ ceiling sell=0.40 で offset boost が頭打ちになる可能性（P2 課題）」と警告 — **この予言が的中**。

---

## 4. クロス Era 因果分析

### 4.1 なぜ Era-D だけが黒字か

```
Era-B: CalibMap なし → base offset 低い (0.195) → clamp 不要 → VG 99% だが boost 弱い (1.4x)
       → offset 合計が中庸 → fill率 高い (43%) → 市場が好適なら黒字 (dfbe3b5)

Era-D: CalibMap あり → base offset 適切 (0.395) → clamp 穏健 (15%) → VG 97%, boost 1.2x
       → offset 合計が「十分な保護」かつ「fill 可能」なバランスゾーン → +2.0/fill

Era-E: CalibMap + 630# boost 3段 → base offset 0.382 + vel/trend/VG 追加 → 合計 >> 0.40
       → clamp ceiling 59% → 保護が 0.40 で頭打ち → Era-D と同じ保護水準なのに余分な boost がfill率を下げた
       → 「boost したのに clamp で切られ、fill だけ減った」= 純粋な劣化
```

### 4.2 Clamp 飽和メカニズム (630# の副作用)

630# 変更前 (Era-D):
```
offset = CalibMap_base × (1 + VG_boost) × regime_mult
       ≈ 0.30 × 1.2 × 1.0 = 0.36 → clamp 0.40 以内 ✅
```

630# 変更後 (Era-E):
```
offset = CalibMap_base × (1 + vel_boost) × (1 + VG_boost) × regime_mult × trend_boost
       ≈ 0.30 × 1.3 × 1.2 × 1.0 × 1.8 = 0.84 → clamp 0.40 で切断 ❌

実測: pre-clamp avg=0.60, max=0.97 → 0.40 で強制切り詰め
```

**帰結**: 630# のすべての閾値引き下げが同時にオフセットを膨張させ、乗算パイプラインで指数的に増幅。clamp ceiling が安全弁として機能したが、「安全弁が常時作動 = 設計逸脱」。

### 4.3 VG 問題の全 Era 共通性

| Era | VG trigger率 | VG avg boost |
|-----|-------------|-------------|
| A | 94% | 1.681 |
| B | 99% | 1.418 |
| C | 99% | 1.133 |
| D | 97% | 1.214 |
| E | 95% | 1.213 |

**全 Era で 94-99% トリガー** — `vpin_continuous_min=0.40` が avg vpin=0.623 に対し低すぎるのは Era 共通問題。ただし実質的な影響度は boost 値次第で、Era-C (1.133) は比較的軽微、Era-A (1.681) は最重。

### 4.4 Death Spiral の時系列

| Era | Preflight% | 原因 |
|-----|-----------|------|
| A | 0% | 残高十分 (入金直後) |
| B | 1% | 残高十分 (3/18 入金の恩恵) |
| C | **40%** | **ターニングポイント** — 残高が閾値を割る |
| D | **45%** | さらに悪化、しかし per-fill は黒字 |
| E | **35%** | ATR floor 修正 + lot 縮小で微改善 |

Era-C で Death Spiral 開始。しかし Era-D は preflight 45% (最悪) でも黒字 — **per-fill edge さえ正なら、low fill rate でも回復可能**。

---

## 5. SHA 別判定サマリ (成績順)

| 順位 | SHA | Era | コミット概要 | fills | /fill | clamp | 判定 |
|------|-----|-----|-------------|-------|-------|-------|------|
| 1 | f34467b | A | 442# Cross-Venue有効化 | 6 | +9.0 | 0% | ✅ (n少) |
| 2 | 447b2ec | D | **554# CalibrationMap** | 41 | **+3.7** | 15%/14% | **✅ 最良** |
| 3 | c7ebd8c | A | 454# micro-timeout | 30 | +2.7 | 0% | ✅ |
| 4 | c164d21 | D | **573# telemetry fix** | 73 | **+2.0** | 3%/7% | **✅ 良好** |
| 5 | dfbe3b5 | B | **499# hard_loss_cap** | 169 | **+0.5** | 0% | **✅ 大量fill黒字** |
| 6 | a9714ad | A | 445# cross-venue EMA | 9 | +0.5 | 0% | — 少量 |
| 7 | 99ca951 | C | 549# EWMA Winsorize | 31 | -0.1 | 0% | — break-even |
| 8 | 8e37cf9 | D | test coverage | 33 | -0.3 | 19% | — 中立 |
| 9 | 52627ff | A | 450# DRY helpers | 14 | -0.4 | 0% | — 中立 |
| 10 | ce31662 | E | 620# skip_gate fix | 70 | -0.6 | 45%/66% | ⚠ clamp |
| 11 | 5a546923 | B | session037 SAC promote | 89 | -0.7 | 0% | ⚠ B=-200 |
| 12 | d93b9a5 | C | 527# JPY精度 | 106 | -1.1 | 0% | ❌ |
| 13 | 88274227 | E | 632# numpy fix | 13 | -1.2 | 57%/83% | ❌ clamp |
| 14 | f840d0e | A | 459# hot-reload | 93 | -1.6 | 0% | ❌ |
| 15 | 20d4f77 | B | session037 test sweep | 115 | -1.7 | 0% | ❌ |
| 16 | 8a63d95 | C | 535# CV kill | 8 | -1.7 | 0% | — 少量 |
| 17 | d0769f2 | A | 458# self-review | 12 | -2.1 | 0% | ❌ fill壊滅 |
| 18 | 548dda2 | B | session037 test align | 58 | -2.4 | 0% | ❌ S=-152 |
| 19 | d79e669 | C | 539# 棚卸し | 22 | -2.5 | 0% | ❌ |
| 20 | **2ac4d05** | **E** | **refactor analysis** | 68 | **-2.9** | **73%/60%** | **❌ 最悪** |
| — | 29fe26e | D | 594# ev_toxic | 0 | — | — | ❌ 全block |
| — | cb1cb85 | D | 575# eDRC | 0 | — | — | ❌ 全block |
| — | 15ee9e0 | E | 630# threshold | 0 | — | — | ❌ 全block |
| — | a8275cc | E | 631# orphan fix | 1 | +0.6 | — | — n=1 |
| — | a8a9d2e | E | 631# venv docs | 10 | 0.0 | — | ⚠ fill壊滅 |

---

## 6. 過去分析ドキュメントとの対照

### 626# (Sell 損失構造分析) との対照
- 626# 報告: sell AS率 69%, AS単価被害 -11.34 JPY, velocity 閾値 6.0bps が到達不可能
- **SHA別検証**: Era-B sell AS率は 39%, Era-D は 29%, Era-E は 36% → **626# の 69% は特定日 (3/25) の値で Era 全体では再現しない**
- **判定**: 626# の vel 閾値問題の指摘は正しいが、修正 (630#) の副作用 (clamp 飽和) を過小評価

### 630# (P1 閾値チューニング) との対照
- 630# 変更: vel 6→4, trend 0.5→0.20, VG 12→6
- 630# セルフレビュー警告: 「clamp 飽和との相互作用 ⚠」「trending_up_sell_offset_boost=1.8 発火増」
- **SHA別検証**: Era-E (630# 適用後) で clamp sell 59%, buy 64% → **セルフレビューの懸念が的中**
- **判定**: 3 つの閾値を同時に引き下げた結果、乗算パイプラインで指数的にオフセットが膨張。1 つずつ段階的に変更すべきだった

### 632# (ATR Floor) との対照
- 632# 変更: ATR mult 2.0→1.2, cap_bps=3.0
- **SHA別検証**: Era-E で narrow block 22% → ATR 修正後は ~4件/h に改善
- **判定**: ✅ 成功。spread_too_narrow 問題を 89% 解消。ただしこれだけでは per-fill edge は改善しない

---

## 7. 構造的知見と改善提案

### 7.1 核心問題: 630# 閾値チューニングの巻き戻し検討

Era-D (+2.0/fill) → Era-E (-1.5/fill) の悪化の主因は 630# の 3 閾値同時変更。

**提案 P0: 630# の部分ロールバック** (信頼度: ★★★★☆)
```yaml
# vel_skip は維持 (626# の指摘で効果確認済み)
sell_velocity_skip_threshold_bps: 4.0  # 維持

# trend は元に戻す — 0.20% は boost 発火過剰
regime_trend_threshold_pct: 0.35  # 0.20→0.35 (0.50 まで戻さず中間値)

# VG vel は 8.0 に緩和 — 6.0 は vel_skip 4.0 との重複が大きすぎ
volatility_guard:
  velocity_threshold_bps: 8.0  # 6.0→8.0
```
**根拠**: vel_skip 4.0 で AS 第一防御線は確保。trend 0.35% + VG 8.0 で第二・第三防御線の過剰発火を抑制 → clamp 飽和率を Era-D 水準 (15%) まで低減。

### 7.2 Sell Clamp Ceiling 微増 (P0 と併用)

**提案 P1: sell ceiling 引き上げ** (信頼度: ★★★☆☆)
```yaml
offset_ceiling_ratio_sell: 0.50  # 0.40→0.50
```
**根拠**: Era-D の 447b2ec は sell_ceil=30% で +3.7/fill を達成。Era-E は 59% で -1.5/fill。
0.50 に引き上げることで、CalibMap + 適度な boost が clamp に当たらなくなり、Era-D の構造を再現可能。
P0 (630# 部分ロールバック) と併用で clamp 飽和を 10-20% に誘導。

### 7.3 VG vpin_continuous_min 引き上げ (従来の P0、優先度据え置き)

**提案 P2: VG recalibration** (信頼度: ★★★★☆)
```yaml
volatility_guard:
  vpin_continuous_min: 0.55  # 0.40→0.55
```
**根拠**: 全 Era で 94-99% トリガーは構造問題。ただし Era-D は VG 97% でも黒字 → 優先度は P0/P1 より下。

### 7.4 変更順序

1. **P0 (trend/VG 閾値緩和)**: config-only、ロールバック容易、clamp 飽和の根本原因を直接攻撃
2. **P1 (sell ceiling 引き上げ)**: P0 の効果不十分なら追加
3. **P2 (VG recalibration)**: P0/P1 後の残存問題として対処

---

## 8. 結論

### SHA 別分析で判明した事実
1. **Era-D (CalibrationMap 期) だけが黒字** — 554# CalibMap + 573# telemetry fix の 2 SHA が +2.0〜+3.7/fill
2. **630# 閾値チューニング (Era-E) は per-fill edge を悪化させた** — clamp 飽和 0%→59-64%、/fill が +2.0→-1.5
3. **dfbe3b5 (499#) の好成績は市場条件要因が大きい** — config/code 変更は crash fix で性能無関係、3/19-20 の市場が好適
4. **Death Spiral は Era-C (3/21-) で開始** — preflight 0%→40%。ただし Era-D は preflight 45% でも黒字 → per-fill edge が鍵
5. **VG 恒常発動は全 Era 共通** — 緊急度は clamp 飽和問題より低い

### 最優先アクション
**630# の trend/VG 閾値を部分ロールバック** → clamp 飽和を Era-D 水準に戻す → per-fill edge 正転を目指す
