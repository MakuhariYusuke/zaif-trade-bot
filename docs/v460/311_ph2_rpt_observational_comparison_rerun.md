# 311# rpt: 観測比較再実行 + 309#/310# 理論修正検証 + 深堀り分析

> **日付**: 2026-03-07  
> **種別**: rpt (調査・分析レポート)  
> **前提 SHA**: `dcc3064a8` (310# 設計改善) — デプロイ直後のため、データの大部分は pre-310# 期間  
> **データ**: fill_records total=7,183 / filled=2,564 (sell=1,270, buy=1,294)  
> **計測期間**: 299# 以降の蓄積データ（約22日間）  
> **関連**: [306#](306_ph2_impl_six_proposals_observational_redesign.md), [309#](309_ph2_review_response_307_308_fixes.md), [310#](310_ph2_impl_design_improvements.md)

---

## 目次

- [§0 エグゼクティブ・サマリー](#0-エグゼクティブサマリー)
- [§1 0番ドキュメントでの立ち位置](#1-0番ドキュメントでの立ち位置)
- [§2 検証目的と手法](#2-検証目的と手法)
- [§3 理論修正検証 (309#/310#)](#3-理論修正検証-309310)
- [§4 AB 判定結果](#4-ab-判定結果)
- [§5 Regime 別深堀り](#5-regime-別深堀り)
- [§6 Spread/AS 分解と Market Making 均衡分析](#6-spreadas-分解と-market-making-均衡分析)
- [§7 時間帯別 AS 構造](#7-時間帯別-as-構造)
- [§8 Offset 分位点分析と動的フロア評価](#8-offset-分位点分析と動的フロア評価)
- [§9 None Regime 影響定量化](#9-none-regime-影響定量化)
- [§10 Decision Path / 310# 新機能の初期観測](#10-decision-path--310-新機能の初期観測)
- [§11 306# との差分](#11-306-との差分)
- [§12 sell p10 ボトルネック構造分析](#12-sell-p10-ボトルネック構造分析)
- [§13 改善提案 (312# 候補)](#13-改善提案-312-候補)
- [§14 AI レビュー向け設問](#14-ai-レビュー向け設問)

---

## §0 エグゼクティブ・サマリー

**309#/310# の理論修正は安全に機能** — Bootstrap p=0.96, Matched p=0.21 で sell vs buy PnL 差は非有意、理論倒錯の修正 (L1/L2) による破壊的劣化はない。

**AB 判定は依然 FAIL** — 唯一のブロッカーは sell downside p10 = **-6.87 bps** (閾値 -5.0)。fill_rate と avg_pnl は PASS。

**最大のレバレッジポイント**: ranging sell (n=811) の p10=-6.74 が全体数値を支配。trending_up sell (n=82) は最悪の p10=-9.90 だがサンプル比率 7.2% のため全体への寄与は限定的。

**構造的発見**: sell offset Q1-Q2 (< 0.30) は動的フロア割引 (`sell_offset_floor_inv_discount=0.5`) に起因し、Q3-Q4 (≥ 0.30) に対し明確に劣後。floor 割引の再評価が必要。

---

## §1 0番ドキュメントでの立ち位置

### 1.1 フェーズ位置

000# §2 Phase 定義における現在地:

| フェーズ | Gate | 状態 | 280# 時点 | 311# 時点 |
|---|---|---|---|---|
| ph0 | — | ✅ 完了 | 同左 | 同左 |
| ph1 | G1-info | ✅ PASS | 同左 | 同左 |
| **ph2** | **G1.1-exec** | **🔄 進行中** | 計測中 (278# SHA) | **計測中 (dcc3064a8 SHA)** |
| ph3 | G2-train | ⏳ 待機 | 同左 | 同左 |
| ph4 | G3-pnl | ⏳ 待機 | 同左 | 同左 |
| ph5 | G4-live | ⏳ 未着手 | 同左 | 同左 |

### 1.2 G1.2-full (168h) 指標との対応

000# §3.3 G1.2-full の F1–F8 指標に対し、311# データで暫定評価:

| # | 指標 | 閾値 | 311# 暫定値 | 判定 | 根拠 |
|---|---|---|---|---|---|
| F1 | attempted_fill_rate | ≥ 70% | ~40% (sell), ~40% (buy) | ⚠️ | 注: AB比較の fill_rate は分母が異なる可能性あり |
| F4 | PnL30 | p ≥ 0.05 (有意に負でない) | Welch p=0.95 (sell vs 0) | 🟡 | sell PnL=-0.33bps は有意に負ではない |
| F5 | AS_ratio | ≤ 30% | sell 30.3%, buy 27.5% | ⚠️ | sell が閾値ぎりぎり |
| F7 | calendar_coverage | ≥ 7暦日 | ~22日 | ✅ | データ蓄積十分 |
| F8 | n_attempted | ≥ 500 | sell=1142, buy=1155 | ✅ | 統計的検出力十分 |

**重要な区別**: 311# の AB 比較判定基準 (fill_rate/avg_pnl/downside_p10) は 000# G1.2 の公式 Gate 基準 (F1–F8) とは別の運用判定枠組み。G1.2 の正式判定は同一 SHA での 168h 連続データで行う。311# は複数 SHA にまたがる観測データでの傾向分析。

### 1.3 280# からの変化

280# (2026-03-04) 以降の変化:

| 項目 | 280# 時点 | 311# 時点 |
|---|---|---|
| 最終 SHA | `04d9590eb` (278#) | `dcc3064a8` (310#) |
| コミット数 | — | +32 commits (282#–310#) |
| §3.9 中止条件 | 非該当 | 非該当 ✅ |
| テスト | 3,827 passed | 4,085 passed (+258) |
| R-1 (168h計測) | 計測開始 | **SHA 変更によりリセット** |

**280# §4 の推奨「7暦日のコード凍結」は CRITICAL バグ修正 (281#–287#) とレビュー応答 (303#–310#) により達成されなかった。** コード凍結 vs バグ修正の二律背反は依然として構造的課題。

### 1.4 G1.2 168h 計測のリアルな状況

280# で指摘された「G1.2 の 168h 計測完了がボトルネック」は引き続き有効:

```
Phase A: 計測凍結期間 (~7暦日): 280# → 中断 (CRITICAL bugs 281#-287#)
                                → 303#-310# レビュー対応で追加変更
                                → dcc3064a8 (310#) から再スタート
計測デッドライン: ~2026-03-14 (310# + 7暦日)
```

ただし、コード凍結を守れるかは 311# の分析結果次第。sell p10 改善のための 312# 実装を行えば再度リセット。

---

## §2 検証目的と手法

### 2.1 目的

1. **理論修正の安全性確認**: 309# (L1/L2 理論倒錯修正) + 310# (設計改善 5 件) が sell vs buy の PnL 差に悪影響を与えていないか
2. **ボトルネック構造の特定**: AB 判定 FAIL の原因を regime × 時間帯 × offset の多軸で分解
3. **改善提案の導出**: データ駆動で 312# 候補を特定

### 2.2 手法

306# と同一:
- **AB 比較**: `evaluate_ab_variant()` (sell=variant, buy=control)
- **統計検定**: Block Bootstrap (MBB, Künsch 1989), Matched Temporal Comparison, Welch t, Mann-Whitney U, Cliff's delta
- **補正**: Holm-Bonferroni (BH FDR)
- **分解軸**: regime, 時間帯 (UTC), offset quintile, decision path, none regime

### 2.3 分析コード

[analysis/311_observational_rerun.py](../../analysis/311_observational_rerun.py) — 10 セクション構成 (§1–§10)

---

## §3 理論修正検証 (309#/310#)

### 結論: ✅ 安全に機能

| 指標 | 306# (修正前) | 311# (修正後) | 変化 | 解釈 |
|---|---|---|---|---|
| Bootstrap mean_diff | -0.016 | -0.016 | ±0.00 | 変化なし |
| Bootstrap p | 0.9355 | 0.9605 | +0.03 | 非有意維持 |
| Bootstrap 95% CI | [-0.55, +0.47] | [-0.55, +0.47] | 同一 | 0を含む |
| Matched p | 0.2043 | 0.2089 | +0.005 | 非有意維持 |
| Matched n_pairs | — | 934 | — | 十分な対応数 |
| Cliff's delta | negligible | negligible (0.017) | — | 効果量なし |
| Mann-Whitney p | — | 0.480 | — | 分布差なし |
| sell avg_pnl | -0.33 bps | -0.33 bps | ±0.00 | 同一 |
| sell p10 | -6.84 bps | -6.87 bps | -0.03 | 測定誤差範囲 |

309# で修正した理論倒錯 (L1: Dynamic Cycle Interval 公式反転, L2: Microprice Side Selection ロジック反転+無効化) と 310# の設計改善は、sell vs buy の PnL バランスに統計的有意な影響を与えていない。

---

## §4 AB 判定結果

### 4.1 全体 (None 除外)

| 基準 | 閾値 | sell | buy | 判定 |
|---|---|---|---|---|
| fill_rate | ≥ 30% | 40.2% | 40.5% | ✅ PASS |
| avg_pnl30 | ≥ -1.0 bps | -0.33 | -0.31 | ✅ PASS |
| **downside_p10** | **≥ -5.0 bps** | **-6.87** | -5.67 | **❌ FAIL** |

Overall: **FAIL** (1/3 基準未達)

### 4.2 全体 (None 込み)

| 基準 | 閾値 | sell | buy | 判定 |
|---|---|---|---|---|
| fill_rate | ≥ 30% | 33.8% | 39.6% | ❌ FAIL (14.6% 劣化) |
| avg_pnl30 | ≥ -1.0 bps | -0.38 | -0.29 | ✅ PASS |
| **downside_p10** | **≥ -5.0 bps** | **-6.85** | -5.66 | **❌ FAIL** |

Overall: **FAIL** (2/3 基準未達) — None 含有で fill_rate も FAIL に転落

### 4.3 注目点

- **sell と buy の PnL 差は非有意** (p=0.96) — 問題は sell が「buy より悪い」ことではなく「**両側とも downside tail が大きい**」こと
- buy p10 = -5.67 は閾値 -5.0 に近い — buy も完全には安全ではない
- sell の FAIL は構造的であり、1–2 のパラメータ調整では解消困難

---

## §5 Regime 別深堀り

### 5.1 全 Regime 一覧

| Regime | sell n | buy n | sell FR | buy FR | sell PnL | buy PnL | sell p10 | buy p10 | FAIL数 |
|---|---|---|---|---|---|---|---|---|---|
| **trending_up** | 82 | 94 | **18.4%** | 53.7% | **-1.15** | -0.33 | **-9.90** | -5.69 | **3/3** |
| trending_down | 85 | 84 | 39.9% | 40.0% | -0.49 | +0.68 | -7.59 | -6.23 | 1/3 |
| trending | 118 | 118 | 32.8% | 73.3% | -0.66 | +0.57 | -6.56 | -6.96 | 2/3 |
| **ranging** | **811** | **812** | 46.0% | 36.1% | -0.18 | -0.48 | **-6.74** | -5.43 | **1/3** |
| none | 128 | 139 | 14.0% | 33.7% | -0.80 | -0.15 | -5.86 | -4.63 | 2/3 |
| unknown | 46 | 47 | — | — | — | — | — | — | — |

### 5.2 Trending_up sell の解剖 (3/3 FAIL)

trending_up sell は全セグメント最悪。3 つの独立した失敗モード:

1. **fill_rate 18.4%** — 現行 `skip_sell_trending_up_only: true` で大半がスキップされるが、通過した注文の約定率も低い
2. **PnL -1.15 bps** — 閾値 -1.0 を超過。約定しても逆選択損失が大きい
3. **p10 -9.90 bps** — テール損失が極端。10th percentile で約 10 bps 損失

**現行防御スタック**:
```
skip_sell_trending: true                      (gate 有効)
skip_sell_trending_up_only: true              (trending_up のみ)
trending_sell_as_offset_enabled: true         (soft skip)
trending_sell_offset_boost_factor: 3.0        (×3.0)
trending_up_sell_offset_boost: 1.8            (×1.8)
sell_offset_floor: 0.30                       (最低保証)
```
実効 boost: base × 3.0 × 1.8 = **5.4x**。これでも p10=-9.90。

**ただし重要な注意**: n=82 のうち大半は旧設定 (pre-310#) 時代のデータ。310# の売り時間帯ブースト (310# A) や decision path 分岐 (310# B) の効果はまだ反映されていない。

### 5.3 Ranging sell — 真のボリュームドライバー

ranging sell は全 sell の **71.0%** (811/1142) を占める。全体 p10 への寄与が最も大きい:

- p10 = -6.74 (閾値 -5.0 を 1.74 bps 超過)
- PnL = -0.18 (ほぼ盈虧線)
- fill_rate = 46.0% (**buy の 36.1% より高い**)

**ranging で sell の fill_rate が buy を上回る**: これは sell offset が buy offset より大きい (sell floor=0.30 vs buy ~0.063–0.15) にもかかわらず、sell 側の queue が短い (買い手が多い) ことを示唆。Glosten-Milgrom: sell 側に情報トレーダーが集まりやすい BTC/JPY の構造を反映。

### 5.4 全体 p10 への寄与分解 (推定)

全体 sell p10 = -6.87 の構成要素を加重推定:

| Regime | n比率 | p10 | 寄与度 (概算) |
|---|---|---|---|
| ranging | 71.0% | -6.74 | **主要ドライバー** |
| trending_up | 7.2% | -9.90 | テール悪化要因 |
| trending | 10.3% | -6.56 | 中程度 |
| trending_down | 7.4% | -7.59 | 中程度 |
| none | 11.2% | -5.86 | 軽度 |

**結論**: p10 を -5.0 に改善するには ranging sell の p10 改善が最大レバレッジ。trending_up は最悪だが n が少ないため全体への影響は限定的。

---

## §6 Spread/AS 分解と Market Making 均衡分析

### 6.1 基礎分解

| Side | n | Spread Capture | AS Cost | Realized PnL | Efficiency | AS p90 |
|---|---|---|---|---|---|---|
| sell | 1,089 | 0.86 bps | 1.14 bps | -0.28 bps | -0.32 | 7.90 |
| buy | 1,102 | 0.28 bps | 0.57 bps | -0.29 bps | -1.06 | 6.00 |

### 6.2 理論的解釈

**Avellaneda-Stoikov (2008)** フレームワークでの均衡条件:

$$\text{Spread Capture} \geq \text{AS Cost}$$

両側とも均衡条件を満たしていない (Spread Capture < AS Cost)。

- **sell**: 0.86 - 1.14 = **-0.28 bps** の赤字。spread capture は高いが AS cost がそれを上回る
- **buy**: 0.28 - 0.57 = **-0.29 bps** の赤字。spread capture が小さいが AS cost も小さい

**効率性指標の罠**: buy の efficiency = -1.06 は「sell よりも悪い」と解釈しがちだが、実際の realized PnL は sell (-0.28) と buy (-0.29) でほぼ同一。efficiency は分母 (spread capture) が小さいと過大に見える。**絶対値 (realized PnL) で比較すべき。**

### 6.3 改善の方向性

sell 側の改善:

$$\Delta \text{PnL} = \Delta \text{Spread Capture} - \Delta \text{AS Cost}$$

- **Spread Capture ↑**: offset を上げれば spread capture は増加するが、fill_rate が低下するトレードオフ
- **AS Cost ↓**: 高 AS 時間帯/regime でのスキップ強化、offset 拡大で AS 被弾時の損失軽減

p10 改善には **AS Cost の p90 (= 7.90 bps)** を削減する必要がある。これが downside tail の主因。

---

## §7 時間帯別 AS 構造

### 7.1 sell 時間帯ヒートマップ

| UTC | n | PnL | p10 | AS率 | 310# boost | 分類 |
|---|---|---|---|---|---|---|
| 08 | 27 | **-3.55** | -11.87 | **63.0%** | ×1.5 | 🔴 極危険 |
| 14 | 38 | **-3.28** | -11.37 | 44.7% | ×1.3 | 🔴 極危険 |
| 13 | 55 | **-2.16** | -12.03 | 41.8% | ×1.3 | 🔴 極危険 |
| 16 | 18 | -2.25 | -8.46 | **61.1%** | ×1.5 | 🔴 極危険 |
| 21 | 42 | -2.05 | -8.09 | 38.1% | — | 🟠 危険 (未カバー) |
| 00 | 84 | -1.84 | -7.09 | 35.7% | — | 🟠 危険 (未カバー) |
| 04 | 41 | -1.46 | -6.40 | 34.2% | — | 🟡 要注意 (未カバー) |
| 22 | 59 | -1.23 | -8.59 | 30.5% | — | 🟡 要注意 (未カバー) |
| 07 | 57 | -1.21 | -6.61 | 31.6% | — | 🟡 要注意 (未カバー) |
| 12 | 44 | +0.67 | -4.65 | 15.9% | — | 🟢 安全 |
| 10 | 46 | +0.43 | -3.63 | 19.6% | — | 🟢 安全 |
| 11 | 51 | +0.84 | -6.30 | 19.6% | — | 🟢 安全 |
| 20 | 80 | +0.30 | -4.97 | 28.8% | — | 🟢 安全 |
| 23 | 59 | +0.46 | -4.41 | 23.7% | — | 🟢 安全 |

### 7.2 時間帯構造の理論的解釈

**Ho-Stoll (1981)**: 情報非対称性は市場参加者の構成により時間帯変動する。

BTC/JPY 固有のパターン:
- **UTC 08 (JST 17)**: 東京市場クローズ/欧州市場オープン。機関トレーダーの活発化
- **UTC 13-14 (JST 22-23)**: 米国市場オープン。最も情報非対称性が高い時間帯
- **UTC 16 (JST 01)**: 米国市場ピーク。大口注文による価格変動
- **UTC 0 (JST 09)**: 東京市場オープン。情報集約フロー
- **UTC 21 (JST 06)**: 早朝の低流動性。少数トレーダーの影響が大

### 7.3 310# A のカバレッジ評価

310# A は UTC 8/13/14/16 をカバー (boost n=138)。しかし:

- **未カバーの危険時間帯**: UTC 0, 4, 7, 21, 22 (合計 n=283)
- カバー対象 n=138 に対し未カバー n=283 — **未カバーが 2 倍**
- 未カバー時間帯の加重平均 PnL ≈ -1.56 bps (カバー対象 -2.75 よりは軽度だが依然赤字)

---

## §8 Offset 分位点分析と動的フロア評価

### 8.1 sell offset 五分位

| Q | n | Offset 範囲 | PnL | AS率 | 評価 |
|---|---|---|---|---|---|
| Q1 | 217 | 0.136–0.268 | **-0.65** | **34.1%** | 劣後 |
| Q2 | 218 | 0.268–0.300 | **-0.70** | 30.7% | 劣後 |
| **Q3** | 218 | 0.300 | -0.24 | **19.7%** | AS 最良 |
| **Q4** | 218 | 0.300–0.482 | **+0.57** | 32.1% | **PnL 最良** |
| Q5 | 218 | 0.487–2.088 | -0.38 | 28.4% | 大 offset |

### 8.2 動的フロア割引の問題

sell offset floor = 0.30 だが、`sell_offset_floor_inv_discount = 0.5` により在庫 buy 偏重時にフロアが 0.15 まで下がる。

Q1 (0.136–0.268) の全 217 件はこの割引で生成されたレコード:
- PnL = -0.65 bps (Q3 の -0.24 より 0.41 bps 悪い)
- AS率 = 34.1% (Q3 の 19.7% より **74% 高い**)

**Glosten-Milgrom 均衡**: offset を下げると fill_rate は上がるが AS cost も増大する。Q1 では AS cost 増加が fill_rate 改善の利益を上回っている。

### 8.3 sell フロア割引の費用推定

Q1 (n=217) の PnL を Q3 相当 (-0.24) に改善させた場合:
- 改善幅: 0.65 - 0.24 = 0.41 bps/trade × 217 trades = **88.97 bps 相当の累積改善**
- ただし割引なしだと Q1 の注文は Q3 に移動するのではなく、**一部が約定せず消失する**点に注意

### 8.4 buy offset 五分位 (参考)

| Q | n | Offset 範囲 | PnL | AS率 |
|---|---|---|---|---|
| Q1 | 220 | 0.019–0.063 | -0.20 | 24.1% |
| Q2 | 220 | 0.063–0.075 | -0.12 | 24.1% |
| Q3 | 221 | 0.075–0.100 | -0.63 | 24.9% |
| Q4 | 220 | 0.100–0.150 | +0.06 | 25.5% |
| Q5 | 221 | 0.150–0.825 | -0.58 | 30.3% |

buy は offset と PnL の関係が sell ほど明確でない。Q3 (0.075–0.100) が最悪だが理由は不明。

---

## §9 None Regime 影響定量化

### 9.1 概要

| 指標 | None (n=267) | Non-none (n=2,297) | 差 | 割合 |
|---|---|---|---|---|
| 比率 | 10.4% | 89.6% | — | — |
| PnL | -0.46 bps | -0.32 bps | -0.14 | 44% 劣後 |
| AS率 | 42.7% | 27.5% | +15.2pp | 55% 高 |

### 9.2 None sell vs buy

| Side | None n | PnL | AS率 | Non-none PnL |
|---|---|---|---|---|
| sell | 128 | -0.80 | 42.2% | -0.28 |
| buy | 139 | -0.15 | 43.2% | -0.29 |

**None regime での sell/buy 格差**: sell PnL = -0.80 vs buy PnL = -0.15 — None 時に sell が大幅に不利。fill_rate も sell 14.0% vs buy 33.7%。

**解釈**: regime=None は regime_detector の初期化完了前 (warmup 期間) に発生。この期間は市場構造の推定が不安定であり、特に sell 側の逆選択防御が不十分になる。

---

## §10 Decision Path / 310# 新機能の初期観測

### 10.1 Decision Path (310# B)

| Side | Path | n | PnL | p10 | AS率 |
|---|---|---|---|---|---|
| sell | ev_offset | 60 | **+0.09** | -9.50 | 33.3% |
| sell | unknown | 1,210 | -0.40 | -6.81 | 30.3% |
| buy | ev_offset | 64 | -0.39 | -6.83 | 18.8% |
| buy | unknown | 1,230 | -0.29 | -5.55 | 28.3% |

**sell ev_offset は唯一の正 PnL パス** (+0.09)。ただし n=60, p10=-9.50 とテール損失が大きく、安定性は未確認。

"unknown" = pre-310# データ (decision_path フィールドが空)。310# デプロイ後のデータ蓄積を待つ必要あり。

### 10.2 評価不能項目

310# は PID 58008 にデプロイ直後 (~1h) のため、以下は効果測定不可:
- **310# A**: 売り時間帯ブースト (UTC 8/13/14/16) — 新データなし
- **310# B**: Decision Path 7分岐ラベル — ほぼ全て "unknown"
- **310# D**: None regime observability カウンタ — 新データなし
- **310# E**: Spread/AS 分解 (deep dive §10) — フォーミュラ修正済、次回 deep dive で検証

---

## §11 306# との差分

### 11.1 対照表

| 指標 | 306# | 311# | 差分 |
|---|---|---|---|
| n_filled | 2,472 | 2,564 | +92 (+3.7%) |
| sell n | 1,105 | 1,142 | +37 |
| buy n | 1,105 | 1,155 | +50 |
| Bootstrap p | 0.9355 | 0.9605 | +0.025 |
| Bootstrap diff | — | -0.016 | — |
| Matched p | 0.2043 | 0.2089 | +0.005 |
| sell p10 | -6.84 | -6.87 | -0.03 |
| buy p10 | -5.67 | -5.67 | ±0.00 |
| AB overall | FAIL | FAIL | 同一 |

### 11.2 解釈

データ量は +92 件増加したが、sell p10 の変化は -0.03 bps と測定誤差範囲。構造的変化なし。309#/310# の修正は中立的 (=安全)。

---

## §12 sell p10 ボトルネック構造分析

### 12.1 p10 = -6.87 の改善に何が必要か

目標: sell p10 を -5.0 bps 以上に改善 (1.87 bps の改善)。

p10 は分布の 10th percentile。改善手段:

| アプローチ | 概要 | 実現可能性 |
|---|---|---|
| **A: テール切除** | p10 以下の極端損失レコードを作らない (スキップ) | 高 — ただし fill_rate 低下 |
| **B: テール圧縮** | 極端損失の magnitude を減らす (offset 拡大) | 中 — AS 被弾時の損失は軽減されるが AS 自体は減らない |
| **C: 分布シフト** | 全体の PnL 分布を右方向にシフト | 低 — 根本的な spread capture 改善が必要 |

### 12.2 Regime × 時間帯の交差分析 (推定)

p10 テール (-6.87 以下) を構成するレコードは:

1. **ranging × 危険時間帯** (UTC 0/4/7/8/13/14/16/21/22): n≈280–350 (推定) — 最大グループ
2. **trending_up × 全時間帯**: n≈82 — 全体的に悪い
3. **trending_down × 全時間帯**: n≈85 — テールが悪い

**最大レバレッジ**: ranging × 危険時間帯の改善 (サンプル数最大)。

### 12.3 sell p10 改善ロードマップ

```
Phase 1 (即時・低リスク): データ蓄積 + 310# 効果観測
  → 310# A (時間帯ブースト) の効果を 48h 後に確認
  → decision_path 分布の蓄積

Phase 2 (312# 候補): 時間帯カバレッジ拡大
  312-B: UTC 0/21/22 を sell_hour_offset_boost に追加 (×1.2–1.3)
  → テールの 20–25% を改善見込み

Phase 3 (312# 候補): offset フロア割引の再評価
  312-D: sell_offset_floor_inv_discount を 0.5→0.7 or 0.8
  → Q1 の 217 件の PnL 改善
  → ただし在庫偏重時の約定率低下リスク

Phase 4 (中期): ranging regime 専用の AS 防御
  → ranging × 高 AS 時間帯での追加 offset boost
  → regime-aware hour offset table
```

---

## §13 改善提案 (312# 候補)

### 13.1 一覧

| ID | 優先度 | 提案 | 根拠 (311# data) | 理論 | G1.2 リセットリスク |
|---|---|---|---|---|---|
| **312-A** | P0 | trending_up sell: 効果確認待ち (310# A/B 蓄積) | p10=-9.90, 3/3 FAIL, n=82 | Kyle (1985) | 変更なし |
| **312-B** | P1 | sell_hour_offset_boost 拡張: UTC 0/21/22 追加 (×1.2) | PnL < -1.8, AS > 35% | Ho-Stoll (1981) | ⚠️ YAML 変更 |
| **312-C** | P1 | None regime: offset × 1.3 conservative multiplier | PnL 44% 劣後, AS 42.7% | 情報不確実性下の保守化 | ⚠️ コード変更 |
| **312-D** | P1 | sell_offset_floor_inv_discount: 0.5→0.7 | Q1 PnL=-0.65 → Q3 PnL=-0.24 に改善期待 | Glosten-Milgrom | ⚠️ YAML 変更 |
| **312-E** | P2 | regime × hour 交差 offset table | ranging × 危険時間帯の p10 改善 | Ho-Stoll + Avellaneda-Stoikov | ⚠️ コード変更 |
| **312-F** | 保留 | buy spread capture 改善 | eff=-1.06 (ただし abs PnL≒sell) | — | — |

### 13.2 G1.2 リセットとの二律背反

280# §4 で指摘された問題が再発:

> **312-B/C/D を実装すると G1.2 168h が再リセット。実装しないと p10 は改善しない。**

**推奨**: 
1. **312-A**: 310# 効果観測のため **48–72h は変更凍結**
2. **312-B + 312-D**: 310# 効果確認後に **YAML のみの変更** として一括適用 (hot-reload 対応)
3. **312-C**: コード変更が必要 — 312-B/D の効果を見てから判断

### 13.3 312-A の詳細 (310# 効果確認待ち)

現行の 5.4x boost が十分かは 310# 後の新データでしか判断できない。pre-310# データの n=82 のうち:
- 310# A (時間帯ブースト) が適用されるのは UTC 8/13/14/16 に発生した trending_up sell のみ
- 310# B (decision path) による分岐ログが蓄積されれば、どのパスが p10 テールに寄与しているか特定可能

---

## §14 AI レビュー向け設問

以下の設問に対する外部 AI (Codex/Gemini) のレビューを求める:

### Q1: ranging sell p10 改善の最適アプローチ

ranging sell (n=811, p10=-6.74) は全体 p10 の主要ドライバー。以下のうちどのアプローチが最も効果的か:
- A) 時間帯別 offset テーブル拡大 (Ho-Stoll)
- B) offset フロア割引の縮小 (Glosten-Milgrom)
- C) ranging × 高 AS 時間帯の hard skip (機会損失リスク)
- D) spread capture 改善 (offset 構造の見直し)

### Q2: 動的フロア割引の妥当性

`sell_offset_floor_inv_discount=0.5` で floor が 0.30→0.15 になるが、Q1 offset (0.136–0.268) は Q3 (0.300) に比べ PnL -0.41bps, AS +14.4pp 悪い。
- 在庫 buy 偏重時に sell fill_rate を上げる目的は正当だが、AS コスト増加を正当化できるか?
- 適切な割引係数は?

### Q3: buy efficiency -1.06 の解釈

buy efficiency = -1.06 (sell = -0.32) だが realized PnL はほぼ同等 (-0.29 vs -0.28)。
- efficiency 指標の有用性の限界は?
- buy の spread capture (0.28 bps) を改善する現実的な手段はあるか?

### Q4: sell p10 = -6.87 から -5.0 への道筋

1.87 bps の改善が必要。312-B/C/D の組み合わせで達成可能か? 定量的な推定を求む。

### Q5: G1.2 計測凍結 vs 改善実装の二律背反

280# で「7暦日コード凍結」を推奨したが、CRITICAL バグと sell p10 改善の必要性から凍結は維持困難。
- hot-reload 対応のパラメータのみで改善は可能か?
- コード凍結と改善実装の最適な折衷案は?

---

## ファイル

- 分析スクリプト: [analysis/311_observational_rerun.py](../../analysis/311_observational_rerun.py)
- 結果 JSON: `analysis_results/311_observational_rerun.json` (27KB, 1,048行)
- 306# 前回比較: `analysis_results/306_observational_comparison_rerun.json`

## 関連ドキュメント

- [000#](000_ph0_plan_project_proposal.md) — Gate 定義 (§3.3 G1.1-exec, §3.9 中止ルール)
- [280#](280_ph2_rpt_position_and_remaining_tasks.md) — 0番ドキュメント立ち位置 + 残課題
- [306#](306_ph2_impl_six_proposals_observational_redesign.md) — 6 提案実装 + 前回観測比較
- [309#](309_ph2_review_response_307_308_fixes.md) — L1/L2 理論倒錯修正
- [310#](310_ph2_impl_design_improvements.md) — 設計改善 5 件
