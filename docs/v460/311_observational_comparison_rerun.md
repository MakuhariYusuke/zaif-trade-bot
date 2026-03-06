# 311# 観測比較 — 309#/310# 理論修正検証

**日時**: 2025-03-07  
**データ**: fill_records (total=7183, filled=2564, sell=1270, buy=1294)  
**前提**: 310# (dcc3064a8) デプロイ直後のため、データの大部分は pre-310# 期間

## 目的

1. 306# で実施した 299# 観測比較を再実行し、309#/310# の理論修正が悪影響を与えていないか検証
2. データ駆動で改善点を特定

## 検証結果

### §1 理論修正の検証 — ✅ PASS

| 指標 | 306# | 311# | 判定 |
|---|---|---|---|
| Bootstrap p | 0.9355 | 0.9605 | 非有意 ✅ |
| Matched p | 0.2043 | 0.2089 | 非有意 ✅ |
| sell avg_pnl | -0.33 bps | -0.33 bps | 同一 ✅ |
| sell p10 | -6.84 bps | -6.87 bps | 同一 ✅ |
| Cliff's delta | negligible | negligible | 同一 ✅ |

**結論**: 309# (L1/L2 修正) + 310# (設計改善) は sell vs buy PnL 差に統計的有意な変化を与えていない。

### §2 AB 判定 — FAIL

**全体** (None 除外):  
- ✅ fill_rate: 40.2% vs 40.5%
- ✅ avg_pnl30: -0.33 vs -0.31 bps
- ❌ downside_p10: **-6.87** vs -5.67 bps (閾値 -5.0)

**全体** (None 込み):  
- ❌ fill_rate: 33.8% vs 39.6% (14.6% 劣化)
- ✅ avg_pnl30: -0.38 vs -0.29 bps
- ❌ downside_p10: -6.85 vs -5.66 bps

### §3 Regime 別

| Regime | sell n | sell PnL | sell p10 | buy p10 | 判定 |
|---|---|---|---|---|---|
| **trending_up** | 82 | **-1.15** | **-9.90** | -5.69 | **3/3 FAIL** |
| trending_down | 85 | -0.49 | -7.59 | -6.23 | p10 FAIL |
| trending | 118 | -0.66 | -6.56 | -6.96 | FR+p10 FAIL |
| ranging | 811 | -0.18 | -6.74 | -5.43 | p10 FAIL |
| none | 128 | -0.80 | -5.86 | -4.63 | FR+p10 FAIL |

**trending_up sell が最悪**: fill_rate 18.4% / PnL -1.15 / p10 -9.90 の三重苦。

### §4 Spread/AS 分解

| Side | Spread Capture | AS Cost | Realized PnL | Efficiency |
|---|---|---|---|---|
| sell | 0.86 bps | 1.14 bps | -0.28 bps | -0.32 |
| buy | 0.28 bps | 0.57 bps | -0.29 bps | -1.06 |

両側とも AS cost > spread capture。realized PnL はほぼ同等 (-0.28 vs -0.29)。

### §5 売り時間帯 (UTC)

危険時間帯 (sell PnL < -1.5 bps AND AS > 30%):

| UTC | n | PnL | p10 | AS率 | 310# boost |
|---|---|---|---|---|---|
| 08 | 27 | -3.55 | -11.87 | 63.0% | × 1.5 |
| 14 | 38 | -3.28 | -11.37 | 44.7% | × 1.3 |
| 13 | 55 | -2.16 | -12.03 | 41.8% | × 1.3 |
| 16 | 18 | -2.25 | -8.46 | 61.1% | × 1.5 |
| 00 | 84 | -1.84 | -7.09 | 35.7% | — |
| 21 | 42 | -2.05 | -8.09 | 38.1% | — |
| 22 | 59 | -1.23 | -8.59 | 30.5% | — |

310# で対象とした UTC 8/13/14/16 は正しいが、UTC 0/21/22 もカバー対象候補。

### §6 Offset 五分位

| Sell Q | Offset 範囲 | PnL | AS率 | 評価 |
|---|---|---|---|---|
| Q1 | 0.136–0.268 | -0.65 | 34.1% | 劣後 |
| Q2 | 0.268–0.300 | -0.70 | 30.7% | 劣後 |
| Q3 | 0.300 | -0.24 | 19.7% | AS 最良 |
| Q4 | 0.300–0.482 | **+0.57** | 32.1% | **最良** |
| Q5 | 0.487–2.088 | -0.38 | 28.4% | 大 offset |

Q4 (offset 0.30–0.48) のみ正 PnL。Q1–Q2 (< 0.30) は明確に劣後。

### §7 Decision Path (310# B) — データ不足

- sell "ev_offset": n=60, PnL=+0.09 (正値!)
- buy "ev_offset": n=64, PnL=-0.39
- 残り全て "unknown" (pre-310# データ)

310# デプロイ後のデータ蓄積を待つ必要あり。

### §8 None Regime (310# D)

- 全 filled の 10.4% (267/2564)
- None PnL: -0.46 bps (non-none: -0.32 bps) — 44% 劣後
- None AS: 42.7% (non-none: 27.5%) — 55% 高
- None sell fill_rate: 14.0% vs buy 33.7% — 58.4% 劣化

### §9 評価不能項目

310# は PID 58008 にデプロイ直後 (~1h) のため、以下は効果測定不可:
- 310# A: 売り時間帯ブースト → 新データ蓄積待ち
- 310# B: Decision Path ラベル → ほぼ全て "unknown"
- 310# D: None regime observability → 新データ蓄積待ち

## 改善提案

| ID | 優先度 | 提案 | 根拠 | 理論 |
|---|---|---|---|---|
| 312-A | P0 | trending_up sell offset boost 強化 | p10=-9.90, 3/3 FAIL | Kyle (1985): AS 下の情報レント |
| 312-B | P1 | sell_hour_offset_boost 拡張 (UTC 0/21/22 追加) | PnL < -1.0, AS > 30% | Ho-Stoll (1981) |
| 312-C | P1 | None regime conservative mode (offset ×1.3) | PnL 44% 劣後, AS 55% 高 | 情報不確実性下の保守化 |
| 312-D | P2 | sell offset Q1 分析 (floor 見直し) | Q1-Q2 明確劣後 | Glosten-Milgrom offset-AS trade-off |
| 312-E | 保留 | buy spread capture 改善 | eff -1.06 だが abs PnL≒sell | 追加データ後 |

### 312-A の詳細

trending_up sell は全セグメント最悪。現行設定:
- `skip_sell_trending: true` + `skip_sell_trending_up_only: true` → ゲート有効
- `trending_sell_as_offset_enabled: true` + factor=3.0 → soft skip
- `trending_up_sell_offset_boost: 1.8` → 追加ブースト
- 実効 boost: base × 3.0 × 1.8 = **5.4x**

5.4x で不十分であることが示唆される。ただし履歴データの多くは旧設定時のもの。
新データ蓄積後に再評価し、改善方向を決定する。

## ファイル

- スクリプト: `analysis/311_observational_rerun.py`
- 結果 JSON: `analysis_results/311_observational_rerun.json`
- 306# 比較: `analysis_results/306_observational_comparison_rerun.json`
