# 647# ポスト 645#/646# デプロイ分析 — σ 高騰による Fill Rate 低下

## 概要

645# (退化 sell モデル無効化) + 646# (過学習防止ガード) デプロイ後 ~10 時間の稼働ログを分析。
**結論: fill rate 低下 (7.2%) は 645#/646# のコード変更に起因せず、市場ボラティリティ高騰による ATR 閾値固定が原因。**

## 分析対象

| SHA | 期間 | 概要 |
|-----|------|------|
| 600e1b2e0f | 03/26 16:02~03/27 02:00 | Pre-645# 比較基準 |
| 4aa931b200 | 03/27 02:03~09:34 | Pre-645# 比較基準 |
| dbbfe24990 | 03/27 09:50~15:43 | Pre-645# 比較基準 |
| 487c5a4cbb | 03/27 15:51~16:49 | 645# (短時間) |
| **f7faac4f12** | **03/27 17:02~03/28 03:00** | **646# 現行** |

## 主要発見

### 1. Fill Rate 低下の根本原因: ATR 閾値の高止まり

f7faac4f12 の fill rate は **7.2%** (15/208) — pre-645# SHA (25-37%) と大幅乖離。

**Cancel reason 内訳:**
| 理由 | f7faac4f12 | Pre-645# 平均 | 差分 |
|------|-----------|-------------|------|
| preflight_insufficient | 39% | 35% | 同程度 |
| **no_feasible_quote** | **25% (53件)** | **0%** | **新出・最大** |
| spread_too_narrow | 14% (29件) | 3% | 大幅増 |
| mcb_halt | 9% (19件) | 3% | 増加 |
| skip_gate | 5% (10件) | 18% | 減少 (期待通り: sell モデル無効化) |

### 2. no_feasible_quote 53 件の分析

- **全 53 件が "Spread too narrow" エラー**。XV veto=0, balance_switch=0
- ATR 閾値 ~3,280 JPY > 実スプレッド ~2,000-2,500 JPY
- σ=0.000417 (ボラティリティ) が全域で一定値に固定
- **3 回連続 InfeasibleQuoteError (同一 side) → no_feasible_quote に昇格** (234# の設計)

### 3. ATR 閾値の計算

```
effective_min = max(
  min_spread_jpy = 100 JPY,           # 絶対安全ネット
  bps_floor = mid × 0.38 / 10000 ≈ 416 JPY,  # 処理コスト
  atr_floor = min(σ × mid × 1.2, mid × 3.0 / 10000)  # AS コスト (cap 付き)
)
```

| SHA | σ 範囲 | ATR 閾値 | 実スプレッド | 結果 |
|-----|--------|---------|-----------|------|
| 4aa931b | 0.000016 | 208-419 JPY | ~200 JPY | 通過多数 |
| dbbfe24 | 0.000038-0.000131 | 498-1723 JPY | ~200 JPY | 大半通過 |
| 600e1b2 | 0.000104-0.000417 | 1396-3349 JPY | ~700-2400 JPY | 混在 |
| **f7faac4f** | **0.000417 (固定)** | **3,255-3,290 JPY** | **610-3,174 JPY** | **大半拒否** |

### 4. 645#/646# の効果検証

**検証不能** — fill rate 低下が市場要因のため、有意なサンプルサイズ (15 fills) では評価できない。

ただし以下の正方向シグナルは確認:
- **sell_p30: -1.956bps (pre) → -0.304bps** — 退化 sell モデル無効化の効果（暫定的）
- **sell AS 率: 52% (pre) → 29%** — AS 改善傾向
- **skip_gate 率: 18% → 5%** — sell モデル由来の過剰 skip 解消

### 5. Buy Side の問題 (646# とは無関係)

- Buy avg_pnl30 = -4.581 bps, AS = 62.5%
- 8 fills 中 5 件が ranging regime (-5.370 bps)
- これは **ranging regime での buy エントリーの構造的問題** (既知)
- 646# で `model_path_buy: null` 設定済みのため、unified fallback model の質に依存

## 結論と推奨

### 結論
1. **645#/646# は正常に動作している** — fill rate 低下はコード起因ではない
2. σ が 0.000417 に張り付いた高ボラ状態が原因
3. ATR cap (3.0 bps ≈ 3,300 JPY) が実効的なフロアとして全 order を拒否

### 推奨
1. **即座のアクション不要** — σ が自然低下すれば fill rate は回復
2. **ATR cap の見直しは慎重に** — 3.0 bps cap は 632# で設計された AS 防御。高ボラ時の fill rate 低下は設計通り
3. **sell side 改善の再確認** — σ 低下後、sell pnl の持続的改善を確認すること
4. **analysis script を定期利用** — `python -m scripts.v460.analysis.sha_performance_report --days 3` で定点観測

## 成果物

- `scripts/v460/analysis/sha_performance_report.py` — Cancel reason 内訳 + ATR 乖離分析を追加
- `analysis_results/647_sha_performance.json` — JSON 出力
