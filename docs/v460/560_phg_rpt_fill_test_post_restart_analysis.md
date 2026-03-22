# 560# Fill Test 再起動後パフォーマンス分析 (3/12–3/22)

| 項目 | 値 |
|------|-----|
| 番号 | 560# |
| 種別 | report |
| 対象期間 | 2026-03-12 ~ 2026-03-22 (11日間) |
| 分析ツール | `analyze_fill_logs.py`, `stopgap_daily_report.py`, `tail_loss_analysis.py`, `side_regime_dashboard.py` |
| 再起動契機 | OHLCV データ更新停止に伴う SAC retrain 連続失敗 (→ 553#–555# で修正) |

---

## 1. 全体サマリ

| 指標 | 値 | 評価 |
|------|-----|------|
| 総注文数 | 6,006 | |
| 約定数 | 1,590 | |
| Fill Rate | 26.5% | 全期間 30.4% より低下 |
| 平均 PnL (30s) | **-0.31 bps** | **マイナス** |
| 累積 PnL (30s) | **-489.1 bps** | 11日で約 -490 bps |
| Adverse Selection 率 | 27.4% | |
| AS 平均損失 | -7.24 bps | |
| Non-AS 平均 PnL | +2.31 bps | Non-AS は健全 |
| VG トリガー率 | 35.0% | |
| Queue Wait 中央値 | 11.2s | |
| Spread 中央値 | 1.88 bps | |

**Key Finding**: Non-AS 取引は +2.31 bps で十分プラスだが、AS 取引の -7.24 bps × 27.4% がそれを上回り、全体を -0.31 bps に引き下げている。

---

## 2. Side 別

| Side | Total | Filled | Rate | avg PnL30 | p10 | p05 | Profitable% | AS率 |
|------|-------|--------|------|-----------|-----|-----|-------------|------|
| buy | 3,178 | 840 | 26.4% | **-0.43** | -4.90 | -7.21 | 47.0% | 22.4% |
| sell | 2,739 | 750 | 27.4% | **-0.17** | -8.81 | -12.49 | 50.3% | 33.1% |

- **sell 側の AS 率が 33.1%** と高く、テールリスク (p10=-8.81) も buy の 1.8 倍
- ただし sell の avg PnL は buy より良好 (-0.17 vs -0.43)
- sell の profitable 率 50.3% → AS さえ制御できれば黒字転換可能

---

## 3. Regime 別

| Regime | Total | Filled | Rate | avg PnL30 |
|--------|-------|--------|------|-----------|
| ranging | 5,482 | 1,413 | 25.8% | -0.35 |
| trending_down | 264 | 92 | 34.8% | **+0.71** |
| trending_up | 260 | 85 | 32.7% | **-0.77** |

- **trending_down で唯一プラス** (+0.71 bps)
- trending_up がワースト (-0.77 bps): 上昇トレンド中の sell AS リスク

---

## 4. 日別推移

| 日付 | 注文 | 約定 | Rate | avg PnL30 | sum PnL30 | 備考 |
|------|------|------|------|-----------|-----------|------|
| 3/12 | 553 | 219 | 40% | **+0.32** | +69.6 | 好調 |
| 3/13 | 625 | 178 | 28% | -0.79 | -139.8 | 急転、AS 増加 |
| 3/14 | 602 | 159 | 26% | -0.47 | -75.2 | |
| 3/15 | 370 | 62 | 17% | **-1.44** | -89.1 | ワースト: low fill + high loss |
| 3/16 | 607 | 83 | 14% | +0.86 | +71.7 | 低約定だが質は良好 |
| 3/17 | 635 | 109 | 17% | -0.84 | -91.6 | |
| 3/18 | 453 | 108 | 24% | -0.13 | -13.5 | 改善兆候 |
| 3/19 | 338 | 115 | 34% | -1.26 | -144.4 | ワースト2: high fill + high AS |
| 3/20 | 546 | 287 | **53%** | +0.03 | +7.9 | 高約定・ほぼ収支均衡 |
| 3/21 | 697 | 130 | 19% | -0.15 | -19.1 | |
| 3/22 | 580 | 140 | 24% | -0.47 | -65.6 | |

**パターン**: fill rate と avg PnL に弱い負の相関 — 高約定日 (3/19, 3/20) は AS にさらされやすい。3/15 は低約定 + 高損失の二重苦で最悪日。

---

## 5. Cancel Reason 分布

| Reason | Count | 比率 | 評価 |
|--------|-------|------|------|
| ranging_low_vol_skip | 718 | 16.3% | 低ボラ時の適切なスキップ |
| sell_dynamic_kill | 653 | 14.8% | sell 防御が積極的 |
| skip_gate | 624 | 14.1% | ML skip gate 健全稼働 |
| preflight_insufficient | 566 | 12.8% | 残高/プリフライト不足 |
| spread_too_narrow | 463 | 10.5% | タイトスプレッド環境 |
| timeout | 360 | 8.2% | |
| no_feasible_quote | 258 | 5.8% | |
| route_to_kill_deadlock | 152 | 3.4% | kill→deadlock パス |
| stale_adverse_drift | 141 | 3.2% | |
| final_clamp_hard_skip | 114 | 2.6% | |

- **sell_dynamic_kill (14.8%)** + **buy_dynamic_kill (1.9%)** = 16.7% → kill 発動率が高い
- **spread_too_narrow (10.5%)** — マーケット競争の激化を示唆

---

## 6. Clamp Saturation 分析

| Side | Clamp率 | pre_clamp avg | effective avg | Clamped PnL | Unclamped PnL |
|------|---------|---------------|---------------|-------------|---------------|
| buy | 99% | 0.3452 | 0.2440 | -0.61 bps | +1.35 bps |
| sell | 99% | 0.3154 | 0.2321 | -0.44 bps | +0.76 bps |

**Critical**: 99% clamp 飽和 — ほぼ全約定が ceiling offset で発注。  
- unclamped は +1.35/+0.76 で健全だが、clamped が -0.61/-0.44
- **offset ceiling の引き上げ**が PnL 改善の最大レバー

---

## 7. テール損失分析 (p10 以下)

### 7.1 Sell テール

| 指標 | 値 |
|------|-----|
| テール閾値 | -8.81 bps |
| テール平均 | -15.44 bps |
| テール p5 (最悪) | -26.94 bps |
| AS 過大表現 | 3.02x (テール AS 100%) |

**時間帯過大表現 (top-3)**:
- UTC 13 (JST 22時): 2.50x
- UTC 00 (JST 09時): 2.16x
- UTC 19 (JST 04時): 1.84x

### 7.2 Buy テール

| 指標 | 値 |
|------|-----|
| テール閾値 | -4.90 bps |
| テール平均 | -9.12 bps |
| テール p5 (最悪) | -14.83 bps |
| AS 過大表現 | 4.47x (テール AS 100%) |

**時間帯過大表現 (top-3)**:
- UTC 14 (JST 23時): **4.06x**
- UTC 13 (JST 22時): 2.73x
- UTC 12 (JST 21時): 1.78x

### 7.3 テール集中時間帯

**Buy/Sell 共通で危険な時間帯**: JST 22-23時 (UTC 13-14)

| UTC | JST | Buy overrep | Sell overrep | 評価 |
|-----|-----|-------------|--------------|------|
| 13 | 22 | 2.73x | 2.50x | **buy/sell 両方で高リスク** |
| 14 | 23 | **4.06x** | 1.61x | **buy 最大リスク** |
| 00 | 09 | - | 2.16x | sell リスク |

---

## 8. Cross-Venue 分析

| 指標 | 値 |
|------|-----|
| CV 適用率 | 21.9% (348/1590 fills) |
| Veto 数 | 0 |
| buy widen PnL | +0.03 bps |
| sell widen PnL | **-1.10 bps** |

- **sell widen が -1.10 bps** → 他取引所スプレッドに合わせた widen が逆効果
- cap_hit が多い (buy: 78, sell: 64) → CV ceiling は有効に機能

---

## 9. 最悪約定 (Top 5 損失)

| # | Side | PnL (30s) |
|---|------|-----------|
| 1 | buy | -72.65 bps |
| 2 | ? | -51.90 bps |
| 3 | ? | -40.56 bps |
| 4 | buy | -33.06 bps |
| 5 | buy | -27.51 bps |

- Top 5 で **-225.69 bps** (全損失 -3,591.83 bps の 6.3%)
- buy 側に集中

---

## 10. 改善提案 (優先度順)

### P0: Offset Ceiling 見直し (Clamp 飽和問題)
- 99% clamp 飽和は offset が低すぎることを示す
- unclamped (+1.35 bps) vs clamped (-0.61 bps) の差が大きい
- **ceiling 引き上げで spread PnL 改善の余地が大きい**

### P1: JST 22-23 時テール集中対策
- UTC 13-14 (JST 22-23) が buy/sell 両方でテール過大表現
- 時間帯別 offset 引き上げ or 追加ガードの検討
- `hour_ceiling_mult` (467#) を JST 22-23 に適用する候補

### P2: Sell AS 率改善
- sell AS 33.1% は buy 22.4% の 1.5 倍
- sell widen (CV) が -1.10 bps で逆効果 → CV sell widen の無効化を検討
- sell_dynamic_kill が 653 回で積極的に稼働中だが AS 制御不十分

### P3: Fill Rate 回復
- 全期間 30.4% → 直近 26.5% に低下
- spread_too_narrow (10.5%) + preflight_insufficient (12.8%) が主因
- 市場環境 (タイトスプレッド) への適応が必要

---

## 11. 分析再現コマンド

```bash
# 包括分析
python scripts/v460/analysis/analyze_fill_logs.py --date-from 2026-03-12 --date-to 2026-03-22

# 日次ヘルスレポート
python scripts/v460/analysis/stopgap_daily_report.py --date-from 2026-03-12 --date-to 2026-03-22 --json

# テール損失分析
python scripts/v460/analysis/tail_loss_analysis.py --date-from 2026-03-12 --date-to 2026-03-22

# Side×Regime ダッシュボード
python scripts/v460/analysis/side_regime_dashboard.py --json
```
