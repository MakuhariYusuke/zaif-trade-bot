# 317# 観測比較実験報告

> **種別**: rpt  
> **起票**: 2026-03-07  
> **起源**: 314# T0-T2 修正後の全データ観測比較実験  
> **コミット**: `72f763d6f` (文書番号体系整理 + 実験実施)

---

## §1 実験概要

314# で T0 (spread capture SHA フィルタ)、T1 (ratio セマンティクス調査)、T2 (mid-distance quintile 分析) を完了した後、
修正済みスクリプト `analysis/311_observational_rerun.py` を全データおよび dcc3064 (310# 改善コード) に対して実行。

### §1.1 実行コマンド

```powershell
# 全データ (ベースライン)
python analysis/311_observational_rerun.py

# dcc3064 限定 (310# 改善コード)
python analysis/311_observational_rerun.py --git-sha dcc3064
```

### §1.2 出力ファイル

| ファイル | 内容 |
|---|---|
| `analysis_results/317_observation_full.txt` | 全 SHA 実験出力 (n=2575 filled) |
| `analysis_results/317_observation_dcc3064.txt` | dcc3064 限定出力 (n=16 filled — 不足) |
| `analysis_results/311_observational_rerun.json` | JSON 構造化結果 |

---

## §2 全データ結果 (n=7254 records, 2575 filled)

### §2.1 A/B 判定: overall=fail

| 指標 | sell | buy | 判定 | 閾値 |
|---|---|---|---|---|
| fill_rate | 40.2% | 39.8% | ✅ OK | |
| avg_pnl30 | -0.3314 | -0.3245 | ✅ OK | > -1.00 |
| downside_p10 | **-6.8735** | -5.6733 | ❌ FAIL | > -5.00 |

- Bootstrap: diff=-0.0069, CI=[-0.5565, +0.5226], p=0.9835 (有意差なし)
- None 含有版も同様に downside_p10 で fail

### §2.2 Regime 別分析

| Regime | sell n | buy n | sell pnl | buy pnl | sell p10 | buy p10 | 判定 |
|---|---|---|---|---|---|---|---|
| none | 128 | 139 | -0.80 | -0.15 | -5.86 | -4.63 | ❌ fill_rate+p10 |
| ranging | 813 | 817 | -0.17 | -0.50 | -6.73 | -5.44 | ❌ p10 |
| trending | 118 | 118 | -0.66 | +0.57 | -6.56 | -6.96 | ❌ fill_rate+p10 |
| trending_down | 88 | 84 | -0.59 | +0.68 | -7.90 | -6.23 | ❌ p10 |
| **trending_up** | **83** | **94** | **-1.16** | **-0.33** | **-9.86** | **-5.69** | **❌ 三重fail** |
| unknown | 46 | 47 | — | — | — | — | insufficient |

**最重要発見**: trending_up sell が全指標最悪 (fill_rate 18.6%, pnl -1.16, p10 -9.86)

### §2.3 Spread / AS 分解

| Side | spread_capture | realized_pnl | AS cost | efficiency |
|---|---|---|---|---|
| SELL | -0.502 bps | -0.379 bps | -0.124 bps | 0.754 |
| BUY | -0.487 bps | -0.306 bps | -0.182 bps | 0.627 |

**両サイドで spread capture が負** — 注文時の mid が fill 時より不利方向に移動 (検出レイテンシバイアス)。
→ S-3 `mid_at_order` フィールド追加で原因特定の道具を整備する動機。

### §2.4 Decision Path

| Side | Path | n | pnl | AS |
|---|---|---|---|---|
| SELL | ev_offset | 66 | -0.004 | 31.8% |
| SELL | unknown | 1210 | -0.399 | 30.2% |
| BUY | ev_offset | 69 | -0.583 | 20.3% |
| BUY | unknown | 1230 | -0.290 | 28.3% |

buy ev_offset の PnL (-0.583) が buy unknown (-0.290) より悪い → S-6 調査動機。

### §2.5 Sell Hour Boost

| 区分 | n | PnL | p10 | AS |
|---|---|---|---|---|
| Boost (UTC 8/13/14/16) | 138 | -2.749 | -11.887 | 49.3% |
| 非 Boost | 1138 | -0.091 | -6.367 | 28.0% |

Boost 時間帯の AS 率 49.3% は非 boost の 1.76 倍。310# で導入した `sell_hour_offset_boost` の効果検証には post-310# データが必要 (§3.1 参照)。

### §2.6 None Regime

| 区分 | n | pnl | AS |
|---|---|---|---|
| None 全体 | 267 (10.4%) | -0.462 | 42.7% |
| None sell | 128 | -0.803 | 42.2% |
| None buy | 139 | -0.149 | 43.2% |
| Non-none | 2308 | -0.328 | 27.5% |

None regime の AS 率 42.7% は non-none (27.5%) の 1.55 倍。売買差も顕著 (sell -0.80 vs buy -0.15)。
→ 318# F5 修正 (Passive MM dead code 修正) で改善を期待。

---

## §3 dcc3064 結果 (n=16 filled — 不十分)

310# 改善コード (dcc3064) 稼働開始から約 3.3 時間時点で 16 fills。統計的に有意な評価には 50+ fills 必要。

### §3.1 蓄積見込み

| 項目 | 値 |
|---|---|
| 観測 fill rate | ~4.6 fills/h |
| 50 fills 到達 | +7.5h (累計 ~11h) |
| 100 fills 目標 | +18h |

### §3.2 次回実行

```powershell
python analysis/311_observational_rerun.py --git-sha dcc3064
```

50+ fills 蓄積後に再実行し、310# コードの効果を定量評価する。

---

## §4 構造的課題の特定

実験結果から以下の構造的課題を特定。詳細な施策は 316# §4 で提案。

| ID | 課題 | 根拠 | 316# 施策 |
|---|---|---|---|
| 1 | downside_p10 全 regime fail | 全行 p10 < -5.0 | S-7 テール分析 |
| 2 | trending_up sell 三重苦 | fill_rate 18.6%, pnl -1.16, p10 -9.86 | S-1 boost 強化 |
| 3 | sell_offset_floor 死亡 | ceiling 0.15 < floor 0.30 | S-5 YAML 整合 |
| 4 | buy ev_offset 逆効果 | pnl -0.583 vs unknown -0.290 | S-6 調査 |
| 5 | spread capture 負 | 両サイド -0.5 bps | S-3 mid_at_order |
| 6 | none sell 最悪 | fill_rate 14%, pnl -0.80 | S-4 → 318# 解決済 |

---

## §5 時間帯別詳細

### §5.1 Sell 最悪時間帯 (AS > 40%)

| UTC | n | PnL | p10 | AS |
|---|---|---|---|---|
| 08 | 27 | -3.546 | -11.872 | **63.0%** |
| 16 | 18 | -2.250 | -8.459 | **61.1%** |
| 14 | 38 | -3.277 | -11.374 | **44.7%** |
| 13 | 55 | -2.156 | -12.033 | **41.8%** |

UTC 8 (JST 17時) と UTC 16 (JST 25時/翌 1 時) は AS 60%+ の危険時間帯。

### §5.2 Sell 最良時間帯 (PnL > 0)

| UTC | n | PnL | p10 | AS |
|---|---|---|---|---|
| 11 | 51 | +0.839 | -6.299 | 19.6% |
| 15 | 61 | +0.795 | -8.060 | 31.1% |
| 05 | 64 | +0.710 | -5.314 | 28.1% |
| 12 | 44 | +0.667 | -4.647 | 15.9% |

---

## §6 関連ドキュメント

| # | 関係 |
|---|---|
| 314 | T0-T2 修正の元タスク |
| 315 | Ceiling/Ratio Semantics 調査 (T1 結果) |
| 316 | 本実験の解釈 + 先行施策 S-1〜S-7 の提案 |
| 318 | none regime 修正 (§2.6 の none 問題への対処) |
| 310 | 現行稼働コード (dcc3064) |
| 306 | 観測比較スクリプト設計元 |
