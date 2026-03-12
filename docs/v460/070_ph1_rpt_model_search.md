# 070# ph1 レポート: 徹底モデルサーチ

| key | value |
|---|---|
| 番号 | 070 |
| フェーズ | ph1 |
| 種別 | rpt |
| 参照 | `scripts/v460/ml/run_070_model_search.py`, `scripts/v460/ml/run_070_deep_analysis.py`, `scripts/v460/ml/run_070_final_analysis.py` |
| 作成日 | 2026-02-16 |
| テスト | 658 passed |
| 目的 | ph2 fill test 再投入に相応しい「儲かるモデル」を徹底探索 |

---

## §0 エグゼクティブサマリ

**結論: 現在のデータ量 (284 AS-labeled, 373 filled) では、いかなる ML モデルもランダム以上の予測力を持たない。**

- 72+ AS 分類器、20 PnL 回帰器、12 ルールベース、6 side 別モデルを網羅的にスイープ
- **全分類器 ROC-AUC ≤ 0.54** (OOF)、**全回帰器 IC ≈ 0**
- 唯一の正 PnL セグメント: queue_wait 60-120s (+0.84 bps) — ただし事前予測不可
- ラウンドトリップ再分析: **平均 -1.10 bps、勝率 44.4%** (以前の +10.3 bps 報告と矛盾)
- SNR (signal-to-noise ratio) = 0.62 / 5.64 = **0.11** — ML が機能する閾値を大幅に下回る

### 推奨アクション

| 優先度 | アクション | 根拠 |
|--------|-----------|------|
| **P0** | データ蓄積を最優先 | 現 284 → 800+ サンプルで ML の信号検出が初めて可能に |
| P1 | spread_offset 拡大 (5% → 6-7%) | per-fill エッジ改善、AS 低減 |
| P2 | PnL 回帰器 (candidate3) の試験的統合 | IC=0.06 は微弱だが唯一の非ゼロシグナル |
| P3 | 現行 Two-Tier SkipGate 継続 | インフラは健全。モデル差替え可能 |

---

## §1 データ概況

```
全レコード:       491
  約定 (filled):  373 (75.9%)
  キャンセル:      118 (24.1%)
AS ラベル付き:     284 (約定のうち spread_at_order 保有 166)
AS 率:            52.1% (コインフリップ)

PnL (30s horizon):
  mean:   -0.6199 bps
  median: -0.1206 bps
  std:     5.6378 bps
  p5/p95: -8.35 / +6.73 bps

Side 別:
  Buy:  n=192, PnL=-0.301 bps, AS=51.0%
  Sell: n=181, PnL=-0.958 bps, AS=53.2%
```

---

## §2 Part A: ルールベース12戦略

in-sample 全データで各ルールを評価。

| # | Rule | PnL改善(bps) | Skip率 | 備考 |
|---|------|-------------|--------|-----|
| R5 | skip_neg_pnl_hours | +1.47 | 68.1% | **in-sample リーク (未来情報使用)** |
| R1 | skip_all_sell | +0.32 | 48.5% | 頑健。sell PnL が恒常的に悪い |
| R10 | skip_narrow_spread | +0.18 | 22.3% | spread データ不足 (166 件) |
| R3 | skip_bad_hours | +0.17 | 43.2% | WF テストでは不安定 |
| R2 | skip_all_buy | -0.34 | 51.5% | 逆効果 |

**結論**: 正直なルールで最善は「sell 全スキップ」(+0.32 bps)。ただし取引量半減。

---

## §3 Part B: AS 分類器スイープ (72+構成)

3 特徴量セット × 24 モデル構成 = 72 パターン。
Walk-forward (expanding window, min_train=50, step=15, embargo=2)。

### 特徴量セット

| Set | 特徴量数 | サンプル数 | 内容 |
|-----|---------|----------|------|
| base | 10 | 284 | 基本特徴量 (queue_wait, side, hour, edge, spread, offset, regime) |
| enriched | 39 | 166 | base + OB microstucture + trade flow (v2) |
| pnl | 35 | 373 | PnL 関連 enriched 特徴量 |

### Top 10 (OOF skip20% PnL 改善順)

| # | Model | ROC-AUC | skip20%改善(bps) |
|---|-------|---------|-----------------|
| 1 | enriched_LR_L1_C0.1_k5 | 0.500 | +0.454 |
| 2 | enriched_LR_C0.1_k5 | 0.498 | +0.362 |
| 3 | enriched_GB_n50_d2_k12 | 0.447 | +0.302 |
| 4 | enriched_GB_n50_d2_k5 | 0.449 | +0.290 |
| 5 | enriched_LR_C1.0_k5 | 0.502 | +0.288 |
| 6 | pnl_LR_C0.1_k5 | 0.527 | +0.166 |
| 7 | enriched_RF_n50_d3_k12 | 0.514 | +0.103 |
| 8 | pnl_LR_L1_C0.1_k5 | 0.492 | +0.047 |
| 9 | base_LR_C1.0_kAll | 0.461 | +0.023 |
| 10 | pnl_RF_n50_d3_k5 | 0.485 | -0.006 |

**全 ROC-AUC ≤ 0.54 = ランダム水準。skip20% 改善は統計的ノイズの範囲内。**

---

## §4 Part C: PnL 回帰器スイープ (20構成)

OOF IC (Spearman) + skip_neg (予測 PnL < 0 をスキップした場合の改善)。

| # | Model | IC | skip_neg改善(bps) |
|---|-------|----|-----------------|
| 1 | base_Ridge_a10_k5 | 0.0003 | +0.611 |
| 2 | enriched_GBR_n30_d2_k8 | 0.062 | +0.499 |
| 3 | base_Ridge_a1_k5 | -0.0009 | +0.491 |
| 4 | enriched_GBR_n30_d2_k12 | 0.055 | +0.467 |

**GBR は IC ≈ 0.06 と非ゼロだが、金融予測としては「微弱」水準。**

---

## §5 Part D: 詳細 Walk-Forward 検証

上位3候補の fold 別分析。

### §5.1 enriched_LR_C0.1_k5 (AS 分類器)

| 指標 | 値 |
|------|---|
| Folds | 8 |
| OOF有効 | 114 |
| Baseline PnL | -1.063 bps |
| Fold ROC-AUC | 0.55, 0.50, 0.50, 0.50, 0.50, 0.47, 0.50, 0.50 |
| Best threshold | th=0.525, skip=57%, improvement=**-0.077 bps (悪化)** |

**判定: 完全にランダム。実用価値ゼロ。**

### §5.2 enriched_GBR_n30_d2_k8 (PnL 回帰器)

| 指標 | 値 |
|------|---|
| Folds | 16 |
| OOF有効 | 320 |
| Baseline PnL | -0.525 bps |
| Fold IC | 0.049, 0.188, -0.079, 0.075, 0.066, 0.175, -0.092, **0.211**, 0.080, -0.233, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0 |
| IC 崩壊 | Fold 11以降 = 全て 0.0 (定数予測に退化) |
| Best config | th=0.0, skip=75.6%, improvement=+0.54 bps |
| 実用的 th=-0.75 | skip=15%, improvement=+0.064 bps (微小) |

**判定: 前半 fold で微弱信号あり (IC 0.05-0.21)。後半で崩壊。データ追加で再検証の価値あり。**

### §5.3 base_LR_C0.01_kAll (AS 分類器)

| 指標 | 値 |
|------|---|
| Folds | 12 |
| OOF有効 | 232 |
| Baseline PnL | -0.619 bps |
| Fold ROC-AUC | 0.46, 0.44, 0.58, **0.72**, 0.35, 0.44, 0.60, 0.55, 0.34, 0.63, 0.40, 0.34 |
| Best config | th=0.5, skip=65.5%, improvement=+0.14 bps |
| th=0.525 | skip=41%, improvement=**-0.14 bps (悪化)** |

**判定: ROC 高分散。Fold 3 の 0.72 は外れ値。改善は skip 65%+ でのみ正 → 実用不可。**

---

## §6 Deep Analysis: データ構造の探索

### §6.1 時間帯別 PnL

| UTC (JST) | n | PnL(bps) | 正率 | AS率 |
|-----------|---|----------|------|------|
| 5 (14) | 30 | **+1.42** | 63.3% | 36.7% |
| 7 (16) | 17 | +0.82 | 64.7% | 35.3% |
| 0 (9) | 15 | +1.35 | 53.3% | 100% |
| 12 (21) | 17 | +0.34 | 56.0% | 44.0% |
| 17 (2) | 18 | **-3.81** | 33.3% | 57.1% |
| 22 (7) | 12 | -2.00 | 41.7% | 50.0% |

**WF テスト**: 時間フィルタの OOF 改善は +0.05 〜 +0.19 bps (不安定、信頼性低)。

### §6.2 Queue Wait × PnL (最重要発見)

| Wait区間 | n | PnL(bps) | 正率 |
|----------|---|----------|------|
| 5-10s | 65 | -0.75 | 44.6% |
| 10-20s | 56 | -0.59 | 44.6% |
| 20-30s | 49 | -1.05 | 34.7% |
| 30-60s | 48 | -0.89 | 50.0% |
| **60-120s** | **44** | **+0.84** | **61.4%** |
| 120s+ | 36 | -0.13 | 52.8% |

**唯一の正 PnL セグメント。しかし queue_wait は注文後にしか判明せず、事前フィルタに使用不可。**

### §6.3 ハイブリッド戦略テスト

| Strategy | n_keep | Skip率 | kept_PnL(bps) | 改善(bps) | 実現可能性 |
|----------|--------|--------|--------------|----------|-----------|
| **queue_wait_60s+** | 80 | 78.6% | **+0.30** | **+0.92** | ❌ 事後情報 |
| edge_filter≥0 | 165 | 55.8% | -0.16 | +0.46 | ❌ 事後情報 |
| buy_only | 192 | 48.5% | -0.30 | +0.32 | ✅ 実装容易 |
| skip_sell_bad_hours | 180 | 3.7% | -0.86 | +0.01 | ✅ 微小効果 |

### §6.4 ラウンドトリップ再分析

```
Total: 180 trips
Mean PnL: -1.10 bps
Win rate: 44.4%
Total:    -198.3 bps
p5/p50/p95: -23.2 / -1.1 / +23.9 bps
```

> ⚠️ 052# の +10.3 bps 報告と矛盾。ペアリング手法の差異による可能性 (052# は position tracking、本分析は consecutive opposite-side pairing)。

### §6.5 edge_at_fill の逆説

| edge閾値 | n | PnL(bps) | 改善 |
|----------|---|----------|------|
| ≥0 bps | 165 | -0.16 | +0.46 |
| ≥1 bps | 59 | -0.43 | +0.19 |
| ≥2 bps | 28 | -0.72 | -0.10 |
| ≥3 bps | 18 | **-1.54** | **-0.92** |

**「良い価格で約定」= 市場が大幅に逆行 = 逆選択。edge が大きいほど悪い。**

---

## §7 保存モデル候補

3 モデルを再学習・保存。いずれも ph2 投入時のフォールバックとして使用可能。

| # | ファイル | 種別 | サンプル数 | 主要特徴量 |
|---|---------|------|----------|-----------|
| C1 | `models/v460/skip_gate_as_070_candidate1.pkl` | AS分類器 (enriched LR C=0.1 k=5) | 166 | depth_imbalance_ob, tfi_300s, velocity_300s, return_60s/300s |
| C2 | `models/v460/skip_gate_as_070_candidate2.pkl` | AS分類器 (base LR C=0.01 all) | 284 | log_queue_wait, side_buy, hour_sin/cos, edge_bps, spread, offset, regime |
| C3 | `models/v460/pnl_regressor_070_candidate3.pkl` | PnL回帰器 (enriched GBR n=30 d=2 k=8) | 373 | velocity_30s(49%), return_30s(20%), trade_count_60s(15%), vpin_accel(10%) |

### C3 特徴量重要度

```
velocity_30s:       48.9%  ← 直近の取引速度が最重要
return_30s:         20.2%
trade_count_60s:    15.1%
vpin_acceleration:  10.0%
trade_count_30s:     5.0%
regime_trending:     0.7%
side_buy:            0.1%
regime_ranging:      0.0%
```

---

## §8 構造的診断

### なぜ ML が効かないのか

| 要因 | 値 | 必要水準 | 判定 |
|------|---|---------|------|
| サンプル数 (AS) | 284 | 800+ | ❌ |
| AS率 | 52.1% | 40% 以下 or 60% 以上 | ❌ 識別困難 |
| SNR (mean/std) | 0.11 | 0.5+ | ❌❌ |
| enriched サンプル | 166 | 500+ | ❌ |
| PnL 分布 | ほぼ正規、重尾 | 偏りがあれば ML 有利 | ❌ |

### 根本問題

1. **データ不足**: 284 サンプルでは、std=5.6 bps のノイズに対し mean=0.6 bps の信号は ML で検出不能
2. **AS 率 52%**: コインフリップ。ML に学習対象がない
3. **事後情報の優位性**: 最良フィルタ (queue_wait, edge_at_fill) は全て事前に不明
4. **特徴量の常数化**: WF の初期 fold で多くの OB/trade 特徴量が constant → SelectKBest が機能不全

---

## §9 Ph2 再投入に向けた提言

### §9.1 推奨構成 (Conservative)

```yaml
# fill_test.yaml 変更案
spread_offset_pct: 6.0      # 5.0 → 6.0 (per-fill edge 改善)
# SkipGate: 現行 Two-Tier 維持
# primary: candidate2 (base_LR_C0.01_kAll, 284 samples)
# fallback: candidate2 (= 同一。OB品質不良時も同モデル使用可)
as_threshold_buy: 0.65       # 据え置き
as_threshold_sell: 0.60      # 069# の側別閾値継続
```

**理由**: candidate2 は全 284 AS サンプルを使用し、特徴量に欠損なし。ROC は低いが、現行モデル (enriched k=12, 166 samples) より安定。

### §9.2 代替案: PnL 回帰器統合

candidate3 (GBR PnL 回帰器) を SkipGate の代替として使用:

- `predicted_pnl < threshold → skip`
- IC = 0.06 (微弱だが唯一の非ゼロ信号)
- 但し後半 fold で崩壊 → 追加データなしでは推奨困難

### §9.3 データ蓄積戦略

```
月曜入金 → fill test 再開 (7日間+)
目標: filled 800+, AS-labeled 600+
期待: SNR 改善 + WF fold 数増加 → ML 信号検出の可能性
再学習: 蓄積後に 070# スクリプト群を再実行
```

### §9.4 収益性の現実

現在のシステムは **per-fill PnL が恒常的に負** (-0.62 bps)。ML フィルタで最善ケースでも -0.30 bps (buy_only)。

**真の収益は spread capture (buy-sell 差額)** であり、per-fill PnL は本質的に負になる傾向がある (逆選択コスト)。ラウンドトリップ収益性は pairing 手法次第で大きく変わる (052# vs 本分析の乖離)。

fill test のメトリクスと実際の P&L を分離して評価する枠組みが必要。

---

## §10 実行スクリプト一覧

| ファイル | 目的 | 出力 |
|---------|------|------|
| `scripts/v460/ml/run_070_model_search.py` | 72+ 分類器 + 20 回帰器 + 12 ルール網羅スイープ | `reports/v460/model_search_070/` |
| `scripts/v460/ml/run_070_deep_analysis.py` | 時間帯・queue wait・RT・regime・多 horizon 分析 | `deep_analysis_results.json` |
| `scripts/v460/ml/run_070_final_analysis.py` | 上位候補の詳細 WF + ハイブリッド + モデル保存 | `final_analysis_results.json` + `.pkl` ×3 |
| `temp/analyze_data.py` | 基礎統計 | stdout |

---

## §11 結語

**72+ モデルの徹底探索の結果、現在のデータ規模では「儲かるモデル」は存在しない。**

これは悲観的結論ではなく、データ駆動型開発の正直な現状把握である。信号が存在するとすれば、velocity_30s と return_30s にわずかな痕跡が見られる (C3 の特徴量重要度)。800+ サンプルでの再検証が、最短の前進経路となる。

短期的には spread_offset 拡大 (5%→6%) による per-fill edge 改善が、ML モデルに依存しない唯一の構造的改善策。
