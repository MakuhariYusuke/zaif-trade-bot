# 073# ph2 レポート: 戦略分析 & パラメータチューニング

| key | value |
|---|---|
| 番号 | 073 |
| フェーズ | ph2 |
| 種別 | rpt + impl |
| 参照 | `scripts/v460/ml/run_073_strategy_analysis.py`, `scripts/v460/ml/run_073_strategy_sweep.py` |
| 作成日 | 2026-02-16 |
| テスト | 662 passed |
| 目的 | fill test データのセグメント分析・WF 検証による戦略改善と防御的パラメータ調整 |

---

## §0 エグゼクティブサマリ

**fill test 2 日分 (373 filled / 491 total) のデータに対し、14 種類の戦略を Walk-Forward 4-fold で検証。**
**結果: robustly positive な戦略は発見できず (070# の結論と整合)。しかし防御的改善を 3 点実装。**

### 現状メトリクス (G1.1-exec FAIL)

| 指標 | 現在値 | G1.1 基準 | 判定 |
|---|---|---|---|
| fill_rate | 76.0% | ≥ 90% | **FAIL** |
| AS_ratio | 39.1% | ≤ 20% | **FAIL** |
| post_fill_30s_pnl | -0.620 bps | ≥ 0 | **FAIL** |
| データ期間 | 2 日 | ≥ 7 日 | **FAIL** |
| サンプル数 | 373 filled | ≥ 200 | PASS |

### 実装した改善 (3 点)

1. **side 別 time_filter** — buy/sell 独立の時間帯フィルタリング
2. **sell offset 0.10 → 0.12** — sell PnL -0.958 に対する追加保守化
3. **E3 sampling 0.33 → 0.50** — 120s horizon データ収集強化

---

## §1 データ概要

- **期間**: 2026-02-13 → 2026-02-15 (UTC)
- **総レコード**: 491 (filled: 373, cancelled: 118)
- **fill_rate**: 76.0%
- **AS_ratio**: 39.1%
- **mean PnL (30s)**: -0.620 bps
- **median PnL (30s)**: -0.500 bps

### Side 別統計

| side | count | mean PnL | median PnL | AS% | 備考 |
|---|---|---|---|---|---|
| buy | 186 | -0.301 | -0.300 | 37.6% | sell の約 1/3 の損失 |
| sell | 187 | -0.958 | -0.700 | 40.6% | **buy の 3.2 倍悪い** |

**→ sell 側の PnL 改善が最優先課題。**

---

## §2 セグメント分析

### §2.1 Side × Hour ヒートマップ (UTC / JST)

**Buy 側トップ/ワースト:**

| UTC | JST | buy PnL (bps) | n | 評価 |
|---|---|---|---|---|
| 04 | 13 | **+3.993** | 14 | ★ 最強 |
| 22 | 07 | +2.040 | 10 | ★ 良好 |
| 20 | 05 | +1.550 | 8 | ★ 良好 |
| 07 | 16 | +0.933 | 6 | ○ 普通 |
| 23 | 08 | **-3.237** | 6 | ✗ 最悪 |
| 15 | 00 | -1.600 | 5 | ✗ 悪い |
| 11 | 20 | -1.168 | 8 | ✗ 悪い |
| 10 | 19 | -0.946 | 12 | △ やや悪い |

**Sell 側トップ/ワースト:**

| UTC | JST | sell PnL (bps) | n | 評価 |
|---|---|---|---|---|
| 15 | 00 | **+2.460** | 5 | ★ 最強 (buy は -1.600) |
| 20 | 05 | +1.289 | 7 | ★ 良好 |
| 05 | 14 | +0.747 | 6 | ○ 普通 |
| 08 | 17 | **-6.725** | 4 | ✗ 壊滅 |
| 04 | 13 | **-5.558** | 12 | ✗ 壊滅 (buy は +3.993) |
| 14 | 23 | -4.615 | 3 | ✗ 壊滅 |
| 03 | 12 | -2.165 | 4 | ✗ 悪い |
| 23 | 08 | -1.667 | 3 | ✗ 悪い |

**→ UTC04 が最大の機会: buy +3.993 vs sell -5.558。side 別 time_filter で解決可能。**
**→ UTC15 も好機: sell +2.460 vs buy -1.600。従来はグローバルブロック対象外だが buy をブロックすべき。**

### §2.2 Queue Wait セグメント

| queue_wait | mean PnL | n | 備考 |
|---|---|---|---|
| 0-10s | -0.756 | 89 | fast fill — AS リスク高 |
| 10-30s | -0.625 | 112 | 標準 |
| 30-60s | -0.654 | 80 | 標準 |
| 60-120s | **+0.841** | 52 | **唯一の正** |
| 120s+ | -0.980 | 40 | timeout 近辺 |

**→ 60-120s の待ち時間が最適ゾーン。しかし n=52 と少数。**

### §2.3 Multi-Horizon PnL

| horizon | mean PnL | n | 備考 |
|---|---|---|---|
| 30s | -0.620 | 373 | 全件、E1 |
| 60s | -0.350 | 124 | E3 サンプル |
| 120s | **+0.101** | 26 | **唯一の正** |

**→ 120s でのみ正転。中期的な mean reversion 効果の示唆。ただし n=26 は検証不十分。**
**→ E3 sampling ratio を 0.33 → 0.50 に引き上げてデータ収集を加速。**

---

## §3 Walk-Forward 4-Fold 戦略検証

### §3.1 方法

- 4-fold 時系列分割 (各 fold ≈ 93 records)
- train: fold 0-2 → test: fold 3、train: fold 0-1 → test: fold 2、等
- 各 fold の test PnL を計算、4/4 正が「robust」の基準

### §3.2 Strategy S0–S8 結果

| # | 戦略 | 条件 | WF mean PnL | 正の fold 数 | pass rate | 評価 |
|---|---|---|---|---|---|---|
| S0 | baseline | フィルタなし | -0.465 | 2/4 | 100% | ✗ |
| S1 | side×time | buy: UTC≠08,04; sell: UTC≠08,14,04 | -0.326 | 2/4 | - | ✗ |
| S2 | queue_wait 60-120s | 60s ≤ qw ≤ 120s のみ | +0.248 | 2/4 | 22% | ✗ 過少 |
| S3 | spread narrow 排除 | spread ≥ 2000 JPY のみ | -0.457 | 2/4 | - | ✗ |
| S4 | regime=range_bound | レジーム限定 | -0.479 | 2/4 | - | ✗ |
| S5 | S1+S2 結合 | side×time + qw | +0.432 | 2/4 | 8% | ✗ 超過少 |
| S6 | offset 10% sim | qw ≤ 5s 除外 | -0.465 | 2/4 | 100% | ✗ |
| S7 | AS side rotation | buy→sell で AS 降順 | -0.503 | 2/4 | - | ✗ |
| S8 | aggressive composite | S1+offset+regime | -0.326 | 2/4 | - | ✗ |

### §3.3 Strategy S9–S14 結果 (Sweep)

| # | 戦略 | 条件 | WF mean PnL | 正の fold | pass rate | 評価 |
|---|---|---|---|---|---|---|
| S9 | conservative side-time | buy: UTC≠{23,11,10,15}; sell: UTC≠{08,04,14,03,23} | -0.449 | 2/4 | 92.6% | ✗ |
| S10 | asymmetric side-time | buy: UTC≠{23,11,10}; sell: UTC≠{08,04,14,03,23,22} | -0.496 | 2/4 | 94.6% | ✗ |
| S11 | best hours only | UTC∈{04,22,20,07,05,15,06} のみ | -0.213 | 1/4 | 25.0% | ✗ |
| S12 | offset 7% sim | qw ≤ 7s の fast fill 除外 | -0.465 | 2/4 | 100% | ✗ |
| S13 | **sell offset boost** | sell qw ≤ 10s 除外 | **-0.330** | 2/4 | 85.2% | △ 最良 |
| S14 | combined best | S9 + S13 | -0.496 | 2/4 | 94.6% | ✗ |

**→ 全 14 戦略で 4/4 fold 正達成なし。**
**→ S13 (sell fast fill 排除) が最良: -0.330 bps、pass rate 85%。**

---

## §4 根本原因分析

### §4.1 データ不足

| 要因 | 現状 | 必要水準 | ギャップ |
|---|---|---|---|
| 期間 | 2 日 | 7 日 (G1.1) | **5 日不足** |
| filled 件数 | 373 | 800+ (070#) | **427 件不足** |
| 時間帯フィルタ影響 | 12h/24h OFF | - | 有効サンプル半減 |

**2 日間のデータでは、どの戦略も統計的に信頼できる正のエッジを検証不可能。**
070# で ROC-AUC ≤ 0.54 (284 samples) と結論した通り、データ量が根本ボトルネック。

### §4.2 構造的問題

1. **sell 側の系統的損失**: buy -0.301 vs sell -0.958。offset 差 (buy=0.05, sell=0.10→0.12) では吸収しきれない
2. **時間帯×side の強い交互パターン**: UTC04 は buy 最強 / sell 最悪。グローバル time_filter では最適化不可能だった
3. **短い PnL horizon**: 30s では噪音が支配的。120s で初めて正転するが n=26

### §4.3 070# との整合

| 070# 結論 | 073# 検証 | 整合性 |
|---|---|---|
| 全 ML モデル ROC-AUC ≤ 0.54 | WF 4/4 正の戦略なし | **完全整合** |
| 284 samples では予測力ゼロ | 373 fills でも改善なし | **整合** |
| 800+ samples 必要 | データ蓄積が critical path | **整合** |

---

## §5 実装した改善

### §5.1 side 別 time_filter (最重要)

**課題**: UTC04 は buy +3.993 / sell -5.558。グローバル time_filter ではどちらかを犠牲にする。

**解決**: `skip_utc_hours_buy` と `skip_utc_hours_sell` を追加。side 毎に独立の時間帯制御。

```yaml
time_filter:
  enabled: true
  skip_utc_hours: [1, 2, 8, 9, 12, 13, 14, 16, 17, 18, 19, 21]  # グローバルフォールバック
  skip_utc_hours_buy: [1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23]
  skip_utc_hours_sell: [1, 2, 3, 4, 8, 9, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23]
```

**変更点 (グローバル比):**
- buy: UTC10, 11, 15, 23 を追加ブロック (全て buy PnL < -0.9)
- sell: UTC3, 4, 22, 23 を追加ブロック (全て sell PnL < -0.8)
- sell: **UTC15 をアンブロック** (sell PnL +2.460、buy は -1.600 なのでブロック)

**実装ファイル:**
- `scripts/v460/run_fill_test.py`: `FillTestConfig` に `skip_utc_hours_buy/sell` 追加、`_is_time_filtered(side)` 拡張
- `configs/v460/fill_test.yaml`: side 別リスト設定
- `tests/unit/v460/test_regime_detector.py`: 5 テスト追加

**影響**: メインループで `_next_side()` 後に side 別フィルタ判定。フィルタされた side の場合は反対 side を試み、両方フィルタされた場合のみスリープ。

### §5.2 sell offset 引き上げ

```yaml
side_offset:
  sell: 0.12   # 073#: 0.10 → 0.12
```

**根拠**: sell PnL -0.958 bps は buy -0.301 の 3.2 倍。追加 0.02 の保守化で sell 側 AS を低減。

### §5.3 E3 sampling 引き上げ

```yaml
e3:
  sampling_ratio: 0.50  # 073#: 0.33 → 0.50
```

**根拠**: 120s horizon が唯一正の PnL (+0.101 bps) を示した。しかし n=26 では検証不十分。
サンプリング比率を 50% に引き上げ、60s/120s データの蓄積を加速。

---

## §6 次ステップ

### §6.1 Critical Path: データ蓄積

1. **fill test 再開** — 上記パラメータで 5 日間追加運用 (目標: 800+ filled)
2. **中間判定** (3 日後) — interim judgment で PnL 傾向を確認
3. **800 samples 到達後** — 070# の ML モデル再訓練 (ROC-AUC > 0.60 を目標)

### §6.2 800 samples 到達後の戦略

| 優先度 | アクション | 期待効果 |
|---|---|---|
| P0 | SkipGate 再訓練 (800+ samples) | ROC-AUC 0.54 → 0.60+ |
| P1 | 120s horizon 正式採用判断 | PnL 計測精度向上 |
| P2 | OB 特徴量復元 (072# toggle) | SkipGate 精度さらに向上 |
| P3 | 動的ロットサイジング有効化 | PnL 正転後のリターン拡大 |

### §6.3 ゲート判定見通し

```
現在: fill_rate=76%, AS=39%, PnL=-0.620 bps (2 日, 373 fills)
目標: fill_rate≥90%, AS≤20%, PnL≥0 bps (7 日, 200+ fills)

ギャップ:
  fill_rate: +14pp → time_filter 最適化で改善 (レコード品質向上)
  AS_ratio: -19pp → offset 保守化 + SkipGate 再訓練
  PnL: +0.62 bps → データ蓄積 + 120s horizon + side 別最適化
```

---

## §7 ファイル変更一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/run_fill_test.py` | `FillTestConfig`: `skip_utc_hours_buy/sell` 追加、`_is_time_filtered(side)` 拡張、メインループ side 別判定 |
| `configs/v460/fill_test.yaml` | side 別 time_filter、sell offset 0.12、E3 sampling 0.50 |
| `tests/unit/v460/test_regime_detector.py` | side 別 time_filter テスト 5 件追加 |
| `tests/unit/v460/test_fill_quality.py` | sell offset assertion 0.10 → 0.12 更新 |
| `scripts/v460/ml/run_073_strategy_analysis.py` | 分析スクリプト (S0-S8, WF-4fold) |
| `scripts/v460/ml/run_073_strategy_sweep.py` | Sweep スクリプト (S9-S14) |
| `docs/v460/073_ph2_rpt_strategy_analysis.md` | 本ドキュメント |
