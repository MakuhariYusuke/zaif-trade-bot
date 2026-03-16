# 440# Toxicity Veto 調査報告: AS Classifier 失敗と代替設計

| 項目 | 内容 |
|---|---|
| 番号 | 440# |
| 分類 | ph4_impl (Phase4 Implementation) |
| 対象 | 437# §7 Phase 1 Toxicity Veto |
| 前提 | 437# S3結果確定、全fill_records 31日分取得済 |
| 目的 | ML-based toxicity veto の検証 → 失敗 → 代替設計・実装 |

---

## §0 Executive Summary

437# §7 Phase 1 が規定した **ML-based Toxicity Veto** の受入基準検証を実施した。

**結論: AS 分類器は機能しない。代替としてデータ駆動の regime-side offset 非対称化を実装する。**

| # | 発見 | 種別 |
|---|---|---|
| F1 | AS 分類器 ROC-AUC ≈ 0.50（ランダム同等）、skip simulation 改善は ≈ 0 bps | **致命的** |
| F2 | 全16特徴量の AS ラベルとの相関 \|r\| < 0.05 — pre-order 特徴量に信号なし | 構造的 |
| F3 | buy+ranging が最悪バケット (PnL=-0.41, n=1319, 36%) であるにもかかわらず、`ranging_offset_discount: 0.90` が offset を 10% 縮小（逆効果） | **設定ミス** |
| F4 | sell+ranging は PnL=-0.13 で near breakeven — buy と sell で根本的に異なるリスクプロファイル | 新事実 |
| F5 | 既存の `unknown_buy_offset_boost: 2.0` は適切。sell+unknown も PnL=-0.39 で要対策 | gap |
| F6 | 既に 67% がスキップ済み。残り 33% の fills は更なる ML フィルターで改善困難 | 構造的制約 |

---

## §1 Skip Simulation 結果 — 受入基準 FAIL

### §1.1 Walk-Forward AS (061# pipeline)

```
Folds:              69 (expanding window, embargo=2)
ROC-AUC (mean):     0.507 ± 0.090   ← ランダム (0.50) と識別不能
PR-AUC (mean):      0.566            ← baseline 0.528 に対し僅差
Brier (mean):       0.251

Skip 20% → -0.090 bps   FAIL (threshold ≥ 1.0)
Skip 10% → -0.011 bps   FAIL (threshold ≥ 1.0)
Baseline PnL: -0.239 bps
Valid samples: 3416
Feature stability (Jaccard): 0.133 ← 極めて不安定
Always selected: [buy_ratio, trade_flow_imbalance_60s]
```

### §1.2 TSCV (057# pipeline)

| Model | ROC-AUC | PR-AUC | Skip 20% | Skip 10% | 判定 |
|---|---|---|---|---|---|
| GB (k=8) | 0.491 | 0.523 | -0.012 bps | +0.008 bps | FAIL |
| LR (C=0.01) | 0.498 | 0.538 | -0.082 bps | -0.033 bps | FAIL |

### §1.3 Threshold Sweep (GB, OOF)

| threshold | n_keep | n_skip | skip_rate | PnL改善 (bps) | AS削減 |
|---|---|---|---|---|---|
| 0.30 | 0 | 2930 | 100% | +0.266 | -0.526 |
| 0.50 | 358 | 2572 | 87.8% | -0.203 | +0.004 |
| 0.55 | 1751 | 1179 | 40.2% | -0.099 | -0.004 |
| 0.60 | 2748 | 182 | 6.2% | -0.042 | -0.003 |
| 0.70 | 2927 | 3 | 0.1% | -0.001 | -0.000 |

**解釈**: モデルの出力確率は 0.50-0.60 に集中。閾値をどこに設定しても PnL 改善は得られない。
skip_rate=6.2% (threshold=0.60) でも改善は -0.042 bps（悪化）。

### §1.4 受入基準

| 基準 | 値 | 判定 |
|---|---|---|
| Skip 20% PnL改善 ≥ 1.0 bps | -0.012 bps | **FAIL** |
| Skip 10% PnL改善 ≥ 1.0 bps | +0.008 bps | **FAIL** |
| 参加率低下 ≤ 10% | N/A (改善なし) | **N/A** |

---

## §2 失敗の根因分析

### §2.1 特徴量-ラベル相関がゼロ

```
side_buy:           r = -0.021
hour_sin:           r = +0.003
hour_cos:           r = -0.012
spread_jpy:         r = +0.013
offset_ratio:       r = +0.006
regime_trending:    r = -0.007
regime_ranging:     r = -0.003
trade_count_60s:    r = -0.041  ← 最大でも |r| = 0.041
buy_ratio:          r = +0.019
trade_flow_imb_60s: r = +0.019
avg_trade_size:     r = -0.011
price_velocity_bps: r = -0.025
vpin_60s:           r = +0.004
side_aligned_tfi:   r = -0.004
side_aligned_vel:   r = -0.004
```

**全特徴量の |r| < 0.05** — pre-order の特徴量は個別取引レベルの AS を予測する情報を含まない。

### §2.2 構造的理由

1. **AS ラベルは 30 秒後の価格変動に依存**: 短期価格変動は本質的に確率的であり、pre-order の microstructure 情報からは予測不能
2. **AS rate ≈ 52.8%**: コイントスに近い。ラベル自体が高ノイズ
3. **deadzone フィルタ**: raw AS=52.5% → filtered AS=27.5%。25% が deadzone (±2.5bps) 内で除外されるが、学習は raw ラベルで実施
4. **67% 既スキップ**: 残りの fills は既に skip_gate, dynamic_kill, regime gates を通過した「生存者」。追加 ML フィルターの余地が極めて小さい
5. **特徴量安定性 Jaccard=0.133**: 15特徴量中 2つしか全foldで選択されない。学習する「信号」がない

### §2.3 結論

> **個別取引レベルの AS 予測は、現行の特徴量空間では達成不可能。**
> toxicity veto の ML-based アプローチは棄却する。

---

## §3 データ駆動の代替分析

### §3.1 Regime × Side 分析

| Regime × Side | n | AS rate | PnL (bps) | 評価 |
|---|---|---|---|---|
| **buy+ranging** | **1319** | **0.520** | **-0.407** | ★ 最悪 (最大量 × 最大損失) |
| sell+ranging | 1333 | 0.533 | -0.134 | △ 微損 |
| buy+trending | 118 | 0.483 | +0.573 | ◎ |
| sell+trending | 118 | 0.551 | -0.660 | ★ |
| buy+trending_down | 114 | 0.447 | +0.851 | ◎ 最良 |
| sell+trending_down | 118 | 0.559 | -0.229 | △ |
| buy+trending_up | 135 | 0.541 | -0.343 | ★ |
| sell+trending_up | 110 | 0.536 | -0.849 | ★ 第2位損失 |
| **buy+unknown** | **47** | **0.681** | **-1.384** | ★ 最悪 AS率+PnL |
| sell+unknown | 46 | 0.522 | -0.388 | △ |

### §3.2 PnL 分布

```
AS=True:   n=1910, mean=-3.903 bps, std=4.600, median=-2.497
AS=False:  n=1726, mean=+3.797 bps, std=4.685, median=+2.332

PnL gap: 7.70 bps
```

AS 判別ができれば 7.70 bps/trade の改善余地がある。問題は個別判別が不可能なこと。

### §3.3 既存ガードの状況

```
skip_gate:              1377 (12.0%)
sell_dynamic_kill:      1113 (9.7%)
spread_too_narrow:       818 (7.1%)
buy_dynamic_kill:        648 (5.6%)
timeout:                 489 (4.2%)
trending_sell_skip:      399 (3.5%)
forced_buy_delay:        389 (3.4%)
balance_forced_skip:     377 (3.3%)
stale_adverse_drift:     349 (3.0%)
per_side_dd_halt:        344 (3.0%)
ranging_low_vol_skip:    205 (1.8%)
TOTAL skipped: 7764 (67.4%)
```

### §3.4 設定ミスの発見

**`ranging_offset_discount: 0.90` は buy 側で逆効果**:

- ranging で offset を 0.90x に縮小 = spread を 10% narrower に設定
- buy+ranging は PnL=-0.41 で最悪バケット
- 「ranging=安定市場→aggressive pricing」の前提が buy 側で成り立っていない
- sell+ranging は PnL=-0.13 なので discount の恩恵がある（fill_rate 向上で損益改善可能性）

**根本原因**: 156# §18 で ranging_offset_discount を設計した際、side 別の AS リスクを検討しなかった。432# のデータ (buy+ranging PF=0.766) がこの問題を定量化した。

---

## §4 実装設計

### §4.1 方針

ML-based per-trade veto ではなく、**regime-side offset 非対称化** で対応する。

理論的根拠:
- **個別取引 AS は予測不能** (§2 で実証)
- **regime×side レベルでは統計的に有意な PnL 差が存在** (§3.1)
- 既存インフラ (RegimeBoostMixin, FillTestConfig) にそのまま乗る
- 設定変更のみで A/B 検証可能

### §4.2 変更 1: Ranging offset の side 非対称化

`_regime_boost_ranging()` (maker_regime_boost.py) を修正:
- `ranging_offset_discount_buy: float | None` を追加
- `ranging_offset_discount_sell: float | None` を追加
- None の場合は既存の `ranging_offset_discount` にフォールバック（後方互換）

**キャリブレーション** (432# + 本分析に基づく):
- buy: `1.15` — 現行 0.90 から反転。offset を 15% 拡大 = wider spread で AS リスク低減
- sell: `0.85` — 現行 0.90 よりやや aggressive。sell+ranging は near-breakeven なので fill_rate 改善が優先

### §4.3 変更 2: Unknown sell の offset boost 追加

`_regime_boost_unknown_buy()` を `_regime_boost_unknown()` に一般化:
- 既存 `unknown_buy_offset_boost: 2.0` はそのまま
- `unknown_sell_offset_boost: float = 1.0` を追加 (1.0 = 無効, 後方互換)
- sell+unknown は PnL=-0.39 で buy+unknown (-1.38) ほど深刻ではないが、1.3x 程度のガードは妥当

### §4.4 変更しないもの

| 項目 | 理由 |
|---|---|
| AS 分類器 (as_classifier.py) | ROC-AUC=0.50 — 修正ではなく使用中止 |
| skip_gate_evaluator.py | ML ベースの skip は残存（既存挙動維持） |
| SidecarSignal に prob_toxic 追加 | ML 予測が無意味なため不要 |
| orchestrator_mid_cycle.py の veto path | 新規 cancel_reason 不要 |
| 新規 cancel_reason (TOXICITY_VETO) | Offset 調整で対応するため veto path 不要 |

---

## §5 期待効果

buy+ranging (n=1319, 36% of fills) の offset を 0.90x → 1.15x に変更:
- fill_rate は低下する（offset 拡大 → 約位が遠くなる）
- fill あたりの PnL は改善する（wider spread で informed flow を回避）
- **net 効果は不確実** — A/B 検証が必要

しかし、少なくとも **逆効果な設定 (ranging buy で discount)** を除去できる。

---

## §6 結論

1. **ML-based Toxicity Veto は棄却**: 437# §7 Phase 1 の受入基準を達成不可能
2. **代替: regime-side offset 非対称化**: データ駆動で最悪バケット (buy+ranging) を修正
3. **既存インフラ活用**: RegimeBoostMixin + FillTestConfig の最小差分で実装
4. **設定ミス修正**: ranging_offset_discount が buy 側で逆効果 — 最も即効性の高い修正
5. **次のステップ**: A/B 検証 (paper trading) で net 効果を確認

---

## Appendix A: Skip Simulation 結果 JSON

完全な結果は `results/v460/skip_simulation_440.json` に保存済み。
