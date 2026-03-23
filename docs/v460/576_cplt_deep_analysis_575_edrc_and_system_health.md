# 576# [cplt] 深堀り分析: 575# eDRC 有効化後の稼働状況 + システム全体改善点洗出し

> **ステータス**: AI分析待ち  
> **作成日**: 2026-03-23  
> **目的**: 575# eDRC 有効化後の初期観測データ + 直近4日間 (3/20–3/23) の定量データを基にした改善余地の洗い出し  
> **参照**: 572# (eDRC A/B toggle), 573# (RobustStats + テレメトリ), 574# (パラメータ推定), 575# (eDRC有効化)

---

## §1. コミット履歴と変更チェーン

```
c164d21d3  fix: _build_fill_record に execution_sigma/adverse_ofi/additive_enabled 引数追加 (573# バグ修正)
cf9ba5ca7  test: simple reward regression coverage
cb1cb8551  575# eDRC有効化: enabled=true, α=0.020, β=0.40, hard_cap=1.0
5b7e8d619  fix: simple reward transaction cost contract
4c2c0d54c  575# eDRC 574#統合: σ→bps変換, hard_cap=1.0, get_robust_inputs()
bba7fb7e7  573# telemetry: execution_sigma/adverse_ofi/additive_enabled to FillRecord
947010441  573# impl: robust_stats.py + section_execution_quality_comparison
d1afbdef6  feat(572): eDRC A/B toggle infrastructure
fa7f793ea  feat: 569# immediate measurement and parameter fixes
```

---

## §2. 573# テレメトリバグの発見と修正

### 2.1 症状

573# コミット (`bba7fb7e7`) で `execution_sigma`, `execution_adverse_ofi`, `execution_additive_enabled` を
`FillRecord` フィールドおよび `fill_cycle_executor.py` の呼び出し側に追加したが、
**受け側の `_build_fill_record()` メソッド引数に追加が漏れていた**。

```
TypeError: FillRecordBuilderMixin._build_fill_record() got an unexpected keyword argument 'execution_sigma'
```

### 2.2 影響

- 575# SHA (`cb1cb8551`) の稼働期間 (UTC 02:38〜08:25, 約5.7h) 中、**1件以上の実約定が record 構築エラーで消失**
  - ログに `Placed order → Order filled → _build_fill_record() TypeError` のシーケンス確認
  - JSONL には fill_price=None のまま記録される（fill 後のフィールドが未設定）
- preflight_insufficient / skip_gate で注文に到達しなかったサイクルは影響なし

### 2.3 修正

`_build_fill_record()` に 3 引数を追加 (コミット `c164d21d3`)。hot-swap restart 完了 (17:23 JST)。

### 2.4 教訓

- テレメトリフィールド追加時のチェーンは 4 箇所の同期が必要:
  1. `FillRecord` dataclass (ztb/metrics/fill_quality.py)
  2. `OffsetPipelineResult` dataclass (offset_pipeline.py)  
  3. `fill_cycle_executor.py` の呼び出し側
  4. **`fill_record_builder.py` の `_build_fill_record()` 引数定義** ← 今回漏れた箇所
- 統合テストでこのチェーンの一貫性を検証する仕組みが必要

---

## §3. 直近4日間パフォーマンス (3/20〜3/23 UTC)

### 3.1 基本統計

| 指標 | 値 |
|------|----|
| Total records | 1,941 |
| Filled | 591 (30.4%) |
| Buy filled | 338/995 (34.0%) |
| Sell filled | 253/933 (27.1%) |
| Avg PnL30 (全体) | -0.07 bps |
| Avg PnL30 (buy) | -0.28 bps |
| Avg PnL30 (sell) | +0.21 bps |
| Sum PnL30 | **-42.41 bps** |
| Total Loss | -1,159.96 bps |
| AS rate | 27.1% (160/591) |
| AS avg PnL | -6.29 bps |
| Non-AS avg PnL | +2.23 bps |

### 3.2 日別パフォーマンス

| 日付 | Orders | Filled | Rate | Avg PnL30 | Sum PnL30 |
|------|--------|--------|------|-----------|-----------|
| 3/20 (木) | 546 | 287 | 53% | +0.03 | +7.9 |
| 3/21 (金) | 697 | 130 | 19% | -0.15 | -19.1 |
| 3/22 (土) | 608 | 152 | 25% | -0.54 | **-81.7** |
| 3/23 (日) | 90 | 22 | 24% | +2.30 | +50.5 |

**3/22 (土) の -81.7 bps が支配的損失源。** 3/23 は旧SHA分のみ約定。

### 3.3 Side 別詳細

#### Buy 側
- Fill rate: 34.0%
- AS率: 20.4% (69/338)
- AS loss avg: -5.28 bps
- P10: -4.37 bps, P05: -5.60 bps
- Profitable: 49.1%

#### Sell 側
- Fill rate: 27.1%
- **AS率: 36.0% (91/253)** ← buy の 1.76 倍
- **AS loss avg: -7.05 bps** ← buy より 1.34 倍深い
- **P10: -8.26 bps, P05: -11.67 bps** ← buy (P10=-4.37) の約2倍
- Profitable: 50.6%
- Worst fills: -22.20, -20.79, -19.66, -16.38, -15.82 bps ← **全て sell**

### 3.4 Cancel Reason 分布

| 理由 | 件数 | 比率 |
|------|------|------|
| preflight_insufficient | 468 | 34.7% |
| skip_gate | 234 | 17.3% |
| no_feasible_quote | 146 | 10.8% |
| timeout | 145 | 10.7% |
| sell_dynamic_kill | 98 | 7.3% |
| spread_too_narrow | 90 | 6.7% |
| final_clamp_hard_skip | 67 | 5.0% |
| buy_dynamic_kill | 31 | 2.3% |
| postonly_crossing_skip | 30 | 2.2% |
| cross_venue_lead_lag_veto | 24 | 1.8% |

### 3.5 時間帯別パフォーマンス (不採算帯の特定)

**不採算帯（Avg PnL30 < -1.0 bps）:**

| UTC | Fills | Avg PnL30 | Profitable% |
|-----|-------|-----------|-------------|
| 03h | 29 | -1.34 | 38% |
| 12h | 27 | -1.63 | 30% |
| 14h | 21 | -2.00 | 38% |
| 17h | 28 | -1.21 | 36% |
| 19h | 22 | -1.05 | 41% |

**採算帯 (Avg PnL30 > +1.0 bps):**

| UTC | Fills | Avg PnL30 | Profitable% |
|-----|-------|-----------|-------------|
| 09h | 32 | +1.16 | 59% |
| 11h | 28 | +1.20 | 61% |
| 20h | 14 | +2.07 | 64% |

### 3.6 Sell 側 時間帯詳細

| UTC | Fills | Avg PnL30 | 注目 |
|-----|-------|-----------|------|
| 02h | 13 | -2.40 | 不採算 |
| 03h | 13 | -1.84 | 不採算 |
| 04h | 12 | -1.13 | 不採算 |
| 12h | 13 | -2.06 | 不採算 |
| 17h | 12 | -1.48 | 不採算 |
| 22h | 4 | -2.05 | 不採算 (少数) |

---

## §4. Clamp Saturation 分析

### 4.1 現行数値

| Side | Clamped | Rate | Pre-clamp avg | Effective avg | Pre-clamp P50 | P90 | P99 |
|------|---------|------|---------------|---------------|---------------|-----|-----|
| Buy | 248/253 | 98% | 0.3431 | 0.2431 | 0.3157 | 0.4856 | 0.6185 |
| Sell | 178/178 | **100%** | 0.3275 | 0.2399 | 0.2910 | 0.5219 | 0.6515 |

### 4.2 PnL分岐 (clamped vs unclamped)

- Buy clamped: +0.16 bps / Buy unclamped: +1.35 bps
- Sell clamped: -0.61 bps (unclamped データなし — 全て clamped)

### 4.3 現行 YAML ceiling 設定

```yaml
offset_ceiling_ratio: 0.15           # 共通デフォルト
offset_ceiling_ratio_buy: 0.35       # buy ceiling (565# P1)
offset_ceiling_ratio_sell: 0.40      # sell ceiling (565# P1)
```

### 4.4 検討課題

- **Sell 側 100% clamp saturation**: sell の clamp 率が 100% ということは、ceiling が低すぎる可能性
  - ただし pre-clamp P50=0.2910 vs ceiling=0.40 なので、P50 は ceiling 内
  - P90=0.5219 が ceiling (0.40) を超えている → 上位 10% が常に clamp
  - eDRC 有効時 (enabled=true) は ceiling=0.40 が C_base になり、σ/OFI に応じて拡大されるため自動的に改善する方向
- **Buy 側 unclamped PnL +1.35 vs clamped +0.16**: ceiling を上げれば PnL 改善の可能性
  - ただし unclamped のサンプル n=5 と少ないため統計的信頼度は低い

---

## §5. eDRC 実装の現在状態

### 5.1 eDRC 数式

$$C_{dynamic} = \min\left(C_{base} \times e^{\alpha \cdot \sigma_{bps} + \beta \cdot OFI_{adverse}},\ \text{hard\_cap}\right)$$

現行パラメータ: α=0.020, β=0.40, C_base=0.40, hard_cap=1.0

### 5.2 入力値パイプライン

```
last_sigma (Parkinson σ, ratio ~0.0001)
  → ×10,000 → σ_bps
  → get_robust_inputs(side) で asymmetric EMA を適用 (additive_enabled時)
  → eDRC α 項として使用

get_adverse_ofi(side) (50 サイクル rolling mean, clamped ≥ 0)
  → get_robust_inputs(side) で median filter を適用 (additive_enabled時)
  → eDRC β 項として使用
```

### 5.3 シミュレーション表

| σ_bps | OFI | C_dynamic | C_dynamic / C_base |
|-------|-----|-----------|-------------------|
| 0 | 0.0 | 0.400 | 1.00× |
| 10 | 0.0 | 0.489 | 1.22× |
| 10 | 0.3 | 0.550 | 1.37× |
| 10 | 0.8 | 0.672 | 1.68× |
| 30 | 0.0 | 0.729 | 1.82× |
| 30 | 0.3 | 0.821 | 2.05× |
| 30 | 0.8 | **1.000** | 2.50× (hard_cap) |
| 50 | 0.0 | **1.000** | 2.72× (hard_cap) |

### 5.4 eDRC vs 既存: side 別 ceiling の非対称性問題

- **既存ロジック**: buy ceiling = 0.35, sell ceiling = 0.40 (side 非対称)
- **eDRC 有効時**: C_base=0.40 が buy/sell 共通 (side 非対称が消失)
  - eDRC resolve_offset_ceiling のコードを見ると、`side` パラメータは受け取るが **`offset_ceiling_ratio_buy/sell` を参照していない**
  - → **buy 側の ceiling が 0.35→0.40 に上がる (意図的かどうか不明)**
  - → **sell 側は 0.40→0.40 で変化なし (σ=0,OFI=0 時)**

---

## §6. 未実装・観測不能項目

### 6.1 spread_capture_bps / adverse_selection_cost_bps

> `spread_capture_bps 未記録 (0/591 fills)`

571# で `section_execution_quality_comparison()` を分析スクリプトに追加済みだが、
**fill_recorder 側で spread_capture_bps を計算・記録する実装がない**。

Kissell & Glantz 指標の定義:
- `spread_capture_bps = (fill_price - mid_at_fill) / mid_at_fill × 10000 × side_sign`
- `adverse_selection_cost_bps = (mid_30s_after - mid_at_fill) / mid_at_fill × 10000 × side_sign`

### 6.2 OB Age (ms)

> `(no data)`

板年齢の記録がない。レイテンシ劣化検知に有用。

### 6.3 Confidence Lot

> `None: 591`

全 fill で confidence lot が None → lot sizing が静的。regime-aware な lot adjustment が機能していない可能性。

### 6.4 Reprice

> `Repriced: 0/591 (0.0%)`

reprice が 0% — 452# micro_timeout の re-quote とは別の機能か？ログには re-quote が出ているが reprice_count=0。

---

## §7. Adverse Selection 構造分析

### 7.1 全体 AS 構造

| 指標 | Buy | Sell |
|------|-----|------|
| AS Rate | 20.4% | **36.0%** |
| AS Avg Loss | -5.28 bps | **-7.05 bps** |
| Non-AS Avg Gain | +2.23 bps | +4.29 bps |
| Edge Ratio (non-AS / AS) | 0.42 | 0.61 |

### 7.2 Sell AS Microstructure Correlation

| 分類 | AS Rate | n |
|------|---------|---|
| Low spread (<1.97 bps) | 37.0% | 135 |
| High spread | 34.7% | 118 |
| Toxic OB imbalance (>0.3) | 34.2% | 76 |
| Normal imbalance | 36.7% | 177 |
| High VPIN (>0.70) | **38.7%** | 142 |
| Low VPIN | 32.7% | 110 |

**VPIN が最も AS 予測力が高い** (高VPIN時 38.7% vs 低VPIN時 32.7%, 差 6.0pp)。
ただし差は小さく、**microstructure 単独では AS を十分に予測できない**。

### 7.3 Cross-Venue 効果

- CV applied: 181/591 (30.6%)
- Buy widen PnL: +0.41 bps (好影響)
- **Sell widen PnL: -1.65 bps** (悪影響)
  - sell 側の cross-venue widen が逆効果の可能性

---

## §8. eDRC YAML 設定異常: enabled=true だが YAML は enabled=false

### 8.1 YAMLファイルの現状確認

575# コミット (`cb1cb8551`) で `enabled: true` に変更されたが、
それ以降のコミット (c164d21d3 = テレメトリバグ修正) では YAML を変更していない。

**確認事項**: 現在の YAML が `enabled: true` のままか `enabled: false` に戻っていないか。

`configs/v460/fill_test.yaml` の experimental_additive_pipeline セクション:
```yaml
experimental_additive_pipeline:
  enabled: true          # ← 575# で有効化
  edrc_alpha: 0.020
  edrc_beta: 0.40
  edrc_c_base: 0.40
  additive_base_bps: 0.0
```

→ enabled=true のまま。eDRC は本番稼働中。

---

## §9. 改善候補リスト (AI 分析依頼)

以下の項目について、データと実装コードを基に多角的に分析・提案を求める。

### P1 (Critical)

1. **Sell AS率 36% の構造的原因**: なぜ sell の AS 率が buy の 1.76 倍なのか？
   - balance_forced は 0 なので inventory skew は直接原因ではない
   - sell_dynamic_kill=98 が "必要な保護" か "過剰フィルタ" か

2. **3/22 (土) の -81.7 bps 集中損失**: 何が起きたのか？
   - 日別で唯一の大幅マイナス。regime / SHA / 時間帯の交絡要因分析

3. **Sell Tail: Worst 5 が ALL sell (-22〜-16 bps)**: テール分布の非対称性
   - buy worst = -7.10 bps vs sell worst = -22.20 bps → 3.1 倍の差

### P2 (Important)

4. **Clamp Saturation 98-100%**: ceiling が機能しているが、eDRC 有効時の効果予測
   - eDRC で ceiling が動的に拡大すると、これまで clamped だった fill が unclamped になる
   - それは期待利得の向上なのか、テールリスクの増大なのか？

5. **Skip Gate 17.3%**: 適切な水準か？
   - sell 側 skip_gate=139/933 (14.9%) → skip されなかったものの AS率=36%
   - skip_gate の precision/recall トレードオフ分析

6. **Cross-Venue Sell Widen が逆効果 (-1.65 bps)**: disabled にすべきか？

7. **final_clamp_hard_skip=67 (5.0%)**: hard_skip_mult=2.5 は適切か？
   - skip された 67 件の"もし実行されていたら"の仮想 PnL 推定

### P3 (Nice to Have)

8. **spread_capture_bps 未記録**: fill_recorder への計算ロジック追加の設計
   - 既に mid_at_fill, mid_30s_after, fill_price が FillRecord に存在 → 算出可能

9. **eDRC side 非対称性消失** (§5.4): C_base を side 別にすべきか？
   - `edrc_c_base_buy`, `edrc_c_base_sell` の分離を検討

10. **σ_bps の実測レンジ**: execution_sigma テレメトリ開始で初めて観測可能に
    - テレメトリバグ修正後のデータでα,βの妥当性を再検証

11. **RobustStats 未統合** (§5.2): get_robust_inputs() は実装済みだが、
    eDRC 無効時 (既存ロジック) では使われない。eDRC 有効時のみ使われる設計で正しいか？

12. **VG triggered 37.7%**: Volatility Guard がほぼ4割のサイクルで発火。
    過剰抑制ではないか？

---

## §10. 分析再現コマンド

```bash
# 全体分析 (3/20-3/23)
python -m scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-20 --date-to 2026-03-23 -o temp/analysis_576_full.txt

# Sell 側のみ
python -m scripts.v460.analysis.analyze_fill_logs --date-from 2026-03-20 --date-to 2026-03-23 --side sell -o temp/analysis_576_sell.txt

# SHA 別 (575# eDRC 有効化後のみ)
python -m scripts.v460.analysis.analyze_fill_logs --git-sha cb1cb8551 -o temp/analysis_576_edrc.txt

# テレメトリバグ修正後 (c164d21d3 以降)
python -m scripts.v460.analysis.analyze_fill_logs --git-sha c164d21d3 -o temp/analysis_576_fixed.txt
```

---

## §11. 現行 eDRC 実装コード (参照用)

### resolve_offset_ceiling (fill_config.py L364-403)

```python
def resolve_offset_ceiling(
    self, side: str, *, utc_hour: int | None = None,
    sigma: float = 0.0, adverse_ofi: float = 0.0,
) -> float:
    if self.experimental_additive_pipeline:
        from math import exp
        ceiling_dynamic = self.edrc_c_base * exp(
            self.edrc_alpha * sigma + self.edrc_beta * adverse_ofi
        )
        if utc_hour is not None and self.hour_ceiling_mult:
            mult = self.hour_ceiling_mult.get(utc_hour)
            if mult is not None:
                ceiling_dynamic *= mult
        return ceiling_dynamic
    # 既存ロジック (side 別 ceiling)
    ceil = self.offset_ceiling_ratio
    if side == "buy" and self.offset_ceiling_ratio_buy is not None:
        ceil = self.offset_ceiling_ratio_buy
    elif side == "sell" and self.offset_ceiling_ratio_sell is not None:
        ceil = self.offset_ceiling_ratio_sell
    if ceil > 0 and utc_hour is not None and self.hour_ceiling_mult:
        mult = self.hour_ceiling_mult.get(utc_hour)
        if mult is not None:
            ceil *= mult
    return ceil
```

**注意**: eDRC 分岐では `offset_ceiling_ratio_buy/sell` が参照されない (§5.4)。

### get_adverse_ofi (maker_price.py L420-429)

```python
def get_adverse_ofi(self, side: str) -> float:
    if not self._ofi_history:
        return 0.0
    ofi_mean = sum(self._ofi_history) / len(self._ofi_history)
    adverse = -ofi_mean if side == "buy" else ofi_mean
    return max(0.0, adverse)
```

### RobustStats (ztb/utils/robust_stats.py)

```python
class RobustStats:
    @staticmethod
    def clip_outliers_mad(data, threshold=3.0) -> np.ndarray: ...
    @staticmethod
    def robust_ema(current_val, prev_ema, alpha, sigma_limit=None) -> float: ...
    @staticmethod
    def asymmetric_ema(current_val, prev_ema, alpha_up, alpha_down) -> float: ...
    @staticmethod
    def median_filter_fast(buffer) -> float: ...
```

### Offset Pipeline 9 段チェーン (offset_pipeline.py)

```
Stage 1: 193# EV → offset_mult (ev_score → offset sensitivity)
Stage 2: 195# Velocity → offset_mult (price velocity → offset boost)
Stage 3: 196# Trending → offset_mult (sell only when trending regime)
Stage 4: 240# Toxicity → offset_mult (Glosten-Milgrom AS budget)
Stage 5: 202# VG supplement → offset_mult (volatility guard sell-side complement)
Stage 6: 458# Macro → offset_mult (macro trend → premium boost)
Stage 7: 215# Alert → offset_mult (alert mode offset)
Stage 8: 372# Sidecar → delta_bps (SAC sidecar bps adjustment)
Stage 9: 421# Final Clamp → ceiling enforcement (eDRC or static)
```

全段が**乗法チェーン** (∏ offset_mult)。将来の加法パイプライン ($\sum$ offset) への移行が 568# M2 で設計済み。
