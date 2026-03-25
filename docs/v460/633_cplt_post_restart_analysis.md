# 633# 根本原因深掘り分析: Death Spiral 構造と資金投入なし改善戦略

## 概要
632# (ATR floor mult=2.0→1.2 + cap_bps=3.0) 適用の検証に加え、入金不可制約下での根本原因解明を実施。7日間(3/19-3/25)のfill records全量を解析し、balance depletion→機会損失→さらなるdepletion という Death Spiral の構造を定量的に特定した。

## 1. 632# ATR Floor 検証結果 (成功)

| 指標 | 旧(mult=2.0) | 新(mult=1.2, cap=3.0) |
|------|-------------|---------------------|
| Floor値 (σ=0.000244) | 5,534 JPY (4.88 bps) | 3,320 JPY (2.93 bps) |
| spread_too_narrow/h | ~38回/h | ~4回/h |
| 削減率 | — | **89%** |

## 2. 7日間全量分析 — 包括サマリー (3/19-3/25)

| 指標 | 値 |
|------|-----|
| 総サイクル | 3,511 |
| 約定 | 1,011 (28.8%) |
| preflight_insufficient | 966 (27.5%) |
| 7日間 Net PnL | **-739 JPY** |
| 約定あたり PnL | **-0.73 JPY/fill** |
| Sell 累計 | -357 JPY (win 2/7日) |
| Buy 累計 | -382 JPY (win 3/7日) |

**致命的事実: maker手数料0%にもかかわらず、per-fill edgeが負。**

## 3. Death Spiral の定量構造

### 3.1 Balance / Notional 推移

| 日付 | First Notional | Last Notional | Fills/Total | Preflight% | Avg Qty (BTC) |
|------|---------------|---------------|-------------|------------|----------------|
| 3/16 | 11,613 | 11,904 | 83/607 | 0% | — |
| 3/17 | 11,866 | 11,865 | 109/635 | 9% | — |
| 3/18 | 11,770 | **26,555** | 108/453 | 19% | — (入金) |
| 3/19 | 26,024 | 25,516 | 115/338 | **0%** | 0.00224 |
| 3/20 | 25,491 | 25,546 | 287/546 | **0%** | 0.00227 |
| 3/21 | 25,540 | 24,754 | 130/697 | **30%** | 0.00223 |
| 3/22 | 24,720 | 24,617 | 152/608 | **36%** | 0.00222 |
| 3/23 | 24,614 | 23,813 | 128/371 | **34%** | 0.00216 |
| 3/24 | 23,817 | 23,336 | 97/411 | **51%** | 0.00210 |
| 3/25 | 23,340 | 23,265 | 100/534 | **37%** | 0.00143 |

**ターニングポイント: 3/21**
- 3/20: 0% block → 3/21: 30% block (一夜で急変)
- 残高が preflight 閾値を割り、以降回復せず
- 3/25 で avg qty が 0.00143 BTC に崩壊 (3/24比 -32%)

### 3.2 サイクル分類の完全分解 (3/25)

| カテゴリ | 件数 | 割合 | 意味 |
|---------|------|------|------|
| filled | 100 | 18.7% | 約定成功 |
| skip_gate | 52 | 9.7% | モデルが不利と判断 |
| spread_too_narrow | 152 | 28.5% | スプレッド不足 (大半が632#前) |
| **preflight_insufficient** | **196** | **36.7%** | 残高不足で発注不可 |
| mcb_halt | 15 | 2.8% | 回路遮断器 |
| timeout | 11 | 2.1% | 発注→未約定 |
| その他 | 8 | 1.5% | clamp_hard_skip等 |

### 3.3 Death Spiral の因果ループ

```
  ┌─ Adverse Selection ──→ Fill損失 ──→ Balance減少 ─┐
  │                                                    │
  │    ┌── fill rate低下 ←── lot縮小 ←── 37%block ←──┘
  │    │
  │    └── 回復機会の喪失 ──→ さらなるBalance減少 ──┘
  └────────────────────────────────────────────────────┘
```

## 4. 根本原因分析 (5因子)

### RC1: Balance Concentration Risk (Impact: 最大)
**27.5%の全サイクルが残高不足で無駄**

BTC/JPY のバランスが偏ると、一方のsideが取引不能になる:
- sell したい → BTC不足 (JPYに集中)
- buy したい → JPY不足 (BTCに集中)

preflight_insufficient は 3/19(0%) → 3/21(30%) → 3/24(51%) と急上昇。
3/18入金時の26K JPY相当が23K JPY未満に減少し、閾値を割った時点でシステム性能が半減。

### RC2: Sell-side Fast Fill Adverse Selection (Impact: 大)
**Sell fast fill (<15s) が構造的に損失**

| 日付 | Sell fast p30 | Sell fast p60 | Sell slow p30 | Buy fast p30 |
|------|-------------|-------------|-------------|-------------|
| 3/24 | -1.02 bps | **-2.15 bps** | -1.57 bps | -1.29 bps |
| 3/25 | -1.44 bps | **-3.69 bps** | -0.25 bps | +0.29 bps |

Sell fast fill は 3/25 で p60=-3.69bps — 即座に逆方向に価格が動き、informed trader に picking off されている。

**根拠**: Sell order が queue の先頭にあるとき、価格が上昇中の大口が market buy を実行 → sell 約定 → 価格さらに上昇 → maker 損失。

### RC3: VG Continuous Mode 恒常的発動 (Impact: 中)
**97%の約定でVGがトリガー — "ガード"ではなくtax**

| パラメータ | 現行値 | 問題 |
|-----------|--------|------|
| vpin_continuous_min | 0.40 | avg vpin=0.623 で常時上回る |
| トリガー率 | 96/100 (3/25) | 97% — ほぼ全件 |
| うちboost>1 | 33/96 | 34% のみ実効boost |
| avg boost | 1.173 | 実質11.7%のoffset上乗せ |

boost分布: median=1.000, p75=1.281 → 半数以上はboost=1.0 (ノーオペ)
66%が「VG triggered」だが実際のoffsetへの影響はゼロ。

boost formula: `1 + (2.0-1.0) × min((vpin-0.40)/0.40, 1.0)²`
- vpin=0.50: norm=0.25, boost=1.063 (微小)
- vpin=0.60: norm=0.50, boost=1.250 (中程度)
- vpin=0.80: norm=1.00, boost=2.000 (最大)

### RC4: Clamp Saturation (Impact: 中)
**モデルの危険信号をクランプが遮断**

| side | ceiling | 飽和率 | pre-clamp avg | pre-clamp max |
|------|---------|--------|---------------|---------------|
| sell | 0.40 | 51% | 0.5979 | 0.9660 |
| buy | 0.35 | 67% | 0.4911 | 0.5545 |

Sell: モデルが avg 0.598 の offset を要求 → 0.40 で切り詰め → 不十分な保護で約定 → AS損失。
Pre-clamp max=0.966 は「モデルが殆ど取引すべきでないと判断」している場面で 0.40 で発注を強制。

### RC5: Spread-too-narrow (632# で89%解決済み)
3/25の152件中、大半は07:00以前 (632#適用前)。
適用後は ~4件/h に減少（正当なガード動作）。

## 5. Side 別 PnL 日次推移

| 日付 | Sell fills | Sell PnL (JPY) | Buy fills | Buy PnL (JPY) |
|------|-----------|---------------|-----------|---------------|
| 3/19 | 57 | **-335** | 58 | -10 |
| 3/20 | 138 | +200 | 149 | -187 |
| 3/21 | 44 | -68 | 86 | +15 |
| 3/22 | 60 | -80 | 92 | -119 |
| 3/23 | 59 | +222 | 69 | +42 |
| 3/24 | 47 | **-230** | 50 | **-166** |
| 3/25 | 44 | -66 | 58 | +43 |

3/19の-335 JPYと3/24の-396 JPYが壊滅的。この2日だけで7日間損失の99%を占める。

## 6. 改善提案 (入金不可制約下)

### P0: VG vpin_continuous_min 引き上げ (信頼度: ★★★★☆)
```yaml
volatility_guard:
  vpin_continuous_min: 0.55  # 0.40→0.55
```
**根拠**:
- 現行0.40は avg vpin=0.623 に対し97%トリガー → 「常時ガード = ガードしていない」
- 0.55に引き上げ → トリガー率 ~55% に低下 (norm計算: (0.623-0.55)/0.40=0.183, boost=1.033)
- 低vpin時 (<0.55) は VG なしの素のoffsetで約定 → fill rate改善
- 高vpin時 (>0.55) は従来通り保護
- **リスク**: 低vpin時のAS増加 → ただし現在boost=1.0で保護効果ゼロのケースがなくなるだけ

### P1: Sell age cap 導入 (信頼度: ★★★☆☆)
```yaml
sell_age_cap_sec: 25  # null→25
```
**根拠**:
- Sell fast fill (<15s) が最悪のPnL (p60=-3.69bps on 3/25)
- 一方 sell slow fill (>15s) は p30=-0.25bps とほぼ break-even
- 逆説的だが、fast fill は "picked off" の証拠 → 長時間待つ方が良いのではなく、早期にキャンセルして再評価すべき
- 506# で提案済みだが未設定 — 25s で stale sell を早期回収
- **リスク**: fill rate 低下 → ただし fast fill は損失なので「約定しない方がまし」

### P2: Sell clamp ceiling 微増 (信頼度: ★★★☆☆)
```yaml
offset_ceiling_ratio_sell: 0.45  # 0.40→0.45
```
**根拠**:
- 51%が0.40天井に張り付き、pre-clamp avg=0.598
- 0.45に引き上げ → 飽和率 ~30% に低下
- モデルが「危険」と判断した場面で適切に退避可能に
- **リスク**: offset拡大 → fill rate低下 → ただしRC4の「不十分な保護で損失」を直接解決
- **警告**: P0(VG)で先にbase offsetが改善してからが望ましい

### P3: 観察値 — 即時変更不要
- **buy_offset_ceiling (0.35)**: buy側は 3/7日で黒字、67%飽和だが実害少ない → 現行維持
- **skip_gate**: SK=-2.7 vs EX=-0.1 で良好に分離 → 現行維持
- **ATR floor**: 632# で解決済み → 現行維持
- **lot sizing**: 0.001 BTC が Coincheck 最小 → 物理的にこれ以上縮小不可

## 7. 改善シミュレーション (概算)

### P0 (VG recalibration) の期待効果
- fill rate: 約定率 ~19% → ~22% (VG boost=1.0のノーオペ 66%が解消、offset微減で約定確率上昇)
- 追加約定: +20 fills/day × 16K notional × 0.5bps edge = +16 JPY/day

### P1 (sell_age_cap=25s) の期待効果
- fast fill (<15s) sell を 50% 回避: 28 fast fills → 14 fast fills
- 回避した fast fill の損失削減: 14 × 16K × 3.69bps/10000 = +8 JPY/day

### P2 (sell ceiling 0.45) の期待効果
- 現在 ceiling hit の 22/43 sells が適切な offset で取引
- AS損失 50% 軽減: (22 × 16K × 1.5bps / 10000) / 2 = +3 JPY/day

### 合算期待値: +27 JPY/day
月間: +810 JPY → 現在23K資産の +3.5%/month
Death Spiral の減速効果。即座の反転には不十分だが、出血を止める第一歩。

## 8. 結論

### 現状の診断
- **直接要因**: per-fill edge が負 (-0.73 JPY) → 取引するほど資産が減少
- **根本要因**: VG恒常発動でoffset固定 + clamp飽和で保護不足 + sell AS
- **悪化要因**: balance concentration → 37% 機会損失 → 回復機会の喪失
- **死因予測**: 現状のまま推移すれば ~3ヶ月で min_lot (0.001 BTC, ~11K JPY) 未満に到達

### 優先順位
1. **P0 (VG recalibration)**: 最も安全で効果の高い変更。VG の本来の設計意図 (スパイク保護) に回帰させる
2. **P1 (sell_age_cap)**: 506# で提案済み。fast fill AS の直接対策
3. **P2 (sell ceiling)**: P0 の効果を確認してから判断
4. **観察**: fill record の日次モニタリング → per-fill edge が正転したか追跡
