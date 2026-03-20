# 503# Sell/Buy 損益要因分析 (2026-03-14〜20)

## サマリー

7日間（3/14-3/20）の fill_test レコード分析。757 fills (buy:382, sell:375)。

| Side | Total PnL | Fill数 | WR | 主因 |
|------|-----------|--------|-----|------|
| **BUY** | **-109.71 JPY** | 382 | 47.1% | ranging での逆選択、fast fill 損失 |
| **SELL** | **-206.28 JPY** | 375 | 49.6% | cross-venue 未適用（構造的）、slow fill 逆選択 |

**両サイド合計: -316.0 JPY / 7日 = -45.1 JPY/日**

---

## 重大発見: Cross-Venue Lead-Lag の非対称性

### 事実

3/14-3/20 の全データで:
- **Buy**: cross_venue_lead_lag_applied = True が 210/382 fills (55%)
- **Sell**: cross_venue_lead_lag_applied = True が **22/375 fills (5.9%)**

### 根本原因

BitFlyer mid < Coincheck mid が恒常的 (409/451 = 91%)。

```
gating_spread = bf_mid - cc_mid < 0  (平均 -3.29 bps)
  → direction = "down"
  → adverse_side = "buy"
```

`maker_risk_guards.py` L236-241 の guard 適用条件:
```python
if hint.adverse_side != side:
    return effective_offset_ratio  # no-op
```

adverse_side="buy" 固定 → **sell サイクルでは guard が常にスキップ**される。

### Cross-Venue の効果 (buy 側)

| 状態 | fills | Total PnL | Avg PnL |
|------|-------|-----------|---------|
| CV applied | 210 | **+8.99** | +0.043 |
| CV not applied | 172 | **-118.70** | -0.690 |

**cross-venue 適用時は黒字、非適用時は赤字** → 効果は実証済み。
sell 側で適用されないのは収益機会の喪失。

### 修正案

`adverse_side` は元来「逆選択を受ける側」を意味する。
現在の direction-based 定義は正しいが、**sell 側にも保護が必要**。

オプション:
1. **両サイド適用**: adverse_side check を除去し、常に guard を実行
2. **対側保護**: sell 側に adverse_side="sell" (direction="up") 時の逆ロジックを追加
3. **設定分離**: sell 側専用の cross-venue threshold/boost を導入

推奨: **オプション1** — guard は逆選択回避の offset_boost であり、側を問わず適用すべき。

---

## Buy 側の損益要因

### 1. Regime 別

| Regime | PnL | fills | WR | Cross-venue |
|--------|-----|-------|-----|-------------|
| ranging | **-83.59** | 332 | 46.4% | 181/332 |
| trending_up | -29.11 | 27 | 48.1% | 10/27 |
| trending_down | +2.98 | 23 | 52.2% | 19/23 |

**ranging で大幅赤字** (-83.59 JPY, 332 fills)。buy の損失の76%が ranging 由来。

### 2. Offset バケット別

| Offset | PnL | fills | WR |
|--------|-----|-------|----|
| <0.10 | -13.63 | 11 | 27% |
| 0.10-0.19 | -17.86 | 47 | 43% |
| **0.19-0.25** | **-103.11** | **274** | **47%** |
| >=0.25 | **+24.89** | 50 | 52% |

offset 0.19-0.25 に 72% の fills が集中し、最大の損失源。
offset >= 0.25 のみ黒字 → **offset が狭すぎる**。

### 3. Fast Fill の逆選択

| 約定速度 | fills | PnL | Avg PnL |
|----------|-------|-----|---------|
| <10s (fast) | 156 | **-64.56** | -0.41 |
| >=30s (slow) | 74 | **+11.02** | +0.15 |

fast fill = 即約定 = **逆選択の兆候**。
価格がこちらに向かって動いている最中に約定 → 30s 後に反転。

### 4. EV スコアの予測精度

| EV | fills | PnL |
|----|-------|-----|
| EV > 0 | 309 | **-100.61** |
| EV <= 0 | 73 | -9.10 |

**EV > 0 の fills が大幅赤字** → EV スコアの予測方向が buy 側で逆転している可能性。

### Buy 側の主要課題

1. **ranging offset が狭い**: 0.20 付近に集中、>=0.25 でのみ黒字
2. **fast fill 逆選択**: 即約定 = 不利な方向への price impact
3. **cross-venue 非適用時の損失大**: CV なしで -118.70 JPY

---

## Sell 側の損益要因

### 1. Regime 別

| Regime | PnL | fills | WR | Cross-venue |
|--------|-----|-------|-----|-------------|
| ranging | **-220.65** | 326 | 50.0% | 18/326 |
| trending_down | +3.94 | 23 | 52.2% | 0 |
| trending_up | +10.44 | 26 | 42.3% | 4 |

**ranging で -220.65 JPY** — sell 損失の 107% が ranging 由来（他 regime は黒字）。

### 2. Offset バケット別

| Offset | PnL | fills | WR |
|--------|-----|-------|----|
| 0.10-0.19 | **+138.58** | 43 | 67% |
| 0.19-0.25 | -231.53 | 126 | 47% |
| >=0.25 | -113.33 | 206 | 48% |

**offset < 0.19 の sell が大幅黒字**（+138.58 JPY, WR 67%）。
リスク制限のための wide offset (>=0.25) が逆に損失拡大。

### 3. Slow Fill の逆選択

| 約定速度 | fills | PnL | Avg PnL |
|----------|-------|-----|---------|
| <10s (fast) | 172 | **+43.14** | +0.25 |
| >=30s (slow) | 61 | **-173.22** | -2.84 |

**buy とは逆パターン**: sell は fast fill が黒字、slow fill が赤字。
slow fill = 注文が板に長時間滞留 → 価格が売り方向に動き続けて逆選択。

Worst slow sell: **-72.65 JPY** (wait=34s, ranging, offset=0.200)

### 4. Dynamic Kill の影響

7日間のスキップ内訳:

| 理由 | Buy | Sell |
|------|-----|------|
| sell_dynamic_kill | - | **312** |
| skip_gate | 67 | 167 |
| final_clamp_hard_skip | - | 36 |

sell_dynamic_kill が **312回** 発動 → sell の取引機会を大幅に制限。

kill 閾値の非対称性:
- sell: **-0.5 bps** (厳しい)
- buy: -0.8 bps (緩い)
- inv_relaxation_scale: sell=0.4 / buy=0.5

### Sell 側の主要課題

1. **cross-venue 未適用（構造的）**: 保護なしで逆選択に晒される
2. **slow fill 逆選択**: 30s 以上の sell が -173.22 JPY (-2.84 JPY/fill)
3. **dynamic_kill 過剰発動**: 312 回で sell 機会の 45% を喪失
4. **wide offset 逆効果**: >=0.25 offset が損失拡大（板に取り残される）

---

## Buy/Sell 共通の問題

### Ranging Market が全体損失の97%

| Side | Ranging PnL | 全体比 |
|------|-------------|--------|
| Buy | -83.59 | 76% |
| Sell | -220.65 | 107% |
| **合計** | **-304.24** | **96%** |

trending 時は **両サイド合計で -11.75 JPY** → ほぼブレイクイーブン。
**損失のほぼ全てが ranging market に集中**。

### Adverse Selection パターン

- **Buy**: fast fill で逆選択 (価格がさらに下がる)
- **Sell**: slow fill で逆選択 (価格がさらに上がる)

これは market microstructure の典型的パターン:
- 即約定 = 有利な方向に動いている → その方向に継続 → maker 損失
- 長時間待ち = 板に滞留 → いずれ成行注文で約定 → 情報独占者の注文

---

## 優先改善案

### P0: Cross-Venue Guard の売り側適用 (推定影響: +50-100 JPY/week)

cross_venue_lead_lag_guard を sell 側にも適用。
buy 側で CV applied が +8.99 vs not -118.70 の差を考えると、
sell 側にも同等の保護効果が期待できる。

### P1: Sell Dynamic Kill 閾値の緩和 (推定影響: +30-50 JPY/week)

- sell threshold: -0.5 → -0.8 bps (buy と統一)
- inv_relaxation_scale: 0.4 → 0.5 (buy と統一)

### P2: Ranging Offset の見直し (推定影響: +40-80 JPY/week)

- buy: 0.20 → 0.25 (>=0.25 バケットのみ黒字)
- sell: wide offset 見直し — offset < 0.19 が黒字の事実

### P3: Slow Fill / Fast Fill 対策

- sell slow fill: timeout_sec 短縮 or stale order 検出の早期化
- buy fast fill: fast_fill_defense の buy 側閾値強化

---

## 設定値の非対称性一覧（参考）

| 項目 | Buy | Sell | 備考 |
|------|-----|------|------|
| base_offset_ratio | 0.05 | 0.18 | sell が 3.6倍広い |
| order_timeout_sec | 90s | 75s | sell が短い |
| post_fill_wait_sec | 30s | 90s | sell が 3倍長い |
| dynamic_kill threshold | -0.8 bps | -0.5 bps | sell が厳しい |
| inv_relaxation_scale | 0.5 | 0.4 | sell が kill されやすい |
| trending_up offset_boost | 0.7 | 1.8 | sell 逆張り時 2.6倍 |
| ranging_offset_discount | 1.15 | 0.85 | buy 拡大 / sell 縮小 |
| fast_fill threshold_sec | 8.0 | 15.0 | sell は検出遅い |
| fast_fill offset_boost | - | 2.5 | sell に強い防御 |
| cross-venue applied率 | 55% | 5.9% | 構造的非対称 |
