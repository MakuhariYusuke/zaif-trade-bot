# 315# rpt: Ceiling / Ratio Semantics 調査報告

> **種別**: rpt (調査報告)  
> **フェーズ**: ph2 (G1.1-exec)  
> **注記**: 314# §6 実行計画の「Phase 1 (T1: 調査)」成果物。000# §2 の ph1 (G1-info) とは異なる。

**実施日**: 2026-03-07  
**対象Issue**: 312# B2 (ceiling不適用疑惑), 313# R2 (ratio意味不一致), 314# B1 (ratio方向二重性)

---

## 1. Executive Summary

**結論: ceiling は正常に動作している。しかし fill record の `effective_offset_used` は信頼できない指標である。**

fill record に記録される `effective_offset_used` は、maker_price.py での ceiling 適用後に fill_cycle_executor.py が 5 段の追加 multiplier を適用した結果であり、ceiling を大幅に超過した値が記録される。しかし、**実際に発注される価格**は ceiling の制約を受けた上で追加 multiplier により保守方向（mid から離れる方向）に調整されているため、価格自体は ceiling が意図する保守性を維持している。

---

## 2. T1-1: Ceiling Root Cause 分析

### 2.1 データ上の現象

311# 深掘り分析（旧式）で sell filled の effective_offset_used を調査:
- 196/197 レコードが ceiling (0.15) を超過
- 最大値: 2.0877（ceiling の 14 倍）
- 中央値: 0.30

**312# B2 の疑問**: ceiling が適用されていないのでは？

### 2.2 maker_price.py での ceiling 適用（正常）

[maker_price.py](../../scripts/v460/lib/maker_price.py#L1661) の `compute()` メソッド内:

```python
# L1661: ceiling 適用（13段オフセットパイプラインの最終段）
if _ceil > 0 and effective_offset_ratio > _ceil:
    effective_offset_ratio = _ceil   # ← 0.15 にクリップ
```

**13段パイプライン**:
base → inv_skew → sell_floor → AS_shift → regime → spread_adapt → kyle → amihud → vol_guard → imb_risk → buy_as_guard → sell_hour → loss_boost → FFD → **ceiling**

ceiling はパイプライン最終段で確実に適用される。`compute()` の戻り値 `(price, spread, ratio)` 時点で ratio ≤ 0.15 は保証される。

### 2.3 fill_cycle_executor.py の POST-ceiling multiplier（根本原因）

[fill_cycle_executor.py](../../scripts/v460/lib/fill_cycle_executor.py#L999-L1085) で `_run_single_cycle_v3` が maker_price.compute() 後に 5 段の追加 multiplier を適用:

| # | 名称 | 行 | aggressive | 方向 (mult>1) | 典型値 |
|---|------|----|-----------|---------------|--------|
| 1 | EV offset (193#) | L999 | **True** | mid に接近 | 0.5–2.0 |
| 2 | Velocity offset (195#) | L1023 | False | mid から離反 | 1.5–3.0 |
| 3 | Trending offset (196#) | L1040 | False | mid から離反 | 1.5–5.4 |
| 4 | Toxicity offset (240#) | L1058 | False | mid から離反 | 1.1–2.0 |
| 5 | VG supplement (202#) | L1085 | False | mid から離反 | 1.5 |

**KEY**: multiplier #2-#5 は `aggressive_when_multiplier_gt_one=False`（デフォルト）。mult > 1.0 の場合:
- buy: price -= delta (より低い価格 = より保守的)
- sell: price += delta (より高い価格 = より保守的)

つまり **価格は mid から離れる（保守的）** が、**recorded ratio は上昇する**。

### 2.4 具体例（sell trending regime）

```
maker_price.compute():
  ratio = 0.12 → ceiling 適用 → ratio = 0.12 (< 0.15, OK)
  sell_price = best_ask - spread * 0.12

fill_cycle_executor 196# trending offset (mult=5.4):
  old_offset = spread * 0.12 = 100円
  new_offset = 100 * 5.4 = 540円
  delta = 440円
  sell_price += delta → 売り注文が440円 "高く" なる（mid から離れる）
  recorded_ratio = 0.12 * 5.4 = 0.648

結果:
  実際の発注価格: mid からさらに遠い（非常に保守的）
  recorded effective_offset_used: 0.648（ceiling の 4.3 倍！）
```

### 2.5 結論

| 項目 | 状態 |
|------|------|
| ceiling 機能 | **正常** — maker_price.py で確実に適用 |
| 312# B2 の原因 | post-ceiling multiplier が ratio を再膨張させる |
| 実際の価格への影響 | multiplier #2-5 は保守方向 → 価格は ceiling 意図より保守的 |
| fill record の ratio | **信頼不可** — price 位置と ratio の関係が単調でない |

---

## 3. T1-2: Ratio Semantics 分析

### 3.1 二つのシステムの ratio 意味

**maker_price.py（L1515-1682）**:
```
ratio ↑ → offset ↑ → price = bid + spread * ratio (buy)
                   → price = ask - spread * ratio (sell)
→ ratio ↑ = mid に接近 = アグレッシブ
```

**fill_cycle_executor.py（L520-562）** `_apply_offset_multiplier`:
- `aggressive_when_multiplier_gt_one=True` (EV offset のみ):
  - mult > 1: price を mid に向けて移動 → **consistent with maker_price**
- `aggressive_when_multiplier_gt_one=False` (velocity/trending/toxicity/VG-supp):
  - mult > 1: price を mid から離す → **REVERSED direction**
  - でも ratio *= mult → ratio は上昇

### 3.2 ratio の意味一覧

| ステージ | ratio ↑ の意味 |
|----------|----------------|
| maker_price compute() | アグレッシブ（mid 接近） |
| EV offset (193#) | アグレッシブ（mid 接近）— consistent |
| velocity offset (195#) | **保守的（mid 離反）— REVERSED** |
| trending offset (196#) | **保守的（mid 離反）— REVERSED** |
| toxicity offset (240#) | **保守的（mid 離反）— REVERSED** |
| VG supplement (202#) | **保守的（mid 離反）— REVERSED** |

### 3.3 fill record の `effective_offset_used` の正体

fill_cycle_executor.py L329:
```python
"effective_offset_used": effective_offset_ratio
```

この ratio は全 multiplier 適用後の累積積。EV offset は mid 接近、残り 4 つは mid 離反という **相反する方向成分が混在** しており、この値だけからは:
1. 発注価格の mid からの距離は推定不能
2. アグレッシブ度も保守度も判定不能

### 3.4 影響範囲

**旧 spread_as_decomposition の致命的誤り**:
```python
# 旧式（完全に誤り）
sc_bps = spread_bps * effective_offset_ratio
# → 0.446 ratio を「spread の 44.6% を capture」と解釈
# → 実際は「ratio が膨張しただけで価格は mid 外」
```

**314# T0-1 で修正済み**:
```python
# 新式（fill_price / mid_at_fill 直接計算）
if side == "sell":
    sc_bps = (fill_price - mid_at_fill) / mid_at_fill * 10000
else:
    sc_bps = (mid_at_fill - fill_price) / mid_at_fill * 10000
```

---

## 4. T1-3: Production Impact Assessment

### 4.1 現行ロジックの評価

post-ceiling multiplier の振る舞い自体は **意図通り**:
- Trending regime / high velocity / high toxicity → 保守的に発注 → AS リスク低減
- EV 高スコア → アグレッシブに発注 → 約定率向上

**問題は ratio の記録方法のみ**。発注価格自体は合理的。

#### 314# B1 (maker_price.py 内部パイプライン) への見解

314# 計画書 §4 B1 は maker_price.py 内部の boost ステージ（VG, regime, sell_hour）が
ratio を増加させ「防御のつもりで攻撃的にしている」と指摘した。

これは **概念的には正しい**: maker_price.py 内では ratio↑ = mid 接近 = アグレッシブ。
しかし **実務的には ceiling (0.15) が全ステージの増加を制限** し、その後
fill_cycle_executor の非 aggressive multiplier が価格をさらに保守方向に押すため、
最終的な発注価格は十分に保守的。

B1 の方向修正（全 boost を ratio↓ 方向に反転）は理論的には正しいが、
現行の ceiling + fill_cycle_executor 補償により実害がないため **NOT RECOMMENDED**。

#### sell_offset_floor (0.30) vs offset_ceiling_ratio (0.15) の矛盾

**316# 追加発見**: YAML で `sell_guard.offset_floor: 0.30`、`offset_ceiling_ratio: 0.15`。
パイプライン順序は floor → ... → ceiling のため、ceiling が常に勝ち **floor は死んだ設定**。

- sell_floor (246#): ratio ≥ 0.30 を保証（最低限のアグレッシブさ確保）
- ceiling (306#): ratio ≤ 0.15 で制限（過剰アグレッシブ防止）
- 0.30 > 0.15 のため floor は効果なし

**影響**: 実害なし（ceiling が勝つ = より保守的 = AS 保護）。
ただし設定の意図が矛盾しており、将来の混乱源になり得る。
floor と ceiling の関係を YAML コメントで明示するか、ceiling を floor 以上に引き上げるべき。

### 4.2 要修正箇所（計測系）

| 対象 | 問題 | 修正方針 |
|------|------|----------|
| `effective_offset_used` | 方向混在で解読不能 | `mid_distance_bps` 追加提案 |
| Offset quintile 分析 | ratio quintile が価格位置を反映しない | fill_price / mid_at_fill 基準に変更 |
| `spread_at_order` bps | L243: `spread / mid_at_fill * BPS` — mid_at_fill は fill 検出時の mid であり order 時の mid ではない | 軽微（%差は小さい） |

### 4.3 要修正判定

| 修正候補 | 優先度 | 理由 |
|----------|--------|------|
| fill record に `mid_distance_bps` 追加 | P2 | 分析の正確性向上。現行運用に影響なし |
| `effective_offset_used` の ratio 意味統一 | **NOT RECOMMENDED** | 4 つの multiplier の保守方向を反転させると AS リスクに直結。現行動作は意図通り |
| ceiling 値の引き上げ | **NOT RECOMMENDED** | ceiling は maker_price 内で正常に機能。post-ceiling mult は正しく保守化を行っている |

### 4.4 mid_at_fill の検出レイテンシバイアス

[pnl_measurer.py](../../scripts/v460/lib/pnl_measurer.py#L90) で `mid_at_fill = await get_mid_price()` は fill **検出時**（fill 実行時ではない）の mid。

- Sell fill: 市場上昇 → fill 検出時 mid > fill_price → 負の spread capture
- Buy fill: 市場下落 → fill 検出時 mid < fill_price → 負の spread capture

311# 修正後の §3 結果:
- Sell: spread_capture = **-0.502 bps**
- Buy: spread_capture = **-0.485 bps**

両サイドとも負 → 検出レイテンシの systematic bias。本質的な AS cost 計算には `order_price` と `best_bid/best_ask at order time` を使った **注文時 mid** ベースの計算が理想。

---

## 5. Actionable Recommendations

### Phase 2 で実施すべき項目（優先度順）

1. **P1: Offset quintile 分析の基盤修正** — 311# §8 を fill_price/mid ベースに変更（T0-1 と同様の思想）
2. **P2: fill record に `mid_at_order` フィールド追加** — order 発注時点の mid を記録し、spread capture の正確な計算を可能にする（検出レイテンシバイアス排除）
3. **P2: `mid_distance_bps` フィールド追加** — `|fill_price - mid_at_order| / mid_at_order × 10000` を記録

### Phase 2 で実施しない項目

- ❌ ceiling 値変更 — 正常動作中
- ❌ post-ceiling multiplier 方向変更 — 意図通りの保守化
- ❌ `effective_offset_used` 計算変更 — 後方互換性破壊のリスク

---

## 6. 修正後 311# 分析結果サマリ

| 指標 | 旧値（バグ有） | 新値（修正後） | 変化 |
|------|---------------|---------------|------|
| Sell spread_capture | +0.862 bps | **-0.502 bps** | -1.36 bps |
| Buy spread_capture | +0.278 bps | **-0.485 bps** | -0.76 bps |
| Sell AS cost | +1.141 bps | **-0.121 bps** | -1.26 bps |
| Buy AS cost | +0.572 bps | **-0.180 bps** | -0.75 bps |
| Sell efficiency | -32.4% | **76.0%** | 根本的に異なる |
| Buy efficiency | -105.7% | **63.0%** | 根本的に異なる |

> **注**: 314# 計画書の「8.3 倍過大評価」は ratio 0.446 と (0.5-0.446)=0.054 の比率
> であり、bps 値ではない。旧式 `sc_bps = spread_bps × ratio` の実際の出力は
> 上記の通り +0.862 bps（ratio の膨張で不正確だが、10 bps 級ではなかった）。

**解釈**: 旧式は ratio の膨張により spread capture を過大評価（+0.9 bps を +0.3 bps の PnL と比較 → "capture は十分" と誤認）。修正後は検出レイテンシにより -0.5 bps の negative spread capture が明らかに。efficiency が正（76%/63%）なのは分子・分母ともに負であるため「損失の76%が entry 由来」の意味。

post-310# の fill data が蓄積され次第、`--git-sha dcc3064` で 310# 単独の効果を検証可能。
