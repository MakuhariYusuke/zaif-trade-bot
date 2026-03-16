# 198# 事後分析: 2026-03-01 朝セッション -53bps ドローダウン

## 1. 概要

2026-03-01 朝セッション (09:04–10:07 JST) において、**12 fills で -53.21bps** を記録し、`daily_drawdown_guard` の hard limit (-50.0bps) により全停止。システムは 10:07 以降 HALT 状態。

前日 (2/28) も -51.21bps で HALT しており、**2 日連続の -50bps 超過**。

## 2. タイムライン

```
01:20  プロセス復帰 (state: day=20260228, pnl=-33.25bps)
01:25  夜間セッション開始 (16 fills, avg=-0.54bps, sum=-8.63bps)
02:22  HALT #1: daily PnL -51.21bps (2/28 day) — 51 fills
09:04  日付リセット: 20260228→20260301 (prev_pnl=-51.21bps, was_halted=True)
09:09  レジーム遷移: ranging → trending_up (BTC ~10,470k)
09:09  朝セッション開始
09:31  レジーム遷移: trending_up → ranging (BTC ~10,440k、30分で -30k下落)
10:02  SOFT drawdown: -47.29bps (**lot 半減失敗**: 0.0010→0.0010)
10:07  HALT #2: daily PnL -53.20bps — 12 fills のみ
14:40+ HALT 継続中 (batch_flush のみ)
```

## 3. 朝セッション全トレード

| Cycle | Side | PnL (bps) | Wait (s) | Reprice | PnL Wait | 備考 |
|-------|------|-----------|----------|---------|----------|------|
| 5208 | buy  | +0.05  | 16.6 | — | 30s | |
| 5212 | sell | -1.82  | 11.1 | — | 45s | fast_fill_defense 発動 |
| 5216 | sell | **-9.94** | 6.1 | — | 45s | **postonly_guard 逆効果** (下記§5.2) |
| 5217 | buy  | -3.94  | 6.0 | — | 15s | fast_fill_defense L2 発動 |
| 5218 | sell | +3.73  | 32.8 | — | 45s | 唯一の sell 勝ちトレード |
| 5223 | buy  | +0.32  | 22.2 | — | 30s | |
| 5224 | sell | -6.87  | 22.5 | — | 90s | velocity=-16.9bps |
| 5225 | buy  | +4.09  | 6.1 | — | 30s | |
| 5226 | sell | -1.10  | 27.5 | — | 90s | |
| 5228 | buy  | **-8.50** | 28.8 | ✓(4.2bps) | 30s | balance_forced buy |
| 5229 | sell | **-23.32** | 34.0 | ✓(17.0bps) | 90s | **最悪トレード** (下記§5.1) |
| 5231 | buy  | -5.91  | 49.5 | ✓(4.5bps) | 30s | balance_forced buy、HALT直前 |

### 3.1 Buy/Sell 分解

| Side | n | Sum | Avg | Worst | Win |
|------|---|-----|-----|-------|-----|
| **BUY**  | 6 | -13.89 | -2.31 | -8.50 | 3 (50%) |
| **SELL** | 6 | -39.32 | -6.55 | -23.32 | 1 (17%) |

**Sell が損失の 74% を占める。**

### 3.2 Reprice 影響

| 区分 | n | Avg PnL | Total |
|------|---|---------|-------|
| Repriced | 3 | **-12.58** | -37.73 |
| Non-repriced | 9 | -1.72 | -15.48 |

**Reprice されたトレードの平均 PnL は 7.3 倍悪い。**

### 3.3 PnL Wait 時間別

| Wait | n | Avg PnL (bps) |
|------|---|---------------|
| 15s  | 1 | -3.94 |
| 30s  | 5 | -1.99 |
| 45s  | 3 | -2.68 |
| 90s  | 3 | **-10.43** |

売り側 (90s wait) は PnL Wait 中に市場がさらに不利方向に動く時間を与えている。

## 4. 市場環境

### 4.1 価格推移
- 09:04: BTC ≈ 10,470,000 JPY
- 09:31: BTC ≈ 10,440,000 JPY (30分で -30k, ca. -29bps)
- 10:07: BTC ≈ 10,415,000 JPY (1h で -55k, ca. -52bps)

**実態**: 朝セッション全体で一貫した下落トレンド。

### 4.2 レジーム検出の遅延

- 09:09 `ranging → trending_up` 検出 (consecutive=3 で遷移に 3 サイクル必要)
- 09:31 `trending_up → ranging` 遷移 (trend_pct=-0.0792)

**問題**: 09:31 の ranging 遷移以降もBTC 価格は下落を続けたが、レジームは「ranging」のまま。velocity=-18.4bps (10:00時点) が示す通り、実態は **trending_down** だが検出されていない。

### 4.3 Volatility

- vol_ratio: 0.155〜0.265 (全サイクル < 0.75 閾値)
- `low_vol_boost` ×1.40 が **12/12 サイクル**で発動 (事実上の常時定数)

### 4.4 VPIN

- 朝セッション全体で高い VPIN (0.66–0.93)
- VG (Volatility Guard) が 11/12 サイクルで発動

### 4.5 スプレッド

- 4 回の「Spread too narrow」(192–444 JPY < min 1000 JPY)
- 狭スプレッド時は offset の実効性がさらに低下

## 5. 根本原因分析

### 5.1 [致命的] stale_order reprice の逆選択増幅 — Cycle 5229 (-23.32bps)

**メカニズム**:

```
10:00:30  Sell 発注 @ 10,435,271 JPY
          ├― low_vol_boost: 0.1800→0.2520 (1.40x)
          ├― VG: 0.2520→0.3000 (velocity=-18.4bps)
          ├― ev_offset: 1.054x (+48 JPY)
          └― vel_offset: 1.77x (+726 JPY)
          (trend_offset: N/A — レジーム=ranging)

10:00:53  [stale_order] 17.0bps drift (mid: 10,433,994→10,416,256)
          → reprice: sell @ 10,416,464 JPY (19,000 JPY 下方へ追随!)

10:01:04  約定 @ 10,416,464 JPY → PnL = -23.32bps (90s wait)
```

**構造的問題**: `order_monitor.py` の `is_drifting_away` ロジック:

```python
is_drifting_away = (
    (side == "buy" and current_mid > mid_at_order)
    or (side == "sell" and current_mid < mid_at_order)
)
```

sell で mid price が下落 → `is_drifting_away=True` → reprice 実行。
しかし sell にとって price 下落は **不利方向** (売り価格が下がる = 損失拡大)。
reprice は「約定しやすく」なるが、**逆選択リスクを増幅**する。

**比較**: Reprice なしなら timeout (77–94s) で unfilled → PnL 0。

### 5.2 [重要] postonly_guard による offset 無効化 — Cycle 5216 (-9.94bps)

**メカニズム**:

```
09:23:03  Sell 計算:
          ├― VG: 0.3600→0.3000 (vpin=0.90)
          ├― ev_offset: 1.109x (+98 JPY)
          └― trend_offset: 2.0x (+1005 JPY)
          → sell price = 10,467,169

09:23:04  [postonly_guard] sell price 10,467,169 <= best_bid 10,468,097
          → adjusted to best_ask 10,469,391 (offset 保護が全て消失)

09:23:10  約定 @ 10,469,391 → PnL = -9.94bps (45s wait)
```

**問題**: 狭スプレッド (best_ask - best_bid ≈ 1,300 JPY) で、計算済み offset を含む sell 価格が best_bid を下回ると、postonly_guard が **best_ask に強制調整**。結果、offset による逆選択防御が完全に無効化。

### 5.3 [重要] daily_drawdown soft limit のロット半減失敗

```
10:02:34  SOFT: daily PnL -47.29bps <= soft limit -30.0bps
          soft lot reduction: 0.0010 → 0.0010 BTC (変化なし!)
```

**原因コード** (`fill_loop_orchestrator.py` L877):

```python
self._current_lot = max(
    self.config.order_quantity,   # = 0.001
    self._current_lot / 2,       # = 0.001 / 2 = 0.0005
)
# max(0.001, 0.0005) = 0.001 ← 変化なし!
```

`order_quantity` が最小ロットのため、半減しても `max()` で元に戻る。**soft limit は事実上の無意味機能**。

### 5.4 [構造的] balance_forced の片側ロック

朝セッションで **7 回** の balance_forced 発生:

```
09:10  JPY insufficient (1,110 < 10,559) → forced sell
09:14  JPY insufficient → forced sell  
09:15  JPY insufficient → forced sell
09:31  BTC insufficient → forced buy
09:33  BTC insufficient → forced buy
09:35  BTC insufficient → forced buy
09:39  BTC insufficient → forced buy
09:55  BTC insufficient → forced buy
10:06  BTC insufficient → forced buy
```

**パターン**: buy → JPY 枯渇 → forced sell 連続 → sell → BTC 枯渇 → forced buy 連続。
0.001 BTC の最小ロットではバランスが常に片側に偏り、**サイド選択の自由度がない**。

### 5.5 [構造的] low_vol_boost の常時発動

vol_ratio が 0.155–0.265 の範囲 (閾値 0.75 との乖離が大きい) で、**全サイクルで 1.40x boost**。  
本来は「低ボラ環境での保守化」だが、閾値が高すぎるため条件付きブーストではなく**定数化**している。

### 5.6 [構造的] レジーム検出遅延と trending_down 未検出

- 09:31 に ranging に遷移後、30 分間で -25k (24bps) 下落
- velocity は -16.9〜-18.4bps を記録 (明確な下落トレンド)
- しかしレジームは「ranging」のまま — **trending_down が検出されない**

結果: trending_down_buy_offset_boost (1.8x) による buy 側の保守化が発動せず、Cycle 5228 (-8.50bps) などの buy 損失に寄与。

### 5.7 [新発見] 90s sell PnL wait の悪影響

sell の post_fill_wait_sec_sell = 90s は「PnL120 > 0 の根拠」で設定されたが、下落トレンド中は **PnL 計測時点での市場がさらに下落** → 見かけ上の PnL が実態より悪化する可能性。

90s wait トレード: avg PnL = **-10.43bps** (n=3) vs 30s wait: avg PnL = **-1.99bps** (n=5)

因果関係の断定は不可 (90s wait は sell であり、sell 自体が悪い環境) だが、下落環境での 90s 保持は損失拡大の疑い。

## 6. 改善提案

### A. stale_order reprice 方向性ガード — **最優先**

| 項目 | 内容 |
|------|------|
| 概要 | sell で mid↓ (不利方向) の drift 時に reprice ではなくキャンセルのみ実行 |
| 期待効果 | Cycle 5229 の -23.32bps → 0bps (unfilled)。推定改善: **+23bps/day** |
| 実装 | `order_monitor.py` の `is_drifting_away` 判定後に新条件追加 |
| リスク | 約定率低下 (sell の unfilled 率増加)。しかし逆選択回避の方が価値が高い |
| 実装コスト | **低** (10 行程度の条件分岐追加) |

**具体案**:
```python
# 現行: drift away → reprice (約定追随)
# 提案: drift away で "不利方向" なら cancel only、"有利方向" なら reprice
# sell: mid↓ は不利 (売値が下がる) → cancel only  
# buy: mid↑ は不利 (買値が上がる) → cancel only
# ※ 現行の is_drifting_away は「注文から離れる方向」= 常に不利方向
# → つまり stale_order reprice 自体が構造的に逆選択増幅
```

**NOTE**: 現行ロジックを再考すると、`is_drifting_away=True` は必ず不利方向。つまり **stale_order reprice は設計上常に逆選択を追う**。「注文が古くなったから追随する」のは MM では合理的だが、post-only maker 戦略では逆に有害。reprice の根本的な見直しが必要。

### B. postonly_guard での offset 保全

| 項目 | 内容 |
|------|------|
| 概要 | postonly_guard 調整後に、元の offset delta を再適用 |
| 期待効果 | Cycle 5216 の -9.94bps 回避。推定改善: **+5–10bps/day** |
| 実装 | `fill_cycle_executor.py` の postonly_guard ブロックに offset 再加算 |
| リスク | offset 再加算で再び best_bid/ask 側に交差する場合のハンドリング |
| 実装コスト | **中** (offset delta の計算と再適用ロジック) |

**具体案**:
```python
# 現行: sell price <= best_bid → price = best_ask (offset 喪失)
# 提案: sell price <= best_bid → price = best_ask + original_offset
#   ただし best_ask + offset が不合理な場合は cancel (spread too narrow)
```

### C. low_vol_boost 段階化 (比例スケーリング)

| 項目 | 内容 |
|------|------|
| 概要 | 二値 1.40 → 連続関数: `1.0 + 0.4 × (1 - vol_ratio/threshold)` |
| 期待効果 | vol_ratio=0.25 時 boost=1.27 (現行 1.40)、微調整の余地確保 |
| 実装 | `maker_price.py` の low_vol_boost ブロック変更 |
| リスク | 小さい (monotonic な関数変更、既存テストは閾値テストのみ) |
| 実装コスト | **低** |

**具体案**:
```python
# 現行: if vol_ratio < threshold → boost = 1.40 (固定)
# 提案: boost = 1.0 + (max_boost - 1.0) * (1.0 - vol_ratio / threshold)
#   vol_ratio=0.0  → boost=1.40 (最大)
#   vol_ratio=0.37 → boost=1.20 (中間)
#   vol_ratio=0.75 → boost=1.00 (閾値、発動なし)
```

### D. trending 中の sell reprice 禁止

| 項目 | 内容 |
|------|------|
| 概要 | レジームが trending (or velocity 高) のとき `stale_max_reprice_sell=0` に動的変更 |
| 期待効果 | trending 中の sell reprice による追随損失防止 |
| 実装 | `order_monitor.py` の stale 判定にレジーム条件追加 |
| リスク | 低 (既に max_reprice_sell=1 の保守設定) |
| 実装コスト | **低** |

### E. balance_forced 冷却期

| 項目 | 内容 |
|------|------|
| 概要 | forced sell/buy 後、一定サイクル数だけ反対 side の forced を制限 |
| 期待効果 | 片側ロック (buy→forced sell→buy→forced sell...) の頻度低減 |
| 実装 | `fill_loop_orchestrator.py` or `side_selector.py` に cooldown 追加 |
| リスク | 中 (在庫不足時にトレード機会を失う) |
| 実装コスト | **中** |

### F. [新規] soft drawdown lot 半減の最小ロット対応

| 項目 | 内容 |
|------|------|
| 概要 | `max(order_quantity, current_lot/2)` のバグ修正 |
| 期待効果 | soft limit (-30bps) 到達時に実際にリスク低減 |
| 実装 | 最小ロットが `order_quantity` と同じ場合は別のリスク低減策を適用 |
| リスク | 低 |
| 実装コスト | **低** |

**具体案**: order_quantity 未満にはできないので、以下のいずれか:
1. **サイクル間隔延長**: soft 発動時に cycle_interval を 2–3 倍に
2. **skip_gate 閾値引き下げ**: soft 発動時に AS probability 閾値を厳格化
3. **sell 一時停止**: soft 発動時に sell (損失の 74%) を N サイクル停止

### G. [新規] sell PnL wait のレジーム連動

| 項目 | 内容 |
|------|------|
| 概要 | trending/高 velocity 環境で sell の post_fill_wait を 90s→30s に短縮 |
| 期待効果 | 下落環境での不必要な保持時間を削減、PnL 計測の精度向上 |
| 実装 | `pnl_measurer.py` にレジーム/velocity 条件追加 |
| リスク | PnL 統計の一貫性が崩れる (要分析基盤側の対応) |
| 実装コスト | **中** |

### H. [新規] レジーム検出感度向上 (trending_down)

| 項目 | 内容 |
|------|------|
| 概要 | velocity ベースの trending_down 検出を補完 (レジーム遷移を速く) |
| 期待効果 | 09:31 以降の下落で trending_down_buy_offset_boost (1.8x) が発動 |
| 実装 | `regime_detector.py` に velocity-based fast transition 追加 |
| リスク | 高 (レジームの頻繁な切り替えは他の戦略に影響) |
| 実装コスト | **高** |

### I. [新規] reprice 時の offset 再計算

| 項目 | 内容 |
|------|------|
| 概要 | stale_order reprice で `compute_maker_price()` を再呼出時に、元のレジーム条件を引き継ぐ |
| 期待効果 | reprice 後も offset 保護が維持される |
| 実装 | `order_monitor.py` の reprice ブロックに offset 引継ぎロジック |
| リスク | 中 (compute の副作用と整合性) |
| 実装コスト | **中** |

## 7. 優先度マトリクス

| 優先度 | 提案 | 期待 PnL 改善 | 実装コスト | リスク |
|--------|------|---------------|-----------|--------|
| 🔴 P0 | **A. reprice 方向性ガード** | +23bps/day | 低 | 低 |
| 🔴 P0 | **F. soft lot 半減バグ修正** | 保険的 | 低 | 低 |
| 🟠 P1 | **B. postonly_guard offset 保全** | +5–10bps/day | 中 | 低 |
| 🟠 P1 | **D. trending 中 sell reprice 禁止** | +5bps/day | 低 | 低 |
| 🟡 P2 | **C. low_vol_boost 段階化** | +2–5bps/day | 低 | 低 |
| 🟡 P2 | **G. sell PnL wait レジーム連動** | 分析精度 | 中 | 中 |
| 🟢 P3 | **E. balance_forced 冷却期** | +3bps/day | 中 | 中 |
| 🟢 P3 | **H. レジーム検出感度向上** | 構造的 | 高 | 高 |
| 🟢 P3 | **I. reprice offset 再計算** | +2bps/day | 中 | 中 |

## 8. 夜間 vs 朝セッション比較

| 指標 | 夜間 (01:25–02:22) | 朝 (09:04–10:07) |
|------|-------------------|-------------------|
| fills | 16 | 12 |
| avg PnL | -0.54bps | **-4.43bps** |
| sum PnL | -8.63bps | **-53.21bps** |
| reprices | 3 (19%) | 6 (**50%**) |
| repriced avg PnL | N/A | -12.58bps |
| balance_forced | 少 | **7回** |
| fast_fill_defense | 少 | **5回** |
| Spread too narrow | 0 | **4回** |

**朝のボラティリティと方向性リスクは夜間の 8 倍の損失を生んだ。**

## 9. データ付録

### 9.1 全 offset chain (朝セッション sell のみ)

| Cycle | base | low_vol | VG | ev_off | vel_off | trend_off | postonly | PnL |
|-------|------|---------|-----|--------|---------|-----------|---------|-----|
| 5212 | 0.18 | — | — | 1.17 | — | — | — | -1.82 |
| 5216 | 0.36 | — | 0.30(vpin) | 1.109 | — | 2.0(+1005) | **→best_ask** | -9.94 |
| 5218 | — | — | — | — | 13.79→2.15 | 2.0(+2606) | — | +3.73 |
| 5224 | 0.18 | 0.25(1.40) | 0.30(vel) | 1.084 | — | — | — | -6.87 |
| 5226 | — | 0.06(1.40) | — | 1.084 | — | — | — | -1.10 |
| 5229 | 0.18 | 0.25(1.40) | 0.30(vel) | 1.054 | 9.20→1.77 | — | — | -23.32 |

### 9.2 レジーム遷移とタイミング

```
09:04  day reset (ranging)
09:09  ranging → trending_up (consecutive=3)
09:31  trending_up → ranging (trend_pct=-0.0792, stability=3)
10:07  HALT (still ranging — trending_down 未検出)
```

velocity at key points:
- 09:28: -31.2bps (Cycle 5218 VG)
- 09:43: -16.9bps (Cycle 5224 VG)
- 10:00: -18.4bps (Cycle 5229 VG)
- 10:00 (reprice): -17.4bps

## 10. レビュー依頼事項

本文書に基づき、以下の観点でレビューを依頼:

1. **提案 A–I の優先順位は妥当か?** 特に A (reprice 方向性ガード) の P0 判定
2. **stale_order reprice の根本的な設計思想**: maker 戦略において「約定追随」は本当に必要か?
3. **soft drawdown lot 半減のバグ (F)**: 修正方法として cycle 間隔延長 vs AS 閾値厳格化 vs sell 停止のどれが適切か?
4. **low_vol_boost 閾値 0.75 の妥当性**: vol_ratio が常時 0.2–0.3 なら閾値を 0.40 程度に下げるべきか?
5. **見落としている根本原因はないか?** (特にモデル予測精度、ev_score の有効性など)
6. **提案間の相互作用**: A と D は reprice 関連で重複あり。B と I も offset 保全で重複。統合実装すべきか?
