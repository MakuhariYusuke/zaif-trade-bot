# 171# Phase 2: Sell Decision Flow & Guard Paradox 技術精査レポート

**日付**: 2026-02-27  
**対象期間**: 2/13〜2/27 (15日間, 4365 records)

---

## 1. Sell Offset 算出フロー完全解析 (`maker_price.py`)

### 1.1 Sell Offset パイプライン (実行順)

```
① base_offset_ratio_sell (config: side_offset.sell = 0.18)
  ↓
② Inventory Skewing (162#): inv_net_imbalance × -1 × max_factor
   → buy偏重(+) → sell offset縮小(促進), sell偏重(-) → sell offset拡大(抑制)
  ↓
③ sell_offset_floor ハードガード (088#): max(offset, 0.20) ← 165# SO-1
   ★ 注意: ②で0.18→0.12に縮小されても③で0.20にフロアクランプ
  ↓
④ _apply_regime_boosts():
   - trending: regime_trending_offset_boost_sell (config値) を乗算
   - high_vol: regime_high_vol_offset_boost を乗算
   - ranging: regime_ranging_offset_discount を乗算
   - low_vol_boost: vol_ratio < threshold → low_vol_offset_boost 乗算
   - unknown_buy_guard: sell には無関係
  ↓
⑤ _apply_spread_adaptive():
   - narrow_spread: narrow_spread_boost_sell 乗算
   - wide_spread: wide_spread_ratio 乗算
   - ★ 事後 sell_offset_floor 再適用 (091#): ⑤の結果が0.20を下回れば再クランプ
  ↓
⑥ _apply_volatility_guard():
   - velocity > threshold or VPIN > threshold → offset_boost_factor 乗算
   - 168# damping: InvSkew が sell 緩和中なら VG boost を抑制
  ↓
⑦ _apply_imbalance_risk():
   - sell && imb > threshold → AS risk → offset_boost 乗算
   - imb >= skip_threshold → ValueError (注文抑止)
  ↓
⑧ FastFillDefense: per-side boost 乗数を乗算
  ↓
⑨ 最終価格: price = best_ask - max(min_offset_jpy, spread × effective_offset_ratio)
   spread guard: price <= best_bid → fallback to best_ask
```

### 1.2 実測 Offset 分布

| Quartile | Offset 範囲 | 件数 | 平均 PnL (bps) |
|----------|-------------|------|-----------------|
| Q1 (低) | 0.000–0.252 | 209 | **-1.225** |
| Q2 | 0.252–0.300 | 456 | -0.220 |
| Q3 | 0.300–0.300 | 359 | -0.302 |
| Q4 (高) | 0.300–1.000 | 523 | -0.289 |

**発見**: Q1 (offset < 0.252) の sells が **-1.225 bps** と突出して悪い。sell_offset_floor = 0.20 の設定にも関わらず 0.252 未満が 209 件存在 → InvSkew/FFD/Spread Adaptive が floor を迂回する経路、または floor 導入前のレコードが混在。

---

## 2. Guard Paradox 検証

### 2.1 定義
**Guard Paradox** = ガードが「中程度のリスク」をフィルタし、結果的に「最悪のケースだけが実行に到達」する逆選別現象。

### 2.2 VG (Volatility Guard) — **パラドックス確認**

| 条件 | 件数 | 平均 PnL (bps) |
|------|------|-----------------|
| VG 発動 sell | 188 | **-0.608** |
| VG 非発動 sell | 734 | -0.443 |

VG 発動時の sell は非発動時より **37%** 悪い。VG は offset をブーストして防御するが、**それでも約定した** ということは、市場が offset を超えて悪方向に動いた超攻撃的な環境であることを意味する。

**解釈**: VG 自体はリスクを正しく検知しているが、VG で offset を上げても約定してしまうケースは、VG の保護範囲を超えた adversarial 環境。VG が「中間リスク」を除外し、「極端リスク」だけが残るという **classic guard paradox** が発生している。

### 2.3 balance_forced_switch — **パラドックス確認**

| 条件 | 件数 | 平均 PnL (bps) |
|------|------|-----------------|
| balance_forced sell | 122 | **-0.659** |
| normal sell | 800 | -0.449 |

balance_forced sell は normal sell より **47%** 悪い。rescue モードで実行された sell は、本来 skip されるべき不利な環境で無理に実行されている。

### 2.4 ガード全体の sell 側キャンセルバランス

| 項目 | Sell | Buy | Sell/Buy 比 |
|------|------|-----|-------------|
| 総レコード | 2628 | 1727 | 1.52x |
| ハードブロック | 995 | 87 | **11.4x** |
| skip_gate | 249 | 225 | 1.1x |
| 約定 | 922 | 937 | 0.98x |
| 全キャンセル | 1706 | 790 | **2.16x** |

**sell 側に 11.4 倍のハードブロック** が集中。trending_sell_skip (396), balance_forced_skip (377), sell_dynamic_kill (167) が三大要因。

---

## 3. Sell Decision フロー全体図

```
サイクル開始
  ↓
[1] _next_side() → sell が候補
  ↓
[2] time_filter (side別) → 時間帯ブロック
  ↓
[3] balance pre-flight check
  ├── sell 残高不足 → opposite=buy に切替
  │   ├── buy も不足 → preflight_insufficient
  │   └── buy OK → _balance_forced = True
  │       ├── skip_balance_forced=True (config)
  │       │   ├── 元side(buy)も不足 → 実行許可 (one_sided_balance)
  │       │   ├── rescue_enabled → offset×2 で実行
  │       │   └── 両方OK → balance_forced_skip ★377件
  │       └── skip_balance_forced=False → 即実行
  └── sell 残高 OK → 正常パス
  ↓
[4] trending_sell_skip (155# §9) ★396件
  ├── trending regime && side==sell
  │   ├── trending_up_only && trending_down → 通過
  │   ├── 連続skip >= max_consecutive(20) → 強制実行
  │   ├── buy側残高不足 → リバランス緩和 (166# HF4)
  │   └── それ以外 → trending_sell_skip
  └── 非trending → 通過
  ↓
[5] sell_dynamic_kill (133# P0-10) ★167件
  ├── rolling 50件 平均PnL < -0.5bps → 売停止 + cooldown 10サイクル
  │   regime_thresholds: trending_up: -0.1, trending_down: -1.0
  └── 上記以外 → 通過
  ↓
[6] run_single_cycle() 内:
  ├── maker_price.compute(sell) → offset 算出 (§1 パイプライン)
  ├── skip_gate_evaluator.evaluate(sell) ★249件
  │   ├── rule_skip_unknown_sell (124#)
  │   ├── rule_velocity_sell_skip (165# AS-R1) ★25件
  │   └── ML model P(AS) >= threshold → skip_gate
  ├── 発注 → ポーリング
  │   ├── 約定 → PnL 計測 (post_fill_wait_sec_sell=90s)
  │   │   └── _track_sell_pnl → sell_dynamic_kill 用追跡
  │   ├── timeout (75s) → cancel ★122件
  │   ├── postonly_reject → cancel ★41件
  │   └── sell_guard_reject (spread > 5000 JPY) ★30件
  └── レコード記録
```

---

## 4. balance_forced_skip 詳細分析

### 4.1 メカニズム

1. `_next_side()` が sell を返す (前回 buy だったため交互)
2. `_check_balance_for_side("sell")` → 残高不足
3. 反対側 buy の残高をチェック → OK → `next_side = "buy"`, `_balance_forced = True`
4. `skip_balance_forced=True` (config) → buy 側も残高 OK なので「両方 OK = スキップ」
5. `cancel_reason = "balance_forced_skip"` で記録

### 4.2 データ所見

- **全 377 件が side=sell**: sell 残高不足 → buy に切替 → スキップ。一方向にしか発生しない
- **全 377 件の regime=None**: regime 情報が記録されていない (スキップ時に regime 未セット)
- **max_consecutive = 0**: rescue_enabled=true により連続カウンタがすぐリセット
- **8.6% のサイクルが dead time**: 発注も約定もしない空サイクル

### 4.3 Side Imbalance 影響

balance_forced_skip は **sell 残高不足時にのみ発火**。これは：
- buy 約定が売り残高を消費している
- sell が十分実行されておらず、在庫が buy 偏重になっている
- Inventory Skewing (162#) が部分的に対処しているが、根本的な sell 残高不足は解消できていない

**構造的問題**: sell ガードが sell 発注機会を 2.16x 削減 → buy 偏重 → sell 残高不足 → balance_forced_skip → さらに sell 機会損失、という **正のフィードバックループ**。

---

## 5. sell_dynamic_kill 追跡内容 (`ztb/risk/sell_dynamic_kill.py`)

### 5.1 メカニズム

`DynamicKillManager` (buy/sell 共用):
1. **track(pnl_bps)**: fill ごとに PnL を `_pnl_history` (list) に追加。最大 window×3 件保持。
2. **check_kill(regime)**: 
   - cooldown 中 → killed=True (残り cooldown をデクリメント)
   - `_pnl_history[-window:]` の平均を計算
   - 平均 < threshold_bps → killed=True, cooldown = resume_window
   - regime_thresholds に一致する regime なら専用閾値を使用

### 5.2 Sell Kill Config (実運用値)

| パラメータ | 値 | 意味 |
|-----------|-----|------|
| window | 50 | 直近 50 sell fills の平均 |
| threshold_bps | -0.5 | 平均 < -0.5bps で kill |
| resume_window | 10 | kill 後 10 サイクル (≈20分) cooldown |
| trending_up | -0.1 | trending_up では ≈0 未満で即 kill |
| trending_down | -1.0 | 下降トレンドでは sell に寛容 |

### 5.3 Kill 統計
- sell_dynamic_kill: 167 件 (3.8%)
- buy_dynamic_kill: 77 件 (1.8%) → sell は buy の **2.2x** の kill 頻度

---

## 6. Skip Gate 精度分析

### 6.1 全体統計

| 項目 | 値 |
|------|-----|
| 総スキップ | 499 |
| sell スキップ | 274 (54.9%) |
| buy スキップ | 225 (45.1%) |
| ML model skip | 474 |
| Rule-based skip | 25 (velocity_sell) |
| Counterfactual PnL 利用可能 | **0件** (actual_pnl_30s が未記録) |

### 6.2 精度評価の限界

`actual_pnl_30s` が skip 時に記録されていないため、**skip gate の precision/recall を直接測定できない**。hindsight filter (hindsight_filter.py) が別途カバーしているが、fill_records 自体には含まれていない。

### 6.3 間接的精度評価

Skip gate が有効に機能しているかの間接指標:
- sell が 274 件 skip されているが、sell avg PnL は依然 **-0.477 bps** → skip gate を通過した sell でも大幅に負
- buy が 225 件 skip されているが、buy avg PnL は **-0.181 bps** → gate 通過 buy はそれなりのパフォーマンス

**sell skip gate の精度が不十分な可能性**: skip 後の残存 sell がまだ -0.477bps と悪い。ただしこれは skip なしの場合との比較データがないため確定できない。

### 6.4 Side 別モデル使用状況

| モデル | 件数 |
|--------|------|
| primary (unified) | 239 (224+15) |
| primary:side_sell | 134 |
| primary:side_buy | 101 |
| rule (velocity) | 25 |

---

## 7. Sell 損失累積ポイント特定

### 7.1 Regime 別 Sell PnL

| Regime | 件数 | 平均 PnL (bps) | Loss/Win |
|--------|------|-----------------|----------|
| ranging | 568 | **-0.223** | 313/255 (55% loss) |
| trending | 118 | **-0.660** | 65/53 (55% loss) |
| trending_down | 36 | **-1.506** | 21/15 (58% loss) |
| trending_up | 26 | **-2.324** | 13/13 (50% loss) |
| unknown | 174 | **-0.693** | 88/86 (51% loss) |

### 7.2 最大損失源

1. **trending_up sell: -2.324 bps** (26件) — trending_sell_skip + sell_dynamic_kill が max_consecutive 安全弁で漏れたケース
2. **trending_down sell: -1.506 bps** (36件) — `skip_sell_trending_up_only=true` で意図的に通過させているが大損
3. **unknown sell: -0.693 bps** (174件) — skip_sell_unknown_regime がルールベースで一部ブロックしているが漏れ多数
4. **ranging sell: -0.223 bps × 568 件** — 件数最多で総損失寄与大

### 7.3 2/23 の異常

2/23 は sell fill rate が **2%** (12/535) まで崩壊。balance_forced_skip が集中した日。sell 側ガードの複合的発動により sell がほぼ完全停止し、大量の dead cycle が発生。

---

## 8. 結論と構造的課題

### 8.1 Guard Paradox — **確認**

VG 発動売と非発動売の PnL 差 (-0.608 vs -0.443) は、guard が中間リスクケースをフィルタし、残された約定済み sell が worst-case に偏ることを示している。**guard は正しくリスクを検知しているが、guard を通過した fill の品質が悪化する** という paradox が実在する。

### 8.2 Sell-Side Over-filtering

| ガード | Sell ブロック数 | 構造的効果 |
|--------|----------------|-----------|
| trending_sell_skip | 396 | sell 機会の 15.1% を削減 |
| balance_forced_skip | 377 | sell 残高不足の症状 (原因ではない) |
| skip_gate (sell) | 274 | ML ベースフィルタ |
| sell_dynamic_kill | 167 | rolling PnL ベース停止 |
| **合計** | **1214** | **sell 試行 2628 中の 46.2%** |

sell 側は試行の **46.2%** がガードにブロックされている。これは buy 側の **18.1%** (312/1727) と比較して **2.5 倍**。

### 8.3 正のフィードバックループ

```
sell ガード過剰 → sell 約定減少 → buy 偏重在庫 → sell 残高不足
  → balance_forced_skip 増加 → さらに sell 機会損失
  → Inventory Skewing が sell offset 縮小を試みるが offset_floor でクランプ
  → 効果限定 → ループ継続
```

### 8.4 推奨アクション

1. **sell_offset_floor の動的化**: 現在 0.20 固定だが、InvSkew が sell 促進を要求している場面では floor を一時的に緩和 (例: 0.10) して sell 約定率を向上させる
2. **trending_down sell の skip 解除確認**: 現在 trending_up_only=true で通過させているが、trending_down sell が -1.506 bps → trending_down sell にも regime-specific offset boost が必要
3. **balance_forced_skip の regime 記録修正**: 現在 regime=None で記録されているため分析が困難。skip record 生成時に `_current_regime_value()` を使用すること
4. **skip_gate counterfactual 追加**: skip 時の `actual_pnl_30s` を記録して precision 直接測定を可能にする
5. **2/23 型崩壊の予防**: 複合ガード同時発動による sell 完全停止を検知するサーキットブレーカーの導入
