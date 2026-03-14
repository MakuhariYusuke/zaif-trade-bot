# 415# fill_test ログ分析 (3/11〜3/14)

## 1. 概況

| 日付 | Total | Real | Filled | SDK | 30s PnL (bps) | Sell Final | コード |
|------|-------|------|--------|-----|---------------|------------|--------|
| 3/11 | 85 | 45 | 28 | 7 | **+0.48** | 0.30 | pre-405# |
| 3/12 | 553 | 306 | 219 | 10 | **+0.31** | 0.30 | pre-405# |
| 3/13 | 625 | 232 | 178 | 266 | **-0.79** | 0.46 | mixed |
| 3/14 | 243 | 75 | 61 | 147 | **-0.77** | 0.50 | 405#+414# |

- **Real**: offset_stages が記録された本サイクル
- **SDK**: `sell_dynamic_kill` による早期キャンセル（全て `order_price=0`）
- **Sell Final**: sell 側 offset pipeline の final ステージ平均

---

## 2. 因果関係マップ

### 2.1 sell_dynamic_kill の爆発 (7→266→147)

```
残高 2mBTC 割れ
  └→ sell 注文価格が 0 に設定される
      └→ sell_dynamic_kill で即時キャンセル (全件 order_price=0)
          └→ SDK 266件/日 (3/13), 147件/日 (3/14)
              └→ 有効 sell サイクル減少: 306 real → 232 → 75
```

**判定**: 残高制限が主因。SDK 対象は全て `order_price=0` であり、rolling PnL kill ロジック
（`SellDynamicKillManager`）とは無関係。残高制限下ではこのカウントはノイズ。

### 2.2 405# Offset Fix のパラドックス

```
405# _effective_max_ratio 導入
  └→ sell 側中間キャップ: 0.30 → 0.50 に拡大
      └→ sell final offset: 0.30 → 0.50 (100% が 0.30 超)
          ├→ fill 位置が mid から遠くなる
          │   └→ fill 成立 = 大きな価格移動が前提
          │       └→ 逆選択 (AS) に当たりやすい
          │           └→ sell AS 率: 22%→38%→37%
          └→ PnL 反転: sell PnL +1.17→-0.82→-1.34 bps
```

**核心**: 405# は「デッドロック解消」として正しいが、**副作用として売側 AS 率を倍増させた**。
deadlock 時代は sell final=0.30 が事実上の「保護壁」として機能していた。

#### AS vs 非AS の PnL 格差

| 日付 | Side | AS PnL | 非AS PnL | AS 率 |
|------|------|--------|----------|-------|
| 3/12 | buy | -5.93 | +1.23 | 25% |
| 3/12 | sell | -6.32 | **+3.26** | 22% |
| 3/13 | buy | -6.58 | +1.65 | 29% |
| 3/13 | sell | **-9.69** | **+4.66** | **38%** |
| 3/14 | buy | -6.42 | +0.44 | 10% |
| 3/14 | sell | -6.53 | +1.66 | **37%** |

**非AS fill は全日・全 side で正**。AS fill のみが全損益を破壊している。

### 2.3 Skip Gate の機能不全

```
skip_gate による早期キャンセル:
  3/12: 112件 (全 cancel の 45%)
  3/13:  42件 (全 cancel の 11%)
  3/14:   4件 (全 cancel の  2%)
```

Online Monitor の最新評価（retrain_scheduler.log 18:00:28）:
```
buy:  skip_rate=7.7%  → pass_pnl=+0.097bps  (微黒)
sell: skip_rate=2.1%  → pass_pnl=-1.157bps   (深赤)
```

**判定**: sell 側の skip_gate がほぼ機能停止（98% pass）。
SDK で sell サイクルの大半がリアルに到達する前にキャンセルされるため、skip_gate の
学習データが buy に偏り、sell 側の判別能力が退化した可能性がある。

### 2.4 VG (Volatility Guard) の飽和

```
VG triggered rate:
  3/12: 217/219 (99%)
  3/13: 175/178 (98%)
  3/14:  61/61  (100%)
```

VG は全 fill で発火 → **判別力ゼロ**。常時 boost を適用する「定数乗数」と化している。
VPIN 平均: 0.69〜0.71（高い一方で変動幅小）。

### 2.5 EV スコアの予測力

| 日付 | EV>0 PnL | EV>0 win | EV≤0 PnL | EV≤0 win |
|------|----------|----------|----------|----------|
| 3/12 | +0.55 | 55% | +0.002 | 47% |
| 3/14 | +0.68 | 56% | **-1.38** | 40% |

3/12 ではEV≤0 でもほぼ breakeven だったが、3/14 では大幅マイナス。
405# による offset 拡大で EV≤0 fill の損失が拡大した。

### 2.6 時間帯パターン

全日通じて悪い時間帯:
- **JST 09h**: 3/12=-3.10, 3/13=-8.48, 3/14=-1.95 — 東京寄り付き
- **JST 23h**: 3/12=-7.00, 3/13=-3.04 — NY 昼前
- **JST 15h**: 3/14=-5.51 — 東京午後

### 2.7 3/12 TypeErrorクラッシュ (2回)

```
bayesian_regime_filter.py L513:
  self._regime_history[-200:]
  TypeError: sequence index must be integer, not 'slice'
```

`_regime_history` が deque でスライス非対応。git_sha `92c588e535de` で2回発生。
後続コミットで修正済みの可能性があるが、要確認。

---

## 3. 改善提案

### P0: AS 防御の強化（最大インパクト）

**根拠**: 非AS fill は常に正。AS fill 1件の損失 (≈-7bps) が非AS 3件の利益 (≈+2bps) を打ち消す。

1. **sell 側 offset ceiling の再検討**: 405# は中間キャップのみ緩和したが、
   `offset_ceiling_ratio_sell=0.50` 自体が高すぎる可能性。
   → sell final ceiling を 0.35〜0.40 に下げて AS 率への影響を A/B テストすべき
2. **AS 事前予測モデル構築** (404# Action 3): VPIN + spread_bps + macro_slope の
   組み合わせで fill 前に AS 確率を推定し、高 AS 確率時は offset を追加拡大 or skip
3. **EV≤0 fill の抑制**: EV score ≤ 0 での fill が損失の主因。
   skip_gate とは別に `ev_score_pretrade < -1.0` で sell をブロックする簡易ゲートを検討

### P1: Skip Gate 再学習

**根拠**: sell skip_rate=2.1% は異常。SDK 汚染でデータバランスが崩壊している。

1. SDK レコード（order_price=0）を skip_gate 学習データから除外する前処理フィルタ追加
2. side 別に独立した skip_gate モデルの検討（現状は unified model）
3. skip_gate の閾値を side 別に分離（現行 threshold_used=0.1 は buy/sell 共通）

### P1: VG 閾値の再調整

**根拠**: 100% 発火は保護として無意味。

1. VPIN 閾値の引き上げ（現行の閾値 vs 実分布を確認し、上位 30% のみ発火するよう調整）
2. VG boost factor の grade 化（VPIN 0.5〜0.7 は×1.2、0.7〜0.8 は×1.5 等）

### P2: 残高制限モード

**根拠**: 現在は sell が order_price=0 で SDK カウントされるだけで、
buy 側の挙動は変わらない。

1. 残高 < 2mBTC 時の「restricted mode」フラグ導入
2. restricted mode では sell 停止 + buy のみ運用（現状事実上そうなっているが明示化）
3. fill_records に `restricted_mode: true` フラグを追加し分析精度向上
4. SDK カウントを restricted_mode 時は除外するメトリクス整備

### P2: git_sha 記録の一貫性

**根拠**: 3/13 レコードで 3種の SHA が混在（12-char / 40-char / 異なるコミット）。

1. fill_test 起動時に git_sha を確定し、ランタイム中は不変とする
2. 現行: 毎サイクルで `git rev-parse HEAD` を実行 → 途中コミットで SHA が変わる

### P3: JST 09h / 23h ガード

**根拠**: 3日連続でこの2時間帯が赤字。

1. 402# の分析で `hard_skip_utc_hours[21]` (=JST 06h) は凍結済み
2. ただし JST 09h (=UTC 00h) は hard_skip 対象外 — skip_gate で対応すべき
3. JST 23h (=UTC 14h) も同様 — パラメータチューニングでなくモデル側で対処すべき

---

## 4. 今回の最重要洞察

> **405# の sell offset deadlock 解消は技術的に正しかったが、
> 「制約が保護として機能していた」ケースの典型例**。
>
> sell final=0.30 の deadlock は、結果的に sell fill を mid 近傍に限定し、
> AS に遭遇する確率を抑制していた。
> 405# で ceiling=0.50 に開放したことで、sell fill は可能になったが
> AS 率が 22%→37% に上昇、PnL が +1.17→-1.34 bps に反転した。
>
> → 正しいアプローチは「deadlock 解消 + AS 防御の同時導入」。
> 現状は前者のみ実施され後者が欠落している。

---

## 5. fill_test 現在の状態

- **PID 37220**: 8a496ef32a0e (414#), 2026-03-14 16:59:46 起動, WS=7MB
- **retrain_scheduler**: PID 36688 (watchdog restart), 次回 retrain まで 7200s
- **online_monitor**: DEGRADED (pass_mean_pnl=-0.523bps < threshold=-0.3bps)
- **残高制限**: 継続中（sell は SDK でほぼ全キャンセル）
