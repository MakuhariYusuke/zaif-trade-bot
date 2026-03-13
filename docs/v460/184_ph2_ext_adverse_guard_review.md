# 184# 逆選択防御施策レビュー依頼 — 外部 AI レビュー

> **作成日**: 2026-02-28  
> **対象**: v460 "Microstructure Edge" BTC/JPY maker-only 自動取引  
> **目的**: 183# で実装した 5 つの逆選択防御施策について、外部 AI の視点で妥当性・見落とし・追加提案を検証する  
> **Git HEAD**: `babc3310e` (183# ログ分析ベース逆選択防御強化)  
> **fill_test 稼働コード**: `dc956168d` (180#) — 182#/183# は未デプロイ（次回再起動で自動適用）  
> **テスト**: 2330 passed, 0 failed  
> **前回レビュー**: 178# (GPT-5.3-Codex / Gemini 3.1 Pro, CycleStrategy 方針)

---

## 目次

1. [レビュー依頼の背景](#1-レビュー依頼の背景)
2. [システム概要](#2-システム概要)
3. [データ分析結果 — 逆選択の構造](#3-データ分析結果--逆選択の構造)
4. [実装した 5 施策の詳細](#4-実装した-5-施策の詳細)
5. [現在の設定全体像](#5-現在の設定全体像)
6. [期待効果と懸念](#6-期待効果と懸念)
7. [179#–182# の未検証変更 (同時デプロイ)](#7-179182-の未検証変更-同時デプロイ)
8. [レビューアへの質問事項](#8-レビューアへの質問事項)

---

## 1. レビュー依頼の背景

### 1.1 問題の核心

15 日間 (2/13–2/27) の fill_test ライブデータを統計分析した結果:

- **1,991 fills のうち 28.2% (561件) が逆選択** (Adverse Selection = 約定後 30s で mid price が不利方向に移動)
- 逆選択 fill は平均 **-5.90 bps** で、累計 **-3,310 bps** の損失
- 非逆選択 fill は平均 **+1.90 bps** (WR 64.4%、累計 +2,715 bps)
- **→ 逆選択さえ除去すればシステムは黒字**

これが v460 収益性改善の最大ボトルネックであることが定量的に確定したため、183# で 5 つの防御施策を実装した。

### 1.2 レビュー依頼の趣旨

1. 5 施策の設計根拠は統計的に妥当か (確認バイアス・過学習リスク)
2. 閾値設定に問題はないか (厳しすぎ / 甘すぎ)
3. 見落としている逆選択要因や、試すべき代替アプローチはあるか
4. 179#–182# との相互作用でリスクはないか
5. fill rate 低下とのトレードオフは適切か

---

## 2. システム概要

### 2.1 取引環境

| 項目 | 値 |
|------|-----|
| 取引所 | Coincheck (日本) |
| 通貨ペア | BTC/JPY |
| BTC 価格帯 | ≈ ¥14,700,000 |
| ロットサイズ | 0.001 BTC (≈ ¥14,700) |
| 注文方式 | Maker limit order (指値注文) |
| 手数料 | Maker: **0%** (無料) |
| 1 bps 換算 | ≈ ¥1.47 |

### 2.2 アーキテクチャ

```
run_fill_test.py (436行, エントリーポイント)
├── FillTestRunner (3 Mixin 合成)
│   ├── FillLoopOrchestratorMixin (1,306行) ── ループ制御・skip chain
│   ├── FillCycleExecutorMixin (704行) ──── 1サイクル: OB→SkipGate→発注→監視→PnL
│   └── FillRecordHelpersMixin ─────────── FillRecord構築
│
├── MakerPriceCalculator (762行) ── offset算出チェーン
│   ├── spread adaptive offset
│   ├── regime trending boost (方向×サイド別)
│   ├── sell guard / fast_fill defense
│   └── volatility guard boost
│
├── SkipGateEvaluator (761行) ── ML + ルール層のスキップ判定
│   ├── LGBM PnL120 回帰モデル (per-side)
│   ├── 適応的閾値調整 (target skip rate)
│   ├── velocity skip (buy/sell)
│   ├── hour_offsets + narrow_spread_offset  ← 183# NEW
│   └── regime threshold offset
│
├── FillTestConfig (1,117行) ── 80+ YAML外部化パラメータ
├── RegimePolicyConfig / CycleStrategy (322行) ── regime別policy (179#)
└── 38 lib modules (adaptation, balance, lot, etc.)
```

### 2.3 注文ライフサイクル

```
1. side 選択 (SideSelector: ラウンドロビン + balance_forced)
2. skip chain (time_filter → regime gating → skip_gate ML → velocity/hour/spread rules)
3. offset 算出 (MakerPriceCalculator: spread × ratio → regime/VG/FFD boost)
4. maker limit 注文 → ポーリング (5s 間隔, 300s timeout)
5. stale 検出 → reprice (30s 後 5bps 乖離で再発注)
6. 約定 → PnL 計測 (30s/60s/120s + ev_weighted)
7. FillRecord 構築 → JSONL 永続化
```

### 2.4 逆選択 (AS) の定義

```python
adverse_selected = (
    (side == "buy"  and mid_30s_after < fill_price) or
    (side == "sell" and mid_30s_after > fill_price)
)
```

約定後 30 秒の mid price が注文者にとって不利方向に動いた場合に `True`。  
maker-only で手数料 0% のため、AS が直接の損益決定要因。

---

## 3. データ分析結果 — 逆選択の構造

### 3.1 概況

| 指標 | 値 |
|------|-----|
| 分析期間 | 2026-02-13 〜 2026-02-27 (15日間) |
| 総レコード | 4,671 |
| Filled | 1,991 (fill rate 42.6%) |
| 30s PnL 平均 | **-0.30 bps** (負 = 損失) |
| WR | 46.3% |
| 逆選択率 | **28.2%** (561/1,991) |

### 3.2 逆選択 vs 非逆選択

| グループ | n | Mean PnL (30s) | WR | 累計 PnL |
|----------|---|----------------|-----|----------|
| Adverse | 561 | **-5.90 bps** | 0.0% | -3,310 bps |
| Non-adverse | 1,430 | **+1.90 bps** | 64.4% | +2,715 bps |
| 全体 | 1,991 | -0.30 bps | 46.3% | -595 bps |

**核心**: 非逆選択 fill は WR 64.4% / +1.90 bps で十分なエッジがある。  
問題は 28.2% の AS fill が **-5.90 bps の大損失** を生んでいる非対称性。

### 3.3 AS の最強予測因子: VG velocity

| Feature | Adverse (med) | Non-adverse (med) | 差分 |
|---------|--------------|-------------------|------|
| VG velocity (bps) | **-0.95** | **+0.83** | 1.78 |
| Spread (JPY) | 狭い傾向 | やや広い | - |

VG velocity (60s window の価格変化率) は AS の最も鮮明な判別因子。  
AS 発生時は価格が急変した直後 (velocity < 0 = 直前に下落) であり、  
流動性テイカーの方向追従フロー（momentum ignition）のシグナル。

### 3.4 時間帯別 AS 率 (JST / UTC)

| JST | UTC | AS% | Fill数 | Sum PnL | 所見 |
|-----|-----|-----|--------|---------|------|
| 01h | **16** | **64%** | 11 | -79.8 | 深夜、流動性枯渇+海外勢 |
| 03h | **18** | **50%** | 10 | -54.8 | 同上 |
| 06h | 21 | 36% | 114 | **-125.8** | PnL合計最悪 |
| 17h | 08 | 42% | 92 | -63.1 | 東京開場後の変動 |
| 23h | **14** | **43%** | 95 | **-134.5** | 欧州開場前 |

高 AS 時間帯は「流動性の薄い深夜帯」と「セッション切替前後」に集中。

### 3.5 スプレッド別 AS 率

| Spread bucket | AS% | Fill数 |
|--------------|-----|--------|
| < 2,000 JPY | **32%** | ~475 |
| 2,000–3,000 | 25% | ~600 |
| 3,000–4,000 | 28% | ~450 |
| > 4,000 | 28% | ~466 |

狭スプレッドは AS 率が **4–7pt 高い**。  
スプレッドが狭い = 板が充実 = テイカーの方向性フローが maker を直撃しやすい。

### 3.6 レジーム別 AS 率

| Regime | AS% | Fill数 |
|--------|-----|--------|
| trending_up | **34.3%** | - |
| ranging | 27.1% | - |
| high_vol | 26.8% | - |

trending_up が最悪。方向性のある相場で maker limit の逆選択リスクが高い（理論通り）。

---

## 4. 実装した 5 施策の詳細

### 施策 1: 時間帯別 skip_gate 閾値オフセット

**根拠**: §3.4 の AS 率悪化時間帯  
**実装**: 既存の `skip_gate_hour_offsets` 機構 (158# P1-6) を YAML 設定

```yaml
skip_gate:
  hour_offsets:
    14: 0.3    # 23h JST — AS 43%, PnL合計最悪
    16: 0.5    # 01h JST — AS 64%, 最厳格
    18: 0.3    # 03h JST — AS 50%
    21: 0.3    # 06h JST — PnL合計最悪
    23: 0.2    # 08h JST — AS 42%
```

**メカニズム**: PnL 回帰モードのため、offset 正=閾値引き上げ=より高い予測PnlでないとFill許可しない。  
**設計意図**: fill を完全禁止せず、ML gate の判断基準を厳格化。

### 施策 2: Buy velocity skip 有効化 + 閾値保守化

**根拠**: §3.3 の VG velocity が AS の最強因子  
**変更前**: sell のみ velocity skip 有効 (閾値 8.0 bps)  
**変更後**: buy/sell 両方有効、閾値 6.0 bps

```yaml
skip_gate:
  sell_velocity_skip_enabled: true
  sell_velocity_skip_threshold_bps: 6.0    # 8.0 → 6.0
  buy_velocity_skip_enabled: true           # false → true
  buy_velocity_skip_threshold_bps: -6.0    # -8.0 → -6.0
```

**メカニズム**: 価格が急変 (|velocity| > 6 bps/60s) した方向への順張り注文を pre-ML でブロック。  
急落後の buy (= falling knife catch) と急騰後の sell を防御。

### 施策 3: 狭スプレッド逆選択ガード (コード変更あり)

**根拠**: §3.5 — spread < 2000 JPY で AS率 32% (全体 28.2% より 4pt 高い)  
**新規コード**: `skip_gate_evaluator.py` L680-695

```python
# 183# narrow spread adverse guard
_spread_offset = 0.0
_ns_thr = self._config.skip_gate_narrow_spread_threshold_jpy
if _ns_thr > 0 and spread_at_order is not None and spread_at_order < _ns_thr:
    _spread_offset = self._config.skip_gate_narrow_spread_offset
    if _spread_offset != 0.0:
        logger.debug(
            "[skip_gate] 183# narrow spread guard: spread=%.0f < %.0f → offset +%.2f",
            spread_at_order, _ns_thr, _spread_offset,
        )

_total_offset = _hour_offset + _spread_offset  # 加算
```

```yaml
skip_gate:
  skip_gate_narrow_spread_threshold_jpy: 2000.0
  skip_gate_narrow_spread_offset: 0.2
```

**新規 Config フィールド**: `fill_config.py` に 2 フィールド追加、`config_hot_reload.py` に登録済み。  
**メカニズム**: hour_offset と加算され、`threshold_offset` として ML gate に渡される。

### 施策 4: Volatility Guard 感度引上げ

**根拠**: §3.3 — VG velocity が AS の最も鮮明な判別因子  
**変更**:

```yaml
volatility_guard:
  velocity_threshold_bps: 12.0   # 15.0 → 12.0 (20%引下げ)
  vpin_threshold: 0.60           # 0.63 → 0.60 (5%引下げ)
```

**メカニズム**: VG トリガー発動 → `offset_boost_factor` (2.0倍) が offset に乗る → maker が板の奥に退避。  
これにより「感度引上げ → 発動頻度増加 → 退避回数増加 → AS 回避」の因果連鎖。

### 施策 5: Narrow spread boost 強化

**根拠**: §3.5 — 狭スプレッドでの AS 率高い  
**変更**:

```yaml
spread_adaptive:
  narrow_spread_boost_buy: 2.0    # 1.5 → 2.0
  narrow_spread_boost_sell: 2.5   # 2.0 → 2.5
```

**メカニズム**: spread < `narrow_spread_bps` (2.5) のとき offset に boost 倍率が掛かる。  
boost 値増加 → offset 拡大 → maker がスプレッド中心からより遠い位置に = fill されにくいがAS回避。

---

## 5. 現在の設定全体像

### 5.1 skip_gate 完全設定

```yaml
skip_gate:
  enabled: true
  buy_enabled: true
  sell_enabled: true
  mode: pnl                           # PnL回帰モード
  model_path: models/v460/skip_gate_lgbm_pnl120.pkl
  model_path_buy: models/v460/skip_gate_lgbm_pnl30_buy.pkl
  model_path_sell: models/v460/skip_gate_lgbm_pnl120_sell.pkl
  as_threshold: 0.50
  as_threshold_buy: 0.50
  as_threshold_sell: 0.50
  pnl_threshold: 0.0
  max_skip_rate: 0.3
  use_ob_features: true
  adaptive_threshold: true
  target_skip_rate_buy: 0.15
  target_skip_rate_sell: 0.250
  adaptive_window: 50
  adaptive_min_samples: 20
  adaptive_step: 0.05
  adaptive_floor: 0.35
  adaptive_ceiling: 0.80
  regime_thresholds:
    high_vol: 0.2
    ranging: 0.1
    trending: -0.1
    trending_up: -0.1
    trending_down: -0.1
  skip_sell_unknown_regime: true
  unknown_buy_offset_boost: 2.0
  sell_velocity_skip_enabled: true
  sell_velocity_skip_threshold_bps: 6.0         # ← 183#
  buy_velocity_skip_enabled: true                # ← 183#
  buy_velocity_skip_threshold_bps: -6.0          # ← 183#
  hour_offsets:                                  # ← 183#
    14: 0.3
    16: 0.5
    18: 0.3
    21: 0.3
    23: 0.2
  skip_gate_narrow_spread_threshold_jpy: 2000.0  # ← 183# NEW
  skip_gate_narrow_spread_offset: 0.2            # ← 183# NEW
```

### 5.2 volatility_guard 完全設定

```yaml
volatility_guard:
  enabled: true
  velocity_window_sec: 60
  velocity_threshold_bps: 12.0     # ← 183# (15→12)
  vpin_threshold: 0.60             # ← 183# (0.63→0.60)
  offset_boost_factor: 2.0
  inv_skew_damping_enabled: true
```

### 5.3 spread_adaptive 完全設定

```yaml
spread_adaptive:
  enabled: true
  narrow_spread_bps: 2.5
  narrow_spread_boost: 2.0
  narrow_spread_boost_buy: 2.0     # ← 183# (1.5→2.0)
  narrow_spread_boost_sell: 2.5    # ← 183# (2.0→2.5)
  wide_spread_bps: 4.5
  wide_spread_ratio: 0.5
```

---

## 6. 期待効果と懸念

### 6.1 反実仮想推計

| 施策 | 対象Fill数 (15d) | 推定PnL改善 | 根拠 |
|------|-----------------|-------------|------|
| 時間帯フィルタ | ~300 | +50–100 bps | 5時間帯でAS率40–64%→fill抑制 |
| Velocity skip | ~30 | +20–40 bps | 急変直後の損失fill阻止 |
| 狭スプレッドguard | ~475 | +30–60 bps | AS率32%→閾値厳格化 |
| VG感度引上げ | 全fill | +20–50 bps | offset退避発動頻度増 |
| Narrow boost | ~475 | +10–30 bps | offset拡大→AS回避 |
| **合計** | | **+130–280 bps/15d** | |

→ 15d 累計 -595 bps が **-315 ~ -465 bps に改善** 見込み (22–55% 損失削減)

### 6.2 懸念事項

| # | 懸念 | 深刻度 | 対策 |
|---|------|--------|------|
| C-1 | **fill rate 低下**: 5施策すべてが「厳格化」方向 → fill rate 42.6% がさらに低下 | HIGH | hot-reload で YAML パラメータ調整可能。fill rate 30% 割れで一部緩和 |
| C-2 | **過学習リスク**: 15日間のデータに特化した閾値 → 市場環境変化で無効化 | MED | 時間帯パターンは市場構造 (流動性周期) に依拠、比較的頑健 |
| C-3 | **施策間の相互作用**: hour_offset + narrow_spread_offset + regime_threshold が加算 → 過度な skip | MED | 最悪ケース: 0.5 + 0.2 + 0.2 = 0.9 offset → fill 困難だが安全側 |
| C-4 | **velocity skip の副作用**: 急変後のリバウンド (= 逆に利益機会) を逃す可能性 | LOW | 6 bps 閾値は十分保守的、エッジ確認まで抑制が合理的 |
| C-5 | **VG 感度過剰**: velocity 12 bps で頻発 → 常時 boost 状態 → fill rate 大幅低下 | MED | ライブ稼働後にログ監視、VG 発動率 50% 超で 14 bps に戻す |

---

## 7. 179#–182# の未検証変更 (同時デプロイ)

183# と同時にデプロイされる 179#–182# の変更:

| # | 内容 | リスク |
|---|------|--------|
| 179# | RegimePolicyConfig + CycleStrategy + Chase | regime policy 初稼働。dynamic_cycle/wait/chase は新機能 |
| 180# | Watchdog 非表示化 + from_yaml 堅牢化 | 低リスク。fill_test は既に 180# で稼働中 |
| 181# | C/D/Chase 有効化 + EV_weighted + StopConditionMonitor | Chase 有効化は trending 時の re-price (最大5回) が新規稼働 |
| 182# | Trend Mode 厳格化 + EV_weighted外部化 + Deadlock regime別緩和 | gated_regime (confidence < 0.55 で trending→ranging 降格) が新規 |

**最大リスク**: 179#/181# の Chase 機構 + 183# の velocity skip が相互作用する可能性。  
Chase は trending 時に drift > 3bps で re-price するが、re-price 直後に velocity > 6bps で skip される可能性がある。  
→ ただし Chase は既存注文の re-price であり、velocity skip は新規注文にのみ適用されるため、直接衝突しない。

---

## 8. レビューアへの質問事項

### Q1: 5施策の方向性は妥当か

5 施策すべてが「厳格化」方向 (= fill 抑制) であり、収益機会の喪失リスクがある。  
**代替として「情報量増加系」(= より良い判断材料でフィルタリング精度を上げる) の施策を優先すべきか？**

例:
- SkipGate モデルの再学習 (VG velocity をより重要な特徴量として強調)
- order flow toxicity 指標 (VPIN 以外) の導入
- 板の上位 level の厚みを使った流動性シグナル

### Q2: VG velocity の閾値 12 bps は適切か

AS fill の VG velocity 中央値は -0.95 bps、非 AS は +0.83 bps。  
VG の velocity_threshold_bps = 12 はこれらよりはるかに大きい。  

**閾値を 12 bps ではなく、たとえば 3–5 bps まで引き下げるべきか？**  
あるいは VG threshold は「offset boost の発動条件」であり「skip 条件」ではないため、  
用途が異なるのでこの程度が妥当か？

### Q3: hour_offset の値は過剰か / 不足か

最大 offset 0.5 (01h JST、11 fills/15d) は統計的に信頼性が低い (n=11)。  
**サンプル数が少ない時間帯の offset を下げるべきか？**  
あるいは「小サンプルだがリスクが大きいので保守的に厳しくする」が正しいか？

### Q4: narrow spread guard は spread_adaptive boost と機能重複していないか

- 施策 3: skip_gate に narrow_spread_offset を加算 (ML 判断を厳格化)
- 施策 5: spread_adaptive.narrow_spread_boost を引上げ (offset を物理的に拡大)

**同一の問題 (狭スプレッド AS) に対して 2 つの防御が重複しているが、相乗効果は期待できるか？**  
それとも施策 3 が機能すれば施策 5 は不要か？

### Q5: 逆選択の「根本原因」についての所見

現在の分析は「どの fill が AS か」の統計分類にとどまっている。  
**逆選択の根本メカニズム** (= なぜ maker が不利な側に Fill されるのか) について、以下の仮説のうちどれが最も蓋然性が高いか？

1. **情報非対称性**: テイカーが持つ方向情報に maker が無防備
2. **Latency disadvantage**: Coincheck API のレイテンシにより、価格変動後の cancel が間に合わない
3. **Lot size disadvantage**: 0.001 BTC のミニマムロットは板のノイズとして狙われやすい
4. **Spread regime**: 狭スプレッド = 板が厚い = 大口フローが maker を押し潰す
5. **その他の構造要因**: レビューアの知見に基づく分析

### Q6: 183# 以降の次アクション提案

5 施策の効果検証 (ライブ 24–48h 後) を前提として、  
**次に着手すべき改善方向** として以下のどれが最も収益インパクトが大きいか？

- A: SkipGate モデル再学習 (VG velocity weight 強化)
- B: Regime gating の改善 (trending_up で sell を完全解放)
- C: Offset 算出チェーンの根本見直し (AS 予測に基づく動的 offset)
- D: Fill rate 改善 (skip が多すぎる場合)
- E: その他アプローチ

---

## 付録 A: FillRecord フィールド一覧 (62項目)

| カテゴリ | フィールド |
|---------|----------|
| **基本** | `cycle_id`, `timestamp`, `side`, `run_id`, `git_sha` |
| **注文** | `order_price`, `order_quantity`, `fill_price`, `filled`, `cancelled`, `cancel_reason`, `error_message` |
| **約定品質** | `queue_wait_sec`, `spread_at_order`, `spread_offset_ratio`, `effective_offset_used` |
| **PnL** | `mid_at_fill`, `mid_30s_after`, `mid_60s_after`, `mid_120s_after`, `post_fill_30s_pnl`, `post_fill_60s_pnl`, `post_fill_120s_pnl`, `actual_measurement_sec`, `early_exit_triggered`, `pnl_at_exit_bps`, `ev_weighted_pnl` |
| **AS** | `adverse_selected`, `adverse_selected_raw` |
| **Regime** | `regime`, `regime_confidence`, `regime_stability`, `regime_trend_pct`, `regime_volatility_ratio` |
| **OB** | `orderbook_imbalance`, `bid_depth_total`, `ask_depth_total`, `mid_price_trend_5s`, `spread_bps` |
| **SkipGate** | `skip_gate_skipped`, `skip_gate_score`, `skip_gate_reason`, `skip_gate_model_used`, `skip_gate_as_prob`, `skip_gate_threshold_used`, `skip_gate_hour_offset` |
| **VG/FFD** | `vg_triggered`, `vg_velocity_bps`, `vg_vpin`, `vg_boost_factor`, `price_velocity_60s`, `ffd_boost_active` |
| **Balance** | `balance_forced_switch`, `balance_forced_consecutive` |
| **Lot** | `confidence_lot_factor`, `order_lot_regime`, `order_lot_effective`, `confidence_lot_mode` |
| **Reprice** | `reprice_count`, `reprice_drift_bps` |

---

## 付録 B: コミット履歴 (直近 10)

| Commit | # | 内容 |
|--------|---|------|
| `babc3310e` | **183#** | ログ分析ベース逆選択防御強化 |
| `3a1f9e380` | 182# | Trend Mode 厳格化 + EV_weighted外部化 + Deadlock regime別緩和 |
| `bea85f26e` | 181# | C/D/Chase 有効化 + EV_weighted + StopConditionMonitor |
| `dc956168d` | 180# | Watchdog 非表示化 + 179# Self-Review + from_yaml堅牢化 |
| `089be3650` | 179# | RegimePolicyConfig + CycleStrategy + _effective_sleep + Chase |
| `dbfbf1bab` | 178# | 177レビュー評価 (docs) |
| `a8df03e93` | 177# | Gemini 3.1 Pro second opinion (docs) |
| `3f6dace13` | 176# | Trending方向×サイド別Offset Asymmetry |
| `7715a52ed` | 176# | (同上、code commit) |
| `5df74e733` | 175# | Code Review Sweep #2 |

---

## 付録 C: ディレクトリ構成 (関連部分のみ)

```
configs/v460/
  fill_test.yaml          # 全パラメータ (YAML 外部化)

scripts/v460/
  run_fill_test.py        # エントリーポイント (436行)
  lib/
    fill_config.py        # FillTestConfig (1,117行)
    skip_gate_evaluator.py # SkipGate (761行)
    fill_loop_orchestrator.py  # ループ制御 (1,306行)
    fill_cycle_executor.py     # 1サイクル (704行)
    maker_price.py        # offset算出 (762行)
    regime_policy.py      # CycleStrategy (322行)
    config_hot_reload.py  # hot-reload (407行)
    + 30 other modules

models/v460/
  skip_gate_lgbm_pnl120.pkl
  skip_gate_lgbm_pnl30_buy.pkl
  skip_gate_lgbm_pnl120_sell.pkl

results/v460/fill_test/
  fill_records_20260213.jsonl ~ fill_records_20260227.jsonl  # 15日分
  logs/fill_test.log  # 47,414行

tests/unit/v460/
  test_183_log_analysis_improvements.py  # 16 tests
  + 60+ other test files (2330 total tests)
```
