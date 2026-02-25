# 165# AS Root Cause Analysis + Velocity Skip Rule + Daily Health Report

> **Session**: 165# (2026-02-25)
> **Prior**: 164# SHAP analysis, 163# stopgap catalog, 162# 10-day analysis
> **Commits**: 545fa23c9

---

## 1. AS (Adverse Selection) Root Cause Analysis

### 1.1 Summary Statistics

| Metric | Value |
|--------|-------|
| Total records | 3,389 |
| Filled | 1,500 (44.2%) |
| AS rate (overall) | 26.6% (399/1500) |
| AS avg_pnl30 | -5.41 bps |
| Non-AS avg_pnl30 | +1.64 bps |

### 1.2 SkipGate Model Predictive Power

**致命的発見: SkipGate Score-PnL 相関がほぼゼロ**

| Scope | Pearson r |
|-------|-----------|
| Overall | -0.0185 |
| Buy | -0.0018 |
| Sell | -0.0319 |

- SkipGate False Negative Rate: **100%** (全399件の AS fill が SkipGate を通過)
- Score bin 分析: 全 bin で AS rate 16.7%-23.7% (差別化能力なし)
- 結論: **adaptive threshold は事実上ランダム選択**

### 1.3 Regime 別 AS 率

| Regime | Fills | AS Rate | Avg PnL30 |
|--------|-------|---------|-----------|
| unknown | 360 | 40.0% | -4.77 bps |
| ranging | 926 | 20.1% | -0.20 bps |
| trending | 76 | 28.0% | -2.78 bps |
| trending_down | 94 | 21.3% | -2.07 bps |
| trending_up | 45 | 44.4% | -5.41 bps |

- unknown regime: 既に skip_sell/buy_unknown_regime: true で対策済み (最新データでは出現 0)
- trending_up: 最高 AS 率だがサンプル少

### 1.4 Model Used 別 AS 率

| model_used | Fills | AS Rate | Avg PnL30 |
|------------|-------|---------|-----------|
| none | 462 | 36.4% | -3.46 bps |
| primary | 949 | 20.8% | +0.39 bps |
| primary:side_sell | 139 | 31.7% | -0.96 bps |

- **primary:side_sell** (31.7% AS, -0.96bps) が主要改善ターゲット
- SHAP #1 特徴量 price_velocity_60s (buy=0.832, sell=1.420) をルールベースに活用

---

## 2. AS-R1: Velocity-based Sell Skip Rule

### 2.1 設計根拠

- SkipGate ML モデル (r \u2248 0) をバイパスするルールベース pre-gate
- SHAP 最重要特徴量 price_velocity_60s を直接閾値判定に利用
- 保守的初期閾値 8.0 bps (キャリブレーション用速度ログを追加)

### 2.2 実装

| File | Change |
|------|--------|
| ill_config.py | +4 fields: sell/buy_velocity_skip_enabled/threshold_bps |
| skip_gate_evaluator.py | 速度ルール (SkipGate ML 前のプリゲート) |
| ill_quality.py | price_velocity_60s field (FillRecord) |
| ill_cycle_executor.py | 速度値の抽出記録 |
| cancel_reasons.py | SKIP_GATE_RULE_VELOCITY_SELL/BUY |
| ill_test.yaml | 	arget_skip_rate_sell: 0.20 \u2192 0.25, 速度ルール設定 |

### 2.3 Config

`yaml
sell_velocity_skip_enabled: true
sell_velocity_skip_threshold_bps: 8.0  # bps, conservative initial
buy_velocity_skip_enabled: false       # buy AS risk is lower
buy_velocity_skip_threshold_bps: -8.0
target_skip_rate_sell: 0.25            # was 0.20
`

### 2.4 Flow

`
124# unknown regime sell skip
  \u2193
165# velocity rule (pre-gate)  \u2190 NEW
  \u2193
ML model evaluate
  \u2193
Decision (skip/accept)
`

---

## 3. 162# P1: Daily Per-Regime 3-Indicator + Stopgap Exit Evaluation

### 3.1 受入基準: \u300c3\u6307\u6a19 + per-regime \u3092\u65e5\u6b21\u3067\u51fa\u529b\u300d \u2714

| Module | Purpose |
|--------|---------|
| stopgap_health.py | Daily regime \u00d7 side 3-indicator + stopgap exit evaluation |
| stopgap_daily_report.py | CLI: --window 168 --json --output |

### 3.2 Daily 3-Indicator Output

`
  --- Daily 3-Indicator (all) ---
       Day     N   Fill%    PnL30      P10    AS%
  20260219   176  70.4%    -0.55    -6.36 29.5%
  20260220   132  60.8%    -0.20    -5.53 20.4%
  20260221   164  43.5%    -0.60    -2.99 14.0%
  ...
`

Per-regime \u00d7 sell breakdown also available per day.

### 3.3 Stopgap Exit Evaluation (163# Table Integration)

| ID | Stopgap | Current Verdict | Key Metric |
|----|---------|-----------------|------------|
| 2-A | trending_sell_skip | KEEP | AS_rate=26.5% (<35% OK), but total PnL=-164.89bps |
| 2-C | sell_dynamic_kill | KEEP | 15.29 kills/day (threshold: <1/day) |
| 2-D | sell_guard | KEEP | cancel_rate=72.4% (threshold: <10%) |
| 6-A | unknown_regime_skip | KEEP | unknown_rate=35.4% (threshold: <5%) |

> \u2728 \u5168 stopgap \u304c KEEP \u5224\u5b9a \u2192 \u73fe\u6642\u70b9\u3067\u306f\u3069\u306e stopgap \u3082 OFF \u3067\u304d\u306a\u3044\u3002\u30e2\u30c7\u30eb\u6539\u5584\u304c\u5148\u6c7a\u300d

### 3.4 未実装 (今後追加予定)

- 3-A/B/C: forced_skip / deadlock / rescue (IS \u5b9f\u88c5\u5f8c)
- 1-A/1-C: time_filter regime-adaptive (Step 2 \u5b9f\u88c5\u5f8c)
- 2-B: max_consecutive_trending_sell_skip (2-A \u9000\u51fa\u5f8c)

---

## 4. Test Results

| Test Suite | Count | Status |
|------------|-------|--------|
| test_velocity_skip_rule.py | 25 | \u2705 All passed |
| test_stopgap_health.py | 32 | \u2705 All passed |
| v460 regression | 1935 | \u2705 All passed |

---

## 5. Remaining Work

| Priority | Task | Status |
|----------|------|--------|
| P0 | Fill test restart (AS-R1 \u9069\u7528) | \u2b50 Next |
| P1 | 163# doc update (AS findings + velocity rule) | Next |
| P1 | Velocity threshold calibration (from logged data) | After restart |
| P2 | SkipGate model rebuild (feature engineering) | Future |
| P2 | SO-2/SO-3 sell offset optimization | After SO-1 calibration |

---

## 6. Update History

| Date | Change |
|------|--------|
| 2026-02-25 | 165# Initial: AS root cause + AS-R1 velocity skip + 162# P1 daily health |
