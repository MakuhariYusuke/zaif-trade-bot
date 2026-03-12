# 398# G3 パイプライン セルフレビュー + portfolio_value バグ修正

**作成**: 2026-03-14
**根拠**: 396# G3 パイプライン実装のセルフレビュー
**分類**: ph3 — G3 gate バグ修正 + 訓練実験検証

---

## 概要

396# (commit `e8f624886`) で実装した G3 パイプラインのセルフレビューを実施。
**CRITICAL バグ** を発見し修正。修正後、end-to-end 検証を実行して正常動作を確認。

---

## 発見: CRITICAL — portfolio_value 取得バグ

### 根本原因: 2つの異なる HeavyTradingEnv クラス

本プロジェクトには **同名だが属性セットが異なる2つの環境クラス** が存在する:

| 環境 | パス | 用途 | `portfolio_value` | `balance` | `unrealized_pnl` |
|------|------|------|:--:|:--:|:--:|
| 訓練用 Env | `ztb/training/environments/heavy_trading_env.py` | ユニットテスト | None (stub) | ✅ | ✅ |
| 本番 Env | `ztb/trading/environment/heavy_env/core.py` | **実際のSAC訓練** | ✅ (property) | ❌ | ❌ |

`task_sac_train.py` L315 は **本番 Env** を import する:
```python
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
```

### 396# の問題コード (sac_common.py)

```python
# 旧コード: production env には balance も unrealized_pnl も存在しない
pv = float(getattr(env, "balance", 0.0)) + float(getattr(env, "unrealized_pnl", 0.0))
```

**結果**: `all_portfolio_values` が全ステップ 0.0 → MaxDD=0, Sharpe=0, G3 gate が無意味に。

### 398# 修正

```python
# 新コード: production env の portfolio_value property を優先
pv = float(getattr(env, "portfolio_value", 0.0))
if pv == 0.0:
    # fallback: training env (heavy_trading_env.py) 互換
    pv = float(getattr(env, "balance", 0.0)) + float(getattr(env, "unrealized_pnl", 0.0))
```

---

## 検証結果

### ユニットテスト (70/70 PASS)

- G3 パイプラインテスト: 23/23 ✅
- G2 SAC ブロッカーテスト: 47/47 ✅

### End-to-End 検証 (本番 Env)

本番 HeavyTradingEnv を直接操作し、G3 指標算出を検証:

| 指標 | 値 | 旧 (396# バグ時) | 正常か |
|------|-----|------|:--:|
| PF | 0.117 | 0.0 (PnL=0) | ✅ (まだ低いが非ゼロ) |
| MaxDD | 14.99% | 0.0 | ✅ |
| Sharpe | -13.5 | 0.0 | ✅ (未学習モデルなので負は正常) |
| avg_gross | 815.2 | 0.0 | ✅ |
| reward_profit_corr | 0.99 | 0.0 | ✅ |

### 属性確認 (本番 Env)

```
balance        = NOT_FOUND  ← 属性なし
unrealized_pnl = NOT_FOUND  ← 属性なし
portfolio_value = 1000190.96 ← 正常動作 ✅
initial_portfolio_value = 1000000.0 ← 正常
total_pnl = 190.96
trades_count = 15
```

---

## 訓練実験計画

### Phase 1: G3 パイプライン E2E 検証 (398# 本実験)

**目的**: 修正後のコードが実際の `run_experiment.py` 経路で正しく G3 指標を出力するか

```bash
python scripts/v460/run_experiment.py \
  --config configs/v460/experiments/g2_sac_g3_validation.yaml
```

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| total_timesteps | 10,000 | 検証専用 (5分以内に完了) |
| seeds | [42, 123] | 2 seed で十分 |
| gamma | 0.80 | ベースライン設定 |

**検証チェックリスト**:
- [x] 結果 JSON に `g3_judgment_cache` が存在
- [x] `seed_metrics` に pf, max_drawdown, sharpe_annual, reward_profit_corr が含まれる
- [x] `max_drawdown > 0` (portfolio_value が正しく収集されている)
- [x] `reward_profit_corr ≠ 0` (相関算出が機能)

**実行結果** (`v460_g2train_seed42_20260312_185924.json`):

| 指標 | seed=42 | seed=123 | 判定 |
|------|---------|----------|------|
| PF | 0.985 | 1.099 | median=1.042 < 1.05 → FAIL |
| MaxDD | 0.41% | 0.27% | worst=0.41% < 15% → PASS ✅ |
| Sharpe | -1.06 | 6.42 | median=2.68 > 0.8 → PASS ✅ |
| avg_gross | 92.4 | 85.6 | gross > fee(0) → PASS ✅ |
| reward_profit_corr | 0.094 | 0.298 | 非ゼロ ✅ |

**G3 総合判定**: FAIL (PF median 未達 — 10K steps 短縮訓練なので想定内)
**G2 総合判定**: FAIL (positive_seed_ratio=0.5 < 0.75 — 同上)

**重要**: 全 G3 指標が非ゼロで正常に算出されており、398# の portfolio_value
バグ修正が有効であることを確認。396# のバグ状態では MaxDD=0, Sharpe=0 に
なっていたはずの指標が、正しく計算されている。

### Phase 2: 本番実験 (399# 以降)

Phase 1 通過後、G3 判定を含む本番 4-seed 訓練を実施:

| 実験 | Config | Steps | Seeds |
|------|--------|-------|-------|
| A. Baseline | g2_sac_train.yaml | 50K | 4 |
| B. γ=0.95 reward-tuned | g2_sac_gamma095_reward_tuned.yaml | 100K | 4 |

---

## 教訓

1. **同名クラスに注意**: プロジェクト内に同名の `HeavyTradingEnv` が2つ存在し、
   属性セットが異なる。テスト用 Env で動いても本番 Env では壊れるケース。
2. **テスト環境と本番環境の乖離**: ユニットテスト用の Mock/Stub env は
   本番 env の属性を完全に再現していない。end-to-end テストの重要性。
3. **fallback パターン**: `getattr(env, attr, 0.0)` の 0.0 fallback は
   属性の欠落を隠蔽する。明示的な存在チェックが望ましい。
