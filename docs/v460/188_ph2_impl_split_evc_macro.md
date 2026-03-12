# 188# ファイル分割 + Phase C ev_weighted SkipGate + Phase D Macro Regime 基盤

> 日付: 2026-02-28
> 前提: 186# Phase A (ヒステリシス+clamp) + 187# Phase B (Chase方向+guard_trace) 完了
> 目的: コード長大化対策 + 186# 計画の Phase C/D 実装

---

## 1. ファイル分割

### 1.1 regime_policy.py → cycle_strategy.py

| 対象 | before | after |
|------|--------|-------|
| `regime_policy.py` | 373 行 | 192 行 (MAX 250) |
| `cycle_strategy.py` | — | 139 行 (MAX 200, 新規) |

- `DefaultCycleStrategy` クラスを `scripts/v460/lib/cycle_strategy.py` に抽出
- `regime_policy.py` に re-export を残し、10 箇所の既存 import を維持
- `CycleStrategy` Protocol + `RegimePolicyConfig` は `regime_policy.py` に残留

### 1.2 fill_cycle_executor.py — _build_fill_record() 抽出

- `run_single_cycle` 内の FillRecord 構築ロジック (~100 行) を `_build_fill_record()` に抽出
- keyword-only 引数 (~30 パラメータ) で明示的渡し → 可読性向上
- ファイル全体: 718 行 (MAX 750)

---

## 2. Phase C: ev_weighted SkipGate (186# C-1)

### 2.1 概要

Buy 側の SkipGate は pnl30 (30 秒後 PnL) モデルで判定していたが、
Sell 側 pnl120 (120 秒後 PnL) との期待値加重を導入。

```
ev = w30 × pnl30 + w120 × pnl120
```

- **Buy**: primary=pnl30 (短期), alt=pnl120 (長期)
- **Sell**: primary=pnl120 (長期), alt=pnl30 (短期)
- **AS mode**: 確率空間の加重平均が不適切なため除外

### 2.2 設定 (fill_test.yaml)

```yaml
skip_gate:
  ev_weighted_enabled: false  # 有効化は pnl120_buy / pnl30_sell モデル訓練後
  ev_w30: 0.4
  ev_w120: 0.6
  model_path_buy_long: null   # pnl120 buy モデルパス
  model_path_sell_short: null  # pnl30 sell モデルパス
```

### 2.3 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `skip_gate_evaluator.py` | `_ALT_MODEL_SLOTS`, `_load_alt_models()`, `_try_ev_weighted_decision()` |
| `fill_config.py` | 5 フィールド追加 + YAML パース |
| `config_hot_reload.py` | ev_weighted 3 キー hot-reload 対象 |

### 2.4 次ステップ

1. pnl120 buy モデルの訓練 (training pipeline から)
2. pnl30 sell モデルの訓練
3. `ev_weighted_enabled: true` で有効化
4. w30/w120 の最適化 (バックテスト)

---

## 3. Phase D: Macro Regime 基盤 (186# D)

### 3.1 概要

5 分 / 15 分スロープベースの中長期マーケット状態判定。
既存の micro regime (FillTestRegimeDetector) と組み合わせて、
トレンド方向の確度向上・矛盾検出を行う。

### 3.2 MacroRegimeDetector

```
timestamp, mid_price → 時間バケット集約 → OLS 線形回帰スロープ → MacroTrend 分類
```

- **バケット**: 30 秒間隔で mid_price を平均化
- **スロープ**: OLS 線形回帰 (bps/min)
- **分類**: slope_threshold=1.0 bps/min, strong=3.0 bps/min

| MacroTrend | 条件 |
|-----------|------|
| STRONG_UP | 5m & 15m slope > threshold |
| WEAK_UP | 片方のみ上昇 |
| NEUTRAL | 閾値内 |
| WEAK_DOWN | 片方のみ下降 |
| STRONG_DOWN | 5m & 15m slope < -threshold |
| INSUFFICIENT | データ不足 |

### 3.3 compose_regimes()

```python
effective_regime, is_aligned = compose_regimes(micro_regime, micro_confidence, macro_result)
```

- `is_aligned=True`: micro/macro 方向一致
- `is_aligned=False`: 矛盾 (例: micro=trending_up, macro=strong_down)
- 呼び出し元で矛盾時に regime を ranging に降格する等の制御に使用 (次フェーズ)

### 3.4 次ステップ

1. `fill_cycle_executor` に MacroRegimeDetector 統合
2. regime 矛盾時のサイクル間隔調整
3. Macro Trend に応じた offset 増減

---

## 4. テスト

- 回帰テスト: 160 テスト全パス (179#〜187#)
- 新規テスト: 24 テスト (`test_188_split_evc_macro.py`)
  - 分割後方互換 (6)
  - ev_weighted 判定 (8)
  - MacroRegimeDetector (7)
  - config hot-reload (1)
  - YAML パース (1)
  - bucket overflow / invalid price (1)
