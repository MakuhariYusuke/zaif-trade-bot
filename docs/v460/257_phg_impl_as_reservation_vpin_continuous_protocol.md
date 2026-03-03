# 257# AS Reservation Price + VPIN Continuous + RegimeDetectorLike Protocol

> **種別**: impl (実装完了報告)  
> **フェーズ**: phg (フェーズ横断: 市場理論・型安全)  
> **基盤**: 257# Codebase Sweep レポート (MT-1, MT-3, F-2)  
> **コミット**: `740a2381b`  
> **テスト**: 3575 (3575 passed)

---

## 概要

257# Codebase Sweep で特定した P1 項目のうち、市場理論強化 2 件と型安全化 1 件を実装。

| ID | カテゴリ | 内容 | ファイル |
|---|---|---|---|
| MT-1 | 市場理論 | AS Reservation Price: 在庫×ボラ連動 offset | maker_price.py |
| MT-3 | 市場理論 | VPIN Continuous: バイナリ→二次関数ランプ | maker_price.py |
| F-2 | 型安全 | RegimeDetectorLike Protocol | regime_detector.py + 3 files |

---

## §1 MT-1: Avellaneda-Stoikov Reservation Price

### 1.1 理論背景

Avellaneda-Stoikov (2008) の reservation price モデル:

$$r = s - q \cdot \gamma \cdot \sigma^2 \cdot \tau$$

| 変数 | 意味 | 推定方法 |
|---|---|---|
| $s$ | mid price | best_bid/ask の中間値 |
| $q$ | 在庫偏重 [-1, 1] | `_decayed_imbalance()` (228# time-decay 付) |
| $\gamma$ | リスク回避度 | `as_reservation_gamma` (config) |
| $\sigma^2$ | micro-volatility | $(spread / 2 \cdot mid)^2$ — Roll (1984) の spread-based estimator |
| $\tau$ | 時間ホライゾン | `as_reservation_tau_sec` (config) |

### 1.2 既存 inv_skew (162#) との関係

| 比較軸 | inv_skew (162#) | AS Reservation (257#) |
|---|---|---|
| 在庫感度 | 線形: `q × max_factor` | 非線形: `q × γ × σ² × τ` |
| ボラ依存 | なし | σ² に比例 |
| 時間依存 | time-decay (228# C2) | τ パラメータ + time-decay |
| 効果 | 一定の在庫リバランス | 高ボラ時の加速リバランス |
| 位置 | パイプライン上流 | inv_skew 直後 (補完) |

### 1.3 実装

- 新メソッド: `_apply_as_reservation_shift(side, spread, mid_price, effective_offset_ratio)`
- パイプライン位置: inv_skew → sell_guard → **AS reservation** → regime_boosts
- offset shift = `q × γ × σ² × τ × sign` (buy: +1, sell: -1)
- クランプ: `[min_offset_ratio, max_offset_ratio]`

### 1.4 Config

```yaml
as_reservation_enabled: false      # デフォルト無効 (フィーチャーフラグ)
as_reservation_gamma: 0.1          # リスク回避度
as_reservation_tau_sec: 120.0      # 時間ホライゾン (秒)
```

---

## §2 MT-3: VPIN Continuous Modulator

### 2.1 課題

従来の Volatility Guard VPIN トリガーはバイナリ閾値判定:
- VPIN < 0.70 → boost なし (1.0x)
- VPIN ≥ 0.70 → 全力 boost (2.0x)

この不連続性により:
- 0.69 → 0.70 の微小変化で offset が 2 倍にジャンプ
- 閾値付近での不安定な振動 (chattering)

### 2.2 二次関数ランプ

$$\text{boost} = 1 + (\text{max\_boost} - 1) \cdot \text{norm}^2$$

$$\text{norm} = \min\left(\frac{\text{vpin} - \text{min}}{\text{threshold} - \text{min}}, 1.0\right)$$

| VPIN | norm | boost (max=2.0) |
|---|---|---|
| 0.40 (min) | 0.00 | 1.00 |
| 0.50 | 0.33 | 1.11 |
| 0.55 | 0.50 | 1.25 |
| 0.60 | 0.67 | 1.44 |
| 0.65 | 0.83 | 1.69 |
| 0.70 (thresh) | 1.00 | 2.00 |

二次関数の選択理由:
- **onset が緩やか**: min 付近では感度が低い (ノイズ耐性)
- **threshold 付近で急峻**: 本当に危険な局面で強く反応
- 線形ランプ対比で「弱い信号に過剰反応しない」特性

### 2.3 velocity trigger との統合

velocity trigger と VPIN continuous の **max** を最終 boost に使用。
両方が発動する場合は強い方の信号を優先する。

### 2.4 Config

```yaml
vg_vpin_continuous_enabled: false   # デフォルト無効
vg_vpin_continuous_min: 0.40        # 連続スケーリング開始 VPIN
```

---

## §3 F-2: RegimeDetectorLike Protocol

### 3.1 Problem

`regime_detector: object | None` が 4 ファイルに散在:
- `maker_price.py` (コンストラクタ)
- `order_monitor.py` (`_resolve_regime_name`, `_should_block_reprice`, `monitor`)
- `adaptation_engine.py` (`try_auto_adapt`, `try_auto_lot_size`)

`_resolve_regime_name` は `hasattr` + `getattr` × 2 で duck-typing していた。

### 3.2 Solution

```python
@runtime_checkable
class RegimeDetectorLike(Protocol):
    @property
    def current_regime(self) -> FillTestRegime: ...
    @property
    def last_volatility_ratio(self) -> float: ...
```

- 定義場所: `regime_detector.py` (FillTestRegimeDetector と同居)
- `FillTestRegimeDetector` は自然に準拠 (実装変更不要)
- 4 ファイルの `object | None` → `RegimeDetectorLike | None` に統一

### 3.3 `_resolve_regime_name` 簡素化

Before (5 行):
```python
if regime_detector is None or not hasattr(regime_detector, "current_regime"):
    return None
current_regime = getattr(regime_detector, "current_regime", None)
if current_regime is None:
    return None
return getattr(current_regime, "value", None)
```

After (3 行):
```python
if regime_detector is None:
    return None
return regime_detector.current_regime.value
```

---

## §4 テスト

25 テスト追加 (`test_257_as_reservation_vpin_continuous_protocol.py`):

| クラス | テスト数 | 対象 |
|---|---|---|
| `TestRegimeDetectorProtocol` | 8 | Protocol runtime_checkable, mock 準拠, source 検証 |
| `TestASReservationShift` | 8 | disabled, neutral, long-buy/sell, clamp, gamma=0, spread 比例, pipeline |
| `TestVPINContinuousModulator` | 9 | binary unchanged/trigger, continuous min/partial/full/cap, quadratic shape, velocity priority |

既存テスト修正 1 件: `test_retrain_hot_reload.py` — MagicMock に `hot_reload_check_interval_sec = 120.0` 追加 (255# getattr 排除の副作用)

---

## §5 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/regime_detector.py` | `RegimeDetectorLike` Protocol 追加 |
| `scripts/v460/lib/maker_price.py` | `_apply_as_reservation_shift()` 追加, VG VPIN continuous 実装, Protocol 型適用 |
| `scripts/v460/lib/order_monitor.py` | `_resolve_regime_name` Protocol 化, 3 引数の型更新 |
| `scripts/v460/lib/adaptation_engine.py` | 2 引数の型を Protocol に更新 |
| `scripts/v460/lib/fill_config.py` | 5 config パラメータ追加 |
| `tests/.../test_257_*.py` | 25 テスト新規 |
| `tests/.../test_retrain_hot_reload.py` | MagicMock 修正 |

---

## §6 パイプライン構造 (更新後)

```
compute()
  ├─ base_offset_ratio (side 別)
  ├─ 162# Inventory Skewing (線形 q × factor)
  ├─ 088# Sell Offset Floor (173# 動的)
  ├─ 257# AS Reservation Shift (q × γ × σ² × τ)   ← NEW
  ├─ 163# Regime Boosts (trending/high_vol/ranging)
  ├─ 163# Spread Adaptive (wide spread → offset 縮小)
  ├─ 163# Volatility Guard (velocity + VPIN)       ← MT-3 continuous
  ├─ 163# Imbalance Risk (AS skip/boost)
  ├─ 226# Loss Boost (指数減衰)
  ├─ 100# FastFillDefense boost
  └─ Finalize (spread guard + price 組立)
```
