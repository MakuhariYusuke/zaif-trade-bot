# 195# velocity_skip ソフト化 + B1' offset 統合

## 概要

193# (ev_weighted → offset) パターンを横展開し、**2 つのハードゲートを連続的 offset 修飾子に変換**した。

### 解決した問題

| 問題 | 元の動作 | 195# 対策 |
|---|---|---|
| velocity_skip で取引機会を完全喪失 | sell/buy velocity > threshold → hard SKIP | offset ×2.0 で保守的発注に変換 |
| B1' と low_vol_offset_boost の重複 | ranging+buy+low_vol → hard SKIP + offset ×1.4 | hard skip 廃止、offset ×1.4 のみ |
| 192# §3 "同じ概念が複数箇所" | velocity_skip: skip_gate + cycle_gate 両方に存在 | cycle_gate は soft 委譲、skip_gate が一元管理 |

## 設計

### A. velocity_skip → offset boost

```
旧: velocity > 6.0bps (sell) → SKIP (100%取引不可)
新: velocity > 6.0bps (sell) → PASS + offset ×2.0 (保守的: mid から2倍離れる)

旧: velocity < -6.0bps (buy) → SKIP (100%取引不可)
新: velocity < -6.0bps (buy) → PASS + offset ×2.0 (保守的: mid から2倍離れる)
```

**効果**: 急変局面でも取引チャンスを残しつつ、AS リスクを offset 拡大で抑制

| velocity | offset boost | 意味 |
|---|---|---|
| sell +10bps (急騰中) | ×2.0 | 売値を mid から 2 倍離す → fill 低確率だが安全 |
| buy -10bps (急落中) | ×2.0 | 買値を mid から 2 倍離す → fill 低確率だが安全 |
| 閾値以下 | ×1.0 | 通常動作 (変更なし) |

### B. B1' ranging_buy_low_vol → offset 委譲

```
旧: ranging + buy + vol_ratio < 0.75 → hard SKIP (完全遮断)
新: ranging + buy + vol_ratio < 0.75 → PASS (maker_price low_vol_offset_boost ×1.4 が対応)
```

**根拠**: maker_price に `low_vol_offset_boost_enabled: true` (×1.4) が既に存在。
hard skip なしでも offset 拡大で十分に AS リスク抑制可能。
さらに 193# ev_offset が併用されるため、negative EV 時は追加で保守化される。

### 193# ev_offset との累積適用

```
maker_price.compute()
  → order_price (base)
    ↓
skip_gate.evaluate()
  → ev_score + velocity_offset_mult
    ↓
[193# ev_offset] order_price × ev_mult (±5%程度)
    ↓
[195# vel_offset] order_price × vel_mult (×2.0)  ← NEW
    ↓
place order
```

最悪ケース (negative EV + adverse velocity):
- ev_offset: ×0.5 (下限)
- vel_offset: ×2.0
- 合計 offset: ×1.0 (相殺されてニュートラル)

## 変更ファイル

### 1. `scripts/v460/lib/fill_config.py`
- 新 config フィールド:
  - `velocity_skip_as_offset_enabled: bool = False`
  - `velocity_offset_boost_factor: float = 2.0`
  - `ranging_buy_low_vol_as_offset: bool = False`
- `SkipGateResult.velocity_offset_mult: Optional[float] = None`
- YAML マッピング追加 (skip_gate section, regime section)

### 2. `scripts/v460/lib/skip_gate_evaluator.py`
- velocity_skip 分岐をリファクタ: sell/buy 条件を統合
- `velocity_skip_as_offset_enabled=True` 時:
  - hard skip せず `result.velocity_offset_mult = boost_factor` を記録
  - `result.price_velocity_60s` に velocity 値を保存
  - ML 判定に進む (early return しない)
- `velocity_skip_as_offset_enabled=False` 時: 旧動作維持

### 3. `scripts/v460/lib/fill_cycle_executor.py`
- 193# ev_offset ブロックの直後に velocity offset ブロック追加
- `sg.velocity_offset_mult` が存在 & ≠1.0 の場合:
  - buy: `order_price -= delta` (保守的 = mid から離れる)
  - sell: `order_price += delta` (保守的 = mid から離れる)
- ログ: `[195# vel_offset] {side}: velocity=N.NNbps → offset_mult=N.NN`

### 4. `scripts/v460/lib/cycle_gate_aggregator.py`
- `_check_ranging_buy_low_vol()`: `ranging_buy_low_vol_as_offset=True` 時に `blocked=False`
- `_check_velocity_skip()`: `velocity_skip_as_offset_enabled=True` 時に `blocked=False`

### 5. `configs/v460/fill_test.yaml`
- `velocity_skip_as_offset_enabled: true`
- `velocity_offset_boost_factor: 2.0`
- `ranging_buy_low_vol_as_offset: true`

### 6. `tests/unit/v460/test_195_velocity_b1_soft.py` (新規)
- 32 テスト:
  - `TestVelocitySkipSoftMode`: config 値、SkipGateResult フィールド
  - `TestVelocityOffsetExecutor`: 方向性、delta 計算、EV との累積
  - `TestRangingBuyLowVolSoftMode`: hard/soft 切替、audit trail、bypass 条件
  - `TestVelocitySkipSoftGateAggregator`: cycle_gate 内の soft 動作
  - `TestConfigYamlParse`: YAML parse、default 値
  - `TestBackwardCompatibility`: 旧モード維持
  - `TestDesignConsistency`: source inspection

### 7. `tests/unit/v460/test_113_resilience.py`
- `run_single_cycle` 行数上限: 520 → 550 (195# vel_offset ブロック追加分)

## テスト結果

```
2561 passed, 0 failed (v460 unit tests)
  - 新規 32 テスト (+32)
  - 既存 2529 テスト 全通過
```

## 193# 残課題の対応状況

| 課題 | 対応 |
|---|---|
| 1. velocity_skip ソフト化 | ✅ 195# 実装済み |
| 2. B1' ranging_buy_low_vol 統合 | ✅ 195# 実装済み |
| 3. trending_sell_skip 簡素化 | ❌ 次セッション |
| 4. balance_forced 問題 | ❌ 構造変更が大きいため別途 |
| 5. sensitivity パラメータ最適化 | ❌ バックテスト必要 |

## 今後の課題

1. **velocity_offset_boost_factor の最適化**: 現在固定 2.0 → velocity に比例した段階的 boost 検討
2. **B1' 低 vol offset boost 倍率の調整**: 現在 1.4 → hard skip 廃止に伴い引き上げ検討
3. **trending_sell_skip 簡素化**: bypass 条件 (HF4, inv_bypass, safety_valve) が複雑
4. **balance_forced 問題**: JPY 枯渇→forced sell→損失パターンの根本対策
