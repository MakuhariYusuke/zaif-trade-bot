# 196# velocity offset 比例化 + trending_sell ソフト化

## 概要

193# (ev_weighted) / 195# (velocity_skip, B1') のハードゲート→ソフトオフセット変換パターンを更に展開し、**2 つの改善を実装**した。

### A. velocity_offset 段階的 boost (195# 残課題)

195# で導入した固定 ×2.0 の velocity_offset_boost_factor を、**閾値超過量に比例した段階的 boost** に変換。

| velocity (bps) | threshold | 固定モード (旧) | 比例モード (新) |
|---|---|---|---|
| 6.0 (閾値ちょうど) | 6.0 | ×2.0 | ×2.0 |
| 9.0 (50% 超過) | 6.0 | ×2.0 | ×2.5 |
| 12.0 (100% 超過) | 6.0 | ×2.0 | ×3.0 |
| 18.0 (200% 超過) | 6.0 | ×2.0 | ×4.0 (上限) |

**根拠**: 閾値をわずかに超えた velocity (6.1bps) と大幅に超えた velocity (18bps) で同じ ×2.0 は不合理。超過量に応じた段階的調整で、市場急変度に比例した保守性を実現。

**計算式**:
```
excess_ratio = abs(velocity) / abs(threshold)  # >= 1.0
boost = 1.0 + (base_factor - 1.0) × excess_ratio
boost = min(boost, velocity_offset_max_mult)
```

### B. trending_sell → soft offset (193# 残課題)

trending regime での sell ハードスキップを、**offset boost による保守的発注に変換**。

| 状況 | 旧モード (hard skip) | 新モード (soft offset) |
|---|---|---|
| trending_up + sell | SKIP (取引不可) | PASS + offset ×3.0 |
| trending_up + sell + HF4 | bypass → PASS (通常価格) | PASS + offset ×3.0 |
| trending_up + sell + inv_bypass | bypass → PASS (通常価格) | PASS + offset ×3.0 |

**効果**:
- **複雑性除去**: HF4 (連続スキップ安全弁), inv_bypass, buy_insufficient bypass の 3 条件が不要に
- **機会損失回避**: sell が常に発注される (保守的価格で)
- 2/25 反実仮想分析: 118 件の sell 強制スキップ → 実際は sell も +1.51bps だった

**設計**: 193# ev_offset / 195# vel_offset と同じパターン
```
CycleGateAggregator._check_trending_sell()
  → trending_sell_as_offset_enabled=True
    → blocked=False + offset_mult=3.0
  → CycleGateResult.trending_offset_mult に伝播
  → orchestrator → run_single_cycle(trending_offset_mult=3.0)
  → fill_cycle_executor: sell order_price += delta (mid から離れる)
```

## 変更ファイル

### 1. `scripts/v460/lib/fill_config.py`
- 新 config フィールド:
  - `velocity_offset_proportional: bool = False`
  - `velocity_offset_max_mult: float = 4.0`
  - `trending_sell_as_offset_enabled: bool = False`
  - `trending_sell_offset_boost_factor: float = 3.0`
- YAML マッピング追加 (skip_gate + loss_control セクション)

### 2. `scripts/v460/lib/skip_gate_evaluator.py`
- velocity offset 計算に比例モード分岐追加
- `velocity_offset_proportional=True` 時: excess_ratio ベースの段階的 boost
- ログに `(proportional)` ラベル追加

### 3. `scripts/v460/lib/cycle_gate_aggregator.py`
- `GateCheckResult.offset_mult: float | None = None` フィールド追加
- `CycleGateResult.trending_offset_mult: float | None = None` フィールド追加
- `_check_trending_sell()`: soft mode 時に `offset_mult` を返す
- `evaluate()`: Gate 3 の `offset_mult` を `trending_offset_mult` に伝播

### 4. `scripts/v460/lib/fill_loop_orchestrator.py`
- `run_single_cycle()` 呼び出しに `trending_offset_mult` パラメータ追加

### 5. `scripts/v460/lib/fill_cycle_executor.py`
- `run_single_cycle()` に `trending_offset_mult` パラメータ追加
- 195# vel_offset ブロックの直後に trending offset ブロック追加
- sell 価格に delta 加算 (mid から離れる → 保守的)

### 6. `configs/v460/fill_test.yaml`
- `velocity_offset_proportional: true`
- `velocity_offset_max_mult: 4.0`
- `skip_sell_trending: true` (false → true に戻す)
- `trending_sell_as_offset_enabled: true`
- `trending_sell_offset_boost_factor: 3.0`

### 7. `tests/unit/v460/test_196_velocity_proportional_trending_soft.py` (新規)
- 34 テスト:
  - `TestVelocityProportionalConfig`: デフォルト値検証
  - `TestVelocityProportionalCalculation`: 比例 boost 計算 (5 パラメトリック + cap + 固定)
  - `TestVelocityProportionalInSkipGate`: source inspection
  - `TestTrendingSellSoftConfig`: デフォルト値検証
  - `TestTrendingSellSoftGate`: soft/hard mode 切替、trending_down 除外、buy 不影響、balance_forced、bypass 不要確認、audit trail
  - `TestTrendingOffsetInExecutor`: パラメータ存在、source inspection
  - `TestGateCheckResultOffsetMult`: 新フィールド検証
  - `TestCycleGateResultTrendingOffset`: 新フィールド検証
  - `TestConfigYamlParse196`: YAML parse 検証
  - `TestBackwardCompatibility196`: デフォルト値で旧モード維持
  - `TestDesignConsistency196`: 設計パターン一貫性

### 8. `tests/unit/v460/test_176_trending_offset_asymmetry.py`
- `test_live_yaml_skip_sell_trending_false` → 196# soft mode に更新d

### 9. `tests/unit/v460/test_113_resilience.py`
- `run_single_cycle` 行数上限: 550 → 570 (196# trend_offset ブロック追加分)

## ドキュメント整理 (同梱)

- `193_ev_weighted_to_offset.md` → `193_ph2_impl_ev_weighted_to_offset.md` (命名正規化)
- `195_velocity_b1_soft_gate.md` → `195_ph2_impl_velocity_b1_soft_gate.md` (命名正規化)
- `194_ph2_impl_cycle_gate_aggregator.md` 新規作成 (欠損ドキュメント補完)
- `index.md` に 193#, 194#, 195#, 196# エントリ追加

## テスト結果

```
2414 passed, 0 failed (196# 変更分)
  - 新規 34 テスト (test_196)
  - 既存テスト 全通過
  - 除外: test_retrain_hot_reload, test_141, test_v460_core (pre-existing lightgbm/sklearn failures)
```

## 今後の課題

1. `velocity_offset_boost_factor` の基底値最適化: 2.0 が最適か、バックテストで検証
2. `trending_sell_offset_boost_factor` の最適化: 3.0 → PnL データから最適値探索
3. **balance_forced 問題**: JPY 枯渇→forced sell→損失パターンの根本対策 (977 回/runtime 発生)
4. **narrow_spread_pause の CycleGateAggregator 統合**: executor の B3 判定をここに移管
5. **maker_price ValueError の Gate 化**: D1-D3 例外を事前チェックで回避
