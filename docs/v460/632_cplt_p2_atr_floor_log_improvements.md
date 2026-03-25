# 632# P2 ATR Floor Calibration & Log/Helper Improvements

## 概要
631# で BPS floor 10× ミスを修正後、ATR floor (`σ×mid×mult`) が新たな支配的フィルターとなり `spread_too_narrow` でトレードをブロックしていた。
Roll proxy 使用時の循環性を分析し、mult 引き下げ + cap 追加で緩和。
併せてログ改善、ztb ヘルパー統合、YAML drift テスト整備を実施。

## 変更詳細

### P2a: ATR Floor Calibration (Critical)

**問題分析**: Roll proxy (`σ = spread/(2×mid)`) 使用時、ATR floor の公式は:
```
ATR_floor = σ × mid × mult
         = (spread/(2×mid) × vol_ratio) × mid × mult
         = spread × vol_ratio × mult/2
```
- `mult=2.0` の場合: `ATR_floor = spread × vol_ratio`
- vol_ratio ≥ 1.0 (trending 時) → **ATR_floor ≥ spread → 全トレードブロック**
- ライブログ: `atr=3928, spread=3185, vol_ratio=1.233` → 9 連続 infeasible

**修正**:
| パラメータ | 変更前 | 変更後 | 根拠 |
|---|---|---|---|
| `min_spread_atr_mult` | 2.0 | 1.2 | ATR≈spread×vr×0.6。vol_ratio>1.67 でブロック開始 |
| `min_spread_atr_cap_bps` | (なし) | 3.0 | 上限 3.0bps ≈ 3,405 JPY @mid=11.35M |

**ファイル**:
- `configs/v460/fill_test.yaml`: mult/cap 値変更
- `scripts/v460/lib/fill_config.py`: `min_spread_atr_cap_bps` フィールド追加
- `scripts/v460/lib/fill_config_parser.py`: flat_keys に `min_spread_atr_cap_bps` 追加
- `scripts/v460/lib/maker_price.py`: `_enforce_spread_guards()` に cap ロジック追加、ログに σ 値追加

### P2b: Clamp Ceiling Analysis (結論: 変更不要)

630# P1 後、clamp 飽和率は 627# の 99-100% → **46-66%** に改善。
`final_clamp_hard_skip` は期間あたり 1-3 件のみ。ceiling 調整は不要。

### Log Improvements

1. **Cycle result にレジーム追加** (`fill_cycle_executor.py`)
   - 各サイクルの結果ログに `regime=xxx` タグを追加
   - 例: `Cycle 16660 result: filled=False, ..., regime=trending_down`

2. **Progress ログに σ/vol_ratio スナップショット** (`orchestrator_post_cycle.py`)
   - `σ=0.000173, vr=1.233` を Progress 行に追加
   - ATR floor 動向の定期的な可視化

3. **spread_too_narrow ログに σ 値** (`maker_price.py`)
   - `atr=3928, σ=0.000173` で ATR floor の根拠を即座に確認可能

### ztb Helper Integration

1. **`RobustStats.clip_outliers_mad`** → `monitor_fill_test.py`
   - 補足統計 (PnL mean/std) に MAD-clip ロバスト版を並記
   - スパイク約定による統計歪みを抑制

2. **調査結果** (変更なし):
   - `@timed()`: sync-only で async cycle には非適用。既存タイミングで充足
   - `MetricsAccumulator`: `gate_judgment.py` / `daily_health_check.py` は既に ztb を十分活用
   - `reproduce_152_metrics.py`: 移行可能だが低優先度 (既動作・refactoring メリット限定)

### YAML Drift Prevention テスト

- `min_spread_atr_mult`, `min_spread_atr_cap_bps`, `regime_trend_threshold_pct`, `mcb_enabled`, `sad_enabled` を KNOWN_YAML_OVERRIDES に追加
- `hard_skip_utc_hours` を KNOWN から除去 (YAML=[] = code default で一致)

## テスト

- `test_239_feasible_quote.py`: 26 passed (新規 TestATRFloorCap: 4 tests)
  - `test_atr_cap_limits_floor`: cap 有効時に ATR floor が抑制されること
  - `test_atr_no_cap_blocks`: cap=0 (無制限) で高 σ 時にブロック
  - `test_atr_cap_config_field_exists`: フィールド存在確認
  - `test_sigma_in_error_message`: エラーメッセージに σ 含有
- `test_336_yaml_code_drift_prevention.py`: 4 passed
- Full suite: 2237 passed, 127 skipped

## 影響範囲

- ライブ fill_test: hot_swap 後に ATR floor 緩和が有効化
- 既存 fill_records: 影響なし (新パラメータは将来のサイクルにのみ適用)
- monitor_fill_test: ロバスト統計行が追加表示
