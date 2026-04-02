# 703# Task 2: sell/trending_up 損失ガード

## 背景
Protocol 688 再分析 (702#) で sell/trending_up が avg_pnl30=-2.01bps (95%CI: -1.21~-2.81, **統計有意**) と判明。
54 fills で -108.6bps の総損失貢献。既存の `sell_ranging_offset` パターンに倣い、sell/trending_up にも同様のガードを適用する。

## 修正箇所

### 1. `scripts/v460/lib/fill_config.py`
- `sell_trending_up_offset: float = 0.0` を FillTestConfig に追加
- 既存 `sell_ranging_offset` (L634付近) の直後に配置

### 2. `scripts/v460/lib/fill_config_parser.py`
- `sell_trending_up_offset` の YAML パース追加
- 既存 `sell_ranging_offset` のパースと同形式

### 3. `scripts/v460/lib/fill_config_validation.py`
- `sell_trending_up_offset` の range validation 追加 (0.0 ~ 2.0)

### 4. `scripts/v460/lib/skip_gate_evaluator.py`
- 既存の `sell_ranging_offset` 適用ロジック (L1000付近の `_calc_adjusted_threshold` またはsimilar) を検索
- 同一パターンで `sell_trending_up_offset` を適用:
  - `if side == "sell" and regime == "trending_up": threshold += self._config.sell_trending_up_offset`

### 5. `configs/v460/fill_test.yaml`
- `sell_trending_up_offset: 0.5` 追加 (sell_ranging_offset: 0.5 の直後)
- コメント: `# 702# P688: sell/trending_up avg_pnl=-2.01bps (n=54, p<0.01). sell_ranging_offsetと同等ペナルティ`

### 6. `configs/v460/fill_test.yaml` — regime_guard_overrides
- `enabled: false` → `enabled: true` に変更
- trending_up 設定を更新:
  - `ev_threshold_premium_bps: 0.3` (0.0→0.3)
  - `spread_as_guard_penalty_multiplier: 1.5` (1.0→1.5)

## テスト
- `tests/unit/v460/test_702_sell_trending_up_guard.py`
  - test_sell_trending_up_offset_applied: regime=trending_up, side=sell → threshold に offset 加算
  - test_sell_trending_up_offset_zero: offset=0.0 → 変化なし
  - test_sell_trending_up_not_applied_to_buy: side=buy → 無適用
  - test_sell_trending_up_not_applied_to_ranging: regime=ranging → sell_ranging_offset のみ
  - test_regime_guard_overrides_trending_up: EV premium 適用の数値精度
  - test_yaml_values_match_design: live YAML の値が計画通りか検証

## 制約
- 既存 `sell_ranging_offset` のテストが引き続きパスすること
- regime_guard_overrides 有効化が他 regime (ranging, trending_down) に影響しないこと
- hot-reload 対応: fill_config_parser 経由で動的更新可能であること
