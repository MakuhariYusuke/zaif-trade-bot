# 704# Task 1: sell_trending_down_offset + spread_as_guard 修正のユニットテスト

## 背景
704# で以下の変更を cplt が直接実装した。Codex はユニットテストのみを担当。

### 変更点
1. **spread_as_guard: `last_spread` → `last_spread_raw` 修正** (orchestrator_mid_cycle.py)
   - `last_spread` は 210# M5 の 60s staleness guard 付きで、60秒以上 compute() なしだと None を返す
   - entry gate の spread_as_guard は独立判定のため `last_spread_raw` (staleness guard なし) を使用すべき
   - 3日間ライブデータで spread_as_guard_triggered が 0% だった root cause

2. **sell_trending_down_offset 追加** (config + parser + validation + hot-reload + evaluator)
   - 3日間データ: sell+trending_down avg=-1.17bps, n=64, total=-74.84bps (全 sell×regime 最大損失)
   - sell_ranging_offset / sell_trending_up_offset と同形式

## テストファイル
`tests/unit/v460/test_704_sell_loss_defense.py`

## テスト要件

### A. sell_trending_down_offset テスト
以下を `test_703_sell_trending_up_guard.py` のパターンに準拠して作成:

1. **test_sell_trending_down_offset_applied**: sell + trending_down で offset が加算される
   - `skip_gate_sell_trending_down_offset=0.5` 設定
   - `_total_offset` に 0.5 が加算されることを検証
2. **test_sell_trending_down_offset_not_applied_buy**: buy + trending_down では加算されない
3. **test_sell_trending_down_offset_not_applied_other_regime**: sell + ranging では trending_down offset は加算されない (ranging は別途 sell_ranging_offset が適用)
4. **test_sell_trending_down_offset_coexistence**: sell_ranging_offset, sell_trending_up_offset, sell_trending_down_offset が同時設定でも相互干渉しない
5. **test_sell_trending_down_offset_validation_range**: fill_config_validation で [0, 2] 範囲チェックが機能する
6. **test_sell_trending_down_offset_hot_reload**: config_hot_reload の allowlist に含まれていることを検証

### B. spread_as_guard staleness 修正テスト
7. **test_spread_as_guard_uses_raw_spread**: orchestrator_mid_cycle の entry gate 評価で `last_spread_raw` が使用されることを検証
   - MakerPriceCalculator のモックで `last_spread` = None, `last_spread_raw` = 300.0 (3bps @ mid=10M) を設定
   - `_spread_bps` が None ではなく約 3.0 になることを検証
8. **test_spread_as_guard_triggered_with_raw**: spread_bps < 15bps (threshold) のとき spread_as_guard_triggered = True になる

### C. regime_guard_overrides trending_down テスト
9. **test_regime_guard_trending_down_penalty**: trending_down で ev_premium=0.3 + penalty_mult=1.5 が適用される
   - `apply_entry_gate_adjustments()` を直接テスト
   - spread_bps=3.0, base_ev=1.0 のとき adjusted_ev = 1.0 - (0.5×1.5) - 0.3 = -0.05

## 制約
- 既存テスト (`test_703_*`, `test_700_*`, `test_695_*`) が引き続きパスすること
- `test_703_sell_trending_up_guard.py` と `test_703_hour_param_retune.py` のパターン・mock 設計を踏襲
- fill_config fixture は `conftest.py` の共有 fixture があればそれを使用、なければ最小限の dataclass コンストラクタ
