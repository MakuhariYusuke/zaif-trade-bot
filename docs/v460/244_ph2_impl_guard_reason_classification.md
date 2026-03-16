# 244# Guard Reason Classification (232# P2-2)

## 概要
`guard_fire_counts` の全 reason を **市場都合 (MARKET)** / **システム都合 (SYSTEM)** /
**回復動作 (RECOVERY)** の 3 カテゴリに分類するモジュールを新設。
運用ログ・状態エクスポートにカテゴリ別集計を追加し、
「なぜ動けなかったか」の診断を簡素化。

## 背景 (232# P2-2)
> "guard reason を「市場都合」と「システム都合」で分類し直す"

## 分類ルール

### MARKET (市場都合): 外部環境に起因する防御
| Guard | 説明 |
|-------|------|
| `gate_unknown_regime_buy_skip` | regime 判別不能 |
| `gate_ranging_low_vol_skip` | ranging + 低ボラ |
| `gate_trending_sell_skip` | trending で sell 危険 |
| `gate_buy_dynamic_kill` | buy rolling PnL 悪化 |
| `gate_sell_dynamic_kill` | sell rolling PnL 悪化 |
| `gate_rule_velocity_*_skip` | 価格速度過大 |
| `gate_rule_skip_unknown_sell` | unknown regime sell |
| `gate_narrow_spread_pause` | spread 狭小 |
| `gate_spread_too_narrow` | spread 閾値未満 |
| `gate_sell_guard_reject` | sell guard 拒否 |
| `gate_toxicity_participation_skip` | 毒性過大 |
| `mcb_halt` / `mcb_warning` | Circuit Breaker |
| `mcb_sad_escalation` | SAD エスカレーション |
| `sad_frozen` / `sad_dry` / `sad_wide` | 市場異常 |
| `toxic_veto_block` | toxic veto |
| `toxicity_participation_skip` | 毒性参加率 skip |
| `quiescence` | 静止期間 |

### SYSTEM (システム都合): 内部制約に起因する停止
| Guard | 説明 |
|-------|------|
| `dd_halt` | Drawdown 上限 |
| `per_side_dd_both_halt` | 両側 DD halt |
| `per_side_halt_switch` | 片側 halt 切替 |
| `balance_forced_halt_block` | balance_forced 阻止 |
| `preflight_insufficient` | 残高不足 |
| `one_sided_freeze_skip` / `one_sided_cooldown_skip` | 片側制限 |
| `hard_skip_utc` / `time_filter_both_sides` | 時間フィルター |
| `phantom_position_detected` / `phantom_veto_block` | ファントムポジション |
| `day_reset_kill_conflict` | 日次リセット競合 |
| `degraded_liquidation_*` | 縮退清算 |

### RECOVERY (回復動作): kill/halt からの復帰
| Guard | 説明 |
|-------|------|
| `dynamic_kill_probe_sell` / `*_buy` | kill probe 発火 |
| `dynamic_kill_force_release_*` | force release |
| `dual_kill_bypass` | 両 kill バイパス |
| `per_side_halt_recovery_active` | halt 回復中 |

## 実装

### 新規ファイル
- `scripts/v460/lib/guard_reason_classifier.py`: 分類ロジック
  - `GuardCategory` enum (MARKET/SYSTEM/RECOVERY)
  - `classify_guard(name)` → `GuardCategory`
  - `categorize_guard_fire_counts(counts)` → カテゴリ別内訳
  - `guard_category_totals(counts)` → カテゴリ別合計

### 変更ファイル
- `scripts/v460/lib/cycle_gate_aggregator.py`:
  - `CycleGateResult.blocking_category` property 追加
- `scripts/v460/lib/fill_loop_orchestrator.py`:
  - `_guard_category_totals()` helper + progress log にカテゴリ集計追加
  - state export に `guard_category_totals` 追加
- `scripts/v460/lib/resilience.py`:
  - `FillTestState.guard_category_totals` フィールド追加

## テスト
- 37 テスト (`test_244_guard_reason_classification.py`)
  - 28 parametrized 分類テスト (全 guard reason)
  - 集計・内訳分解テスト
  - `CycleGateResult.blocking_category` 統合テスト
- 全 3407 v460 テスト通過
