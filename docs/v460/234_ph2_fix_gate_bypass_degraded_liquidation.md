# 234# Gate Bypass 廃止・縮退清算・片側エスカレーション

## 概要

232# (Codex) と 233# (Gemini 3.1 Pro) の外部AIレビューで独立して合意された
**3つの構造的欠陥**を修正する。

## 合意事項 (232# + 233#)

両レビュアーは以下で一致:
1. FFD 230#/231# は堅実だが**ボトルネックではない**
2. `balance_forced` が Kill Gate をバイパスする設計が**致命的**
3. `one_sided_consecutive_limit` が警告のみで**ハードブレイカーがない**
4. `no_feasible_quote` (制約集合崩壊) が**未検出**

## 修正内容

### P0-A: balance_forced Gate Bypass 全廃 + 縮退清算モード

**問題**: `balance_forced=True` 時に全 Gate (`not balance_forced`) をバイパス。
これにより Kill Gate の安全機能が無効化され、プロアクティブ防御が機能しない状態で
リアクティブ防御 (FFD) のみに依存していた。

**修正**:
- `cycle_gate_aggregator.py` の全 `_check_*` メソッドから `not balance_forced` を削除
  - Gate 1: `_check_unknown_regime_buy`
  - Gate 2: `_check_ranging_buy_low_vol`
  - Gate 3: `_check_trending_sell` (特殊パス廃止、統一パスに一本化)
  - Gate 4: `_check_buy_dynamic_kill`
  - Gate 5: `_check_sell_dynamic_kill`
  - Gate 7: `_check_unknown_regime_sell`
  - dual_kill 条件: `_dual_kill = is_buy_killed and is_sell_killed` (not balance_forced 削除)

- **縮退清算モード** (Gate 4/5 with balance_forced):
  - Kill Gate blocked + balance_forced + degraded_liquidation_enabled → `result.degraded_liquidation=True`
  - 完全ブロックではなく、min lot (20%) + wide offset (3x) + duty cycle (1-in-3) で安全に縮退清算
  - Orchestrator でデューティサイクル制御
  - Executor で offset 拡大 + lot 縮小を適用

**設定パラメータ**:
```yaml
degraded_liquidation_enabled: true
degraded_liquidation_lot_mult: 0.2      # 通常 lot の 20%
degraded_liquidation_offset_mult: 3.0   # offset を 3 倍
degraded_liquidation_duty_cycle: 3      # 3 サイクルに 1 回のみ実行
```

### P0-B: one_sided_consecutive_limit エスカレーション (3段階)

**問題**: 片側連続取引の上限到達時、interval 延長の警告のみ。
ハードブレイカーがなく、損失が拡大し続ける。

**修正**: 3段階エスカレーション
1. **Stage 1** (limit 到達): interval ×3 (既存動作維持)
2. **Stage 2** (limit + cooldown_offset): cooldown — N サイクルスキップ
3. **Stage 3** (limit + freeze_offset): freeze — 当該 side を N サイクル凍結

**設定パラメータ**:
```yaml
one_sided_escalation_cooldown_offset: 2   # limit+2 で cooldown
one_sided_escalation_cooldown_cycles: 2   # 2 サイクルスキップ
one_sided_escalation_freeze_offset: 4     # limit+4 で freeze
one_sided_escalation_freeze_cycles: 3     # 3 サイクル凍結
```

**リセット**: `_one_sided_balance` が False になった時点で全カウンタリセット。

### P0-C: no_feasible_quote 早期検出

**問題**: spread 制約 (min_spread + sell_guard) の同時充足不能時、
`spread_too_narrow` と `sell_guard_reject` が交互に発生し、bot が無限ループ。

**修正**:
- Executor に `_consecutive_no_feasible` カウンタを追加
- `spread_too_narrow` / `sell_guard_reject` が 3 回連続 → `NO_FEASIBLE_QUOTE` に昇格
- 成功時 (`_compute_maker_price` 成功) でカウンタリセット
- `cancel_reasons.py` に `NO_FEASIBLE_QUOTE` 定数追加

## 変更ファイル

### 本体コード
| ファイル | 変更内容 |
|---------|----------|
| `cycle_gate_aggregator.py` | `not balance_forced` 全削除、degraded liquidation 判定追加 |
| `fill_loop_orchestrator.py` | duty cycle カウンタ、cooldown/freeze 制御、エスカレーション |
| `fill_cycle_executor.py` | degraded offset/lot 適用、no_feasible_quote 検出 |
| `fill_config.py` | 8 新設定フィールド + YAML パース |
| `cancel_reasons.py` | 4 新定数 (`NO_FEASIBLE_QUOTE`, `degraded_liquidation_duty_skip`, etc.) |

### テスト
| ファイル | 変更内容 |
|---------|----------|
| `test_234_gate_bypass_removal.py` | **新規** 32 テスト (P0-A/B/C 全カバー) |
| `test_113_resilience.py` | 行数上限 650→700 |
| `test_155_hindsight_review.py` | bypass テスト → 廃止検証テストに置換 |
| `test_194_cycle_gate.py` | bypass テスト → block/degraded テストに更新 |
| `test_195_velocity_b1_soft.py` | balance_forced bypass → block 検証に更新 |
| `test_196_velocity_proportional_trending_soft.py` | dead config テストに更新 |
| `test_197_boost_optimization_gate_integration.py` | 統一パステストに更新 |
| `test_220_deadlock_fixes.py` | bypass クラス → 廃止検証に全面改訂 |
| `test_223_review_response.py` | dual_kill_bypassed=True に更新 |
| `test_229_cleanup_counter_rename.py` | カウンタ挙動更新 |

### 設定
| ファイル | 変更内容 |
|---------|----------|
| `configs/v460/fill_test.yaml` | 8 新パラメータ追加 + dead config コメント |

## テスト結果

```
3185 passed, 0 failed (3153 既存 + 32 新規)
```

## 設計判断

### なぜ完全ブロックではなく縮退清算か

`balance_forced=True` は「在庫が片側に偏り、清算が必要」という状態を示す。
完全ブロックだと在庫がさらに偏り、状況が悪化する。
縮退清算 (min lot + wide offset + duty cycle) は:
- Kill Gate の安全意図を尊重 (通常の 20% lot, 3x offset)
- 在庫の漸次解消は可能 (完全停止より安全)
- duty cycle で頻度を制限 (市場への影響最小化)

### balance_forced_apply_trending_offset の dead config 化

Gate 3 (trending_sell) の特殊 balance_forced パスを廃止したため、
soft mode は balance_forced に関係なく常に offset を適用する。
`balance_forced_apply_trending_offset` は後方互換のため残すが、実質的に無効。
