# 332# refactor: run_continuous Phase 4 — Balance/MidCycle Mixin 抽出

## 概要

`run_continuous` メソッドの残存インライン ~908 行を 2 つの新規 Mixin + 既存 Mixin への
追加により ~80 行のオーケストレーションに圧縮。God Object 分割の Phase 4 完了。

## 背景

- 328# タスク監査で `fill_loop_orchestrator.py` の God Object 問題を特定
- 325# (Guard/Lifecycle/PostCycle Mixin) → 329# (fill_config split) → 330# (pre-cycle 抽出) と段階的に分割
- 330# 後も `run_continuous` に ~908 行のインラインロジックが残存
- 目標: `run_continuous` < 500 行 → 実績: **~80 行** (目標大幅達成)

## 変更一覧

### 新規ファイル

| ファイル | 行数 | 責務 |
|---------|------|------|
| `orchestrator_balance.py` | 278 | Balance/preflight 解決: side 切替, inventory escape, preflight failure |
| `orchestrator_mid_cycle.py` | 688 | Mid-cycle 判定: one-sided skip, balance_forced skip, gate 評価, cycle 実行, sleep |

### 既存ファイル変更

| ファイル | 変更前 | 変更後 | 内容 |
|---------|--------|--------|------|
| `fill_loop_orchestrator.py` | 1228 | 407 | `run_continuous` 本文を ~80 行に圧縮、dead import 削除 |
| `orchestrator_pre_cycle.py` | 503 | 661 | CycleContext フィールド復活 + 3 メソッド追加 |

### テスト修正

ソース解析テスト (文字列検索) が `fill_loop_orchestrator.py` を直接参照していたものを
`read_fill_test_runner_source()` (全 Mixin 連結) に更新:

- `_fill_test_source.py`: 新 2 ファイルを `_FILL_TEST_RUNNER_SOURCES` に追加
- `test_091_fixes.py`: 3 テスト — balance ロジック参照先更新
- `test_139_review_fixes.py`: 3 テスト — preflight/sleep 参照先更新
- `test_145_structural_fixes.py`: 1 テスト — regime_mult 参照先更新
- `test_154_deadlock_prevention.py`: 1 テスト — rescue mode 参照先更新
- `test_155_hindsight_review.py`: 1 テスト — trending bypass 参照先更新
- `test_158_regime_deadlock_fix.py`: 5 テスト — regime update/skip 参照先更新
- `test_166_remaining_tasks.py`: 1 テスト — deadlock alternation 参照先更新
- `test_196_velocity_proportional_trending_soft.py`: 1 テスト
- `test_226_loss_boost_decay_inv_skew_state.py`: 1 テスト
- `test_227_ranging_obi_velocity_ema_import_fix.py`: 2 テスト
- `test_229_cleanup_counter_rename.py`: 1 テスト
- `test_276_blocking_policy_dry.py`: 1 テスト

## Mixin 構成 (332# 後)

```
FillLoopOrchestratorMixin (407 行)
  ├─ OrchestratorBalanceMixin    (278 行)  ← NEW
  ├─ OrchestratorGuardsMixin     (246 行)
  ├─ OrchestratorLifecycleMixin  (538 行)
  ├─ OrchestratorMidCycleMixin   (688 行)  ← NEW
  ├─ OrchestratorPostCycleMixin  (448 行)
  └─ OrchestratorPreCycleMixin   (661 行)
                            合計: 3,266 行 (8 ファイル)
```

## run_continuous フロー (332# 後)

```
run_continuous (~80 行)
  ├─ _init_run_session()               [lifecycle]
  ├─ while loop:
  │   ├─ _process_daily_reset()        [pre_cycle]
  │   ├─ _handle_dd_halt()             [guards]
  │   ├─ _check_alert_mode()           [pre_cycle]  ← NEW
  │   ├─ _check_circuit_breakers()     [guards]
  │   ├─ _check_hard_skip_utc()        [guards]
  │   ├─ _process_phantom_guard()      [guards]
  │   ├─ _prepare_cycle_context()      [pre_cycle]  ← NEW
  │   ├─ _resolve_side_vetos()         [pre_cycle]
  │   ├─ _apply_time_filter()          [pre_cycle]
  │   ├─ _update_regime_fallback()     [pre_cycle]  ← NEW
  │   ├─ _resolve_balance_and_preflight() [balance] ← NEW
  │   ├─ _handle_one_sided_skip()      [mid_cycle]  ← NEW
  │   ├─ _handle_balance_forced_skip() [mid_cycle]  ← NEW
  │   ├─ _handle_forced_buy_delay()    [mid_cycle]  ← NEW
  │   ├─ _evaluate_and_handle_cycle_gate() [mid_cycle] ← NEW
  │   ├─ _handle_toxicity_skip()       [mid_cycle]  ← NEW
  │   ├─ _handle_degraded_liquidation() [mid_cycle] ← NEW
  │   ├─ _execute_and_track_cycle()    [mid_cycle]  ← NEW
  │   └─ _post_cycle_sleep()           [mid_cycle]  ← NEW
  └─ _finalize_run()                   [lifecycle]
```

## テスト結果

```
4105 passed, 15 warnings in 31.76s
```
