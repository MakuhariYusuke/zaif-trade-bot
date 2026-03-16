# 325# fill_loop_orchestrator God Object 分割

## 背景

`fill_loop_orchestrator.py` は **2,849 行**に膨張し、プロジェクト最大の God Object であった。
38 メソッドが単一クラスに密集し、責務の混在が保守性を大幅に低下させていた。

322# (maker_price → 3 Mixin) / 323# (executor → 2 Mixin) の分割パターンを踏襲し、
orchestrator を 3 つの Mixin に分離する。

## 分割結果

| ファイル | 行数 | 責務 |
|---|---:|---|
| `fill_loop_orchestrator.py` | 1,594 | run_continuous + ループユーティリティ |
| `orchestrator_guards.py` | 247 | リスクガード評価, kill 判定, veto 管理 |
| `orchestrator_lifecycle.py` | 574 | セッション初期化/終了, 状態 snapshot/restore |
| `orchestrator_post_cycle.py` | 448 | サイクル後処理, 進捗ログ, adaptation 委譲 |
| **合計** | **2,863** | (オーバーヘッド: docstring + import で +14 行) |

**削減率: 2,849 → 1,594 行 (44% 削減)**

## Mixin 責務マッピング

### OrchestratorGuardsMixin (orchestrator_guards.py)
14 メソッド — リスクガード評価, kill 判定, toxic veto

| メソッド | 行数 | 概要 |
|---|---:|---|
| `_is_side_killed` | 60 | Glosten-Milgrom kill (286# 在庫緩和付き) |
| `_track_side_pnl` | 16 | side 別 PnL 追跡 |
| `_assess_toxicity` | 17 | DynamicKillManager 委譲 |
| `_assess_buy/sell_toxicity` | 3+3 | side ラッパー |
| `_inc_guard_fire` | 5 | 発火カウンタ increment |
| `_guard_category_totals` | 6 | guard_reason_classifier 委譲 |
| `_tick_toxic_veto` | 13 | veto デクリメント |
| `_feed_mcb_sad` | 15 | MCB/SAD フィード |
| `_opposite_side` | 3 | 反対サイド (static) |
| `_check_regime_stop_conditions` | 28 | fill_rate/pnl 停止条件 |
| `_is_time_filtered` | 7 | TimeFilter 委譲 |
| `_check_balance_for_side` | 11 | BalanceChecker 委譲 (async) |
| `_cancel_stale_orders` | 36 | 滞留注文キャンセル (async) |

### OrchestratorLifecycleMixin (orchestrator_lifecycle.py)
8 メソッド — セッション生存期間管理

| メソッド | 行数 | 概要 |
|---|---:|---|
| `_warmup_daily_drawdown_from_records` | 65 | DD guard warmup from fill records |
| `_warmup_kill_managers_from_records` | 38 | kill manager PnL 履歴 replay |
| `_build_state_snapshot` | 66 | FillTestState 構築 |
| `_maybe_skip_state_save` | 19 | 時間ゲート付き state save |
| `_restore_common_state` | 93 | DD/veto/one-sided/guard_fire 復元 |
| `_init_run_session` | 175 | セッション初期化 (async) |
| `_finalize_run` | 40 | 最終クリーンアップ (async) |
| `_cleanup_sync` | 69 | atexit 同期 cleanup |

### OrchestratorPostCycleMixin (orchestrator_post_cycle.py)
9 メソッド — サイクル後処理 + adaptation

| メソッド | 行数 | 概要 |
|---|---:|---|
| `_process_post_cycle` | 180 | PnL 追跡, loss cooldown, DD update, loss_cap |
| `_log_progress_and_adapt` | 152 | 進捗ログ, state save, adaptation (async) |
| `_compute_dynamic_interval` | 19 | σ 連動サイクル間隔 |
| `cleanup_heartbeat` | 16 | heartbeat task cleanup (async) |
| `_build_adapt_kwargs` | 3 | AdaptationEngine 委譲 |
| `_build_lot_kwargs` | 3 | AdaptationEngine 委譲 |
| `_update_dynamic_loss_cap` | 5 | 動的 loss_cap 委譲 (async) |
| `_try_auto_adapt` | 18 | 自動 adaptation 委譲 |
| `_try_auto_lot_size` | 8 | lot sizing 委譲 |

## MRO / 継承構造

```
FillLoopOrchestratorMixin(
    OrchestratorGuardsMixin,
    OrchestratorLifecycleMixin,
    OrchestratorPostCycleMixin,
)

FillTestRunner(
    FillRecordHelpersMixin,
    FillCycleExecutorMixin,
    FillLoopOrchestratorMixin,  # 3 Mixin を内包
    AbstractCycleRunner,
)
```

既存テストの import パス (`from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin`)
は MRO 継承により変更不要。

## テスト結果

- **4096 passed** (v460 全テスト)
- 既存 integration failures (CustomPPO, AnomalyDetector) は 325# 無関係

## 残タスク

- `run_continuous` (1,280 行) はローカル変数の相互依存により更なる分割は現時点で非効率
- `fill_config.py` (1,954 行) が次の God Object 分割候補
