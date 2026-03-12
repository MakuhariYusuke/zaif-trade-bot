# 330# run_continuous pre-cycle 抽出 + σ floor + ゼロ除算ガード

> 2026-03-13 | refactor + bugfix | 328# Phase 3 実行 + deep-scan quick-win

---

## 概要

328# タスク棚卸しの Phase 3「run_continuous God Method Split」を実行。
`run_continuous` の先頭 ~372 行（pre-cycle ガードチェーン）を
新ファイル `orchestrator_pre_cycle.py` に抽出し、
`CycleContext` データクラスで状態を受け渡す設計に移行。

並行して deep-scan で発見されたバグ 5 件を quick-win として修正。

---

## 成果

### A. orchestrator_pre_cycle.py 新規作成 (~380 行)

| メソッド | 役割 | 旧位置 (run_continuous) |
|---|---|---|
| `_process_daily_reset()` | 日次境界リセット (soft_dd_mult, lot_scale, toxic_veto, one_sided, kill_mgr) | §9.0 直後 |
| `_handle_dd_halt(st) → bool` | DD halt チェック + MCB/SAD feed + halt 終了 cleanup | §9.1 |
| `_check_circuit_breakers(st) → bool` | MCB HALT/WARNING, SAD FROZEN/DRY/WIDE, MCB×SAD escalation | §9.2-§9.3 |
| `_check_hard_skip_utc(st) → bool` | UTC 時間帯 hard skip (entering/exit 追跡) | §9.4 |
| `_process_phantom_guard()` | tick_veto + reconcile + phantom 検出ハンドリング | §9.4.5 |
| `_resolve_side_vetos(st, ctx) → bool` | per-side DD + toxic veto + phantom veto → side 切替 | §9.5 |
| `_apply_time_filter(st, ctx) → bool` | time filter + 086# デッドロック + heartbeat + batch flush | §9.6 |

**CycleContext dataclass**: `next_side`, `balance_forced`, `is_rescue`, `one_sided_balance`, `inventory_escape`, `regime_mult` の 6 フィールド。

### B. fill_loop_orchestrator.py 縮小

- **1595 → 1223 行** (-372 行, -23.3%)
- `OrchestratorPreCycleMixin` を MRO に追加
- run_continuous の pre-cycle 部分が 7 メソッド呼び出しに圧縮

### C. Deep-scan quick-win バグ修正 (5 件)

| ID | 修正内容 | ファイル |
|---|---|---|
| T4/B3 | σ floor 導入 (`sigma_floor=1e-6`) — σ=0 時のゼロ除算防止 | maker_microstructure.py, fill_config.py |
| B1 | `_scale_offset_ratio` ゼロ除算ガード | maker_price.py |
| B2 | `adaptation_engine` base_offset_ratio ゼロ除算ガード (`max(x, 1e-8)`) | adaptation_engine.py |
| B4 | `kyle_lambda_enabled + !imbalance_enabled` 組合せ warning | fill_config_validation.py |
| T5 | `vol_ratio_floor` の config 化 (hardcoded 0.1 → `cfg.vol_ratio_floor`) | maker_microstructure.py, fill_config.py, fill_config_parser.py |

---

## テスト影響

Source-inspection テスト (run_continuous ソースコードを直接検査するテスト) 10 ファイルを更新。
抽出先メソッドまたは `read_fill_test_runner_source()` を参照するよう移行。

| テストファイル | 修正内容 |
|---|---|
| `_fill_test_source.py` | `orchestrator_pre_cycle.py` をソースリストに追加 |
| `test_091_fixes.py` | `orchestrator_pre_cycle.py` を直接検査 |
| `test_139_review_fixes.py` | `read_fill_test_runner_source()` 使用 |
| `test_203_dd_state_persistence.py` | `_handle_dd_halt` を検査 |
| `test_211_mcb_sad_escalation.py` | `_check_circuit_breakers` を検査 |
| `test_226_loss_boost_decay_inv_skew_state.py` | `_handle_dd_halt` を検査 |
| `test_230_ffd_deadzone_streak_guards.py` | `orchestrator_pre_cycle` モジュール検査 |
| `test_254_frozen_side_persist_getattr_cleanup.py` | `orchestrator_pre_cycle` モジュール検査 |
| `test_266_market_theory_protocol.py` | sigma_floor 動作変更に合わせた期待値更新 |
| `test_275_dry_separation_and_theory.py` | `read_fill_test_runner_source()` 使用 |
| `test_276_blocking_policy_dry.py` | 両ファイル結合ソースで検査 |
| `test_281_deadlock_fix.py` | `_resolve_side_vetos` を検査 |
| `test_fill_test_config.py` | `read_fill_test_runner_source()` 使用 |
| `test_regime_detector.py` | `_apply_time_filter` を検査 |

**最終結果: 4105 passed, 0 failed**

---

## ファイル一覧

### 新規作成
- `scripts/v460/lib/orchestrator_pre_cycle.py`

### 変更 (プロダクション)
- `scripts/v460/lib/fill_loop_orchestrator.py` (1595→1223 行)
- `scripts/v460/lib/maker_microstructure.py` (σ floor + vol_ratio_floor)
- `scripts/v460/lib/maker_price.py` (ゼロ除算ガード)
- `scripts/v460/lib/adaptation_engine.py` (ゼロ除算ガード)
- `scripts/v460/lib/fill_config.py` (sigma_floor, vol_ratio_floor フィールド追加)
- `scripts/v460/lib/fill_config_parser.py` (sigma_floor, vol_ratio_floor パース)
- `scripts/v460/lib/fill_config_validation.py` (kyle_lambda+imbalance warning)

### 変更 (テスト)
- 14 テストファイル (詳細は上表)

---

## 残タスク (331# 以降)

- D1: skip_gate_evaluator `except Exception` → 具体例外への絞り込み (12 箇所)
- D2: fill_cycle_executor `except Exception` 分離
- D3: order_monitor cancel 失敗 → phantom_position_guard 統合
- T1: Parkinson σ intraday scaling 検証
- T2: AS fill_rate_k 実証キャリブレーション
- T6: Inventory imbalance EMA 平滑化
- T8: inv_skew trending regime partial disable
- T9: Toxicity budget 情報非対称性検出 (利益時)
- Phase 4: Balance/gate/execution ロジック抽出 (~800 行残)
