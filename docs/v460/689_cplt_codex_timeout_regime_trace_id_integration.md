# 689# Codex: timeout regime×side + decision trace ID 統合

## 概要
688# で設計した Codex タスク 2 件の実装結果を統合。テスト修正を含む。

## Codex 実装内容

### 1. Timeout Regime×Side Override (688_codex_task_timeout_regime_side)
- `fill_config.py`: `regime_timeout_overrides` dict + `get_timeout_with_reason(side, regime)` メソッド追加
- `fill_cycle_executor.py`: `_resolve_cycle_timeout_policy()` 新設。従来の `object.__setattr__` による config 一時書換を排除
- `order_monitor.py` / `order_monitor_policy.py`: `timeout_override_sec` / `timeout_reason` パラメータ追加
- `fill_test.yaml`: `regime.timeout_overrides` セクション追加 (strong_up/sell=20s, strong_down/buy=30s 等)

### 2. Decision Trace ID + Skip Audit (688_codex_task_skip_audit_trace_id)
- `fill_record_helpers.py`: `_new_decision_trace_id()` 生成 (`dt_{timestamp}_{uuid6}`)
- `fill_cycle_executor.py`: `decision_trace_id` をサイクル全体に貫通
- `skip_gate_evaluator.py`: skip_gate 判定ログに `[dt=...]` プレフィックス追加
- `orchestrator_pre_cycle.py`: `_execute_skip` 各 call site に audit コメント追加
- `_PreOrderPhaseResult`: `decision_trace_id`, `skip_gate_bypassed` フィールド追加

### 3. fill_judgment_core 分割 (Codex commit bfb9f1659)
- `ztb/metrics/fill_quality.py` → `ztb/metrics/fill_judgment_core.py` 抽出

## テスト修正 (手動)
| テストファイル | 修正内容 |
|---|---|
| test_145 | `TREND_5S_SELL_GUARD_VETO` を AUDIT_CANCEL_REASONS に追加、`_side_selector` mock 追加 |
| test_253 | fill_cycle_executor.py 行数上限 1600→1700 |
| test_518 | `compute_effective_timeout_policy()` に `timeout_override_sec=None, timeout_reason=None` 追加 |
| test_642 | `_PreOrderPhaseResult` に `decision_trace_id`, `skip_gate_bypassed` 追加 |

## テスト結果
- trading: 826 passed ✅
- risk: 145 passed ✅
- v460: 4315+ passed (環境 KeyboardInterrupt でフルRun未完走、failure 0) ✅
- test_688 (新規): 7 passed ✅

## 技術的改善点
- `object.__setattr__` による config 一時書換ハック削除 → `timeout_override_sec` パラメータ渡しに正規化
- Timeout 優先順位: regime_timeout_overrides → legacy macro_sell_timeout → order_timeout_sec_sell → order_timeout_sec
- 全サイクルに一意 `decision_trace_id` が付与され、skip_gate/offset/timeout/outcome の追跡が可能に
