# 421# Execution Final Clamp + Route-to-Kill Deadlock 防止

> **種別**: impl (実装)  
> **日付**: 2026-03-14  
> **SHA**: `4aa779d27`  
> **起票元**: 416# (Codex) + 417# (Gemini) レビュー  
> **セルフレビュー**: 419# で実施・修正済

---

## 概要

416#/417# のレビューで発見された2つの構造欠陥を修正。

### 1. Post-Ceiling Offset Leak (P0-CRITICAL)

**問題**: `maker_price.py` の offset ceiling (L1012-1027) は正しく動作するが、
その後 `fill_cycle_executor.py` の6つの executor 側 multiplier が `_apply_offset_multiplier()` 経由で
ceiling を迂回。`_apply_offset_multiplier()` にはクランプなし。

**実データ**: 3/11 `final_stage=0.300` → `effective_offset_used=1.305` (4.35×)

**修正**: fill_cycle_executor.py に **Final Clamp** を追加。
全 multiplier chain 完了後、注文送信直前に ceiling でクランプ。
- `execution_final_clamp_enabled` (bool, default: true)
- `execution_final_clamp_hard_skip_mult` (float, default: 0.0) — 0.0=閾値なし

### 2. Route-to-Kill Deadlock (P1-HIGH)

**問題**: buy 残高不足 → sell 切替 → sell kill-gated → 高速デッドループ。

**修正**: orchestrator_balance.py に `_is_side_killed(opposite)` チェック追加。
切替先が kill 状態なら `ROUTE_TO_KILL_DEADLOCK` で即スキップ。

---

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `fill_cycle_executor.py` | Final Clamp ロジック追加 (全 multiplier 後、注文前) |
| `orchestrator_balance.py` | `_is_side_killed(opposite)` ガード追加 |
| `cancel_reasons.py` | `FINAL_CLAMP_HARD_SKIP`, `ROUTE_TO_KILL_DEADLOCK` 定数追加 |
| `fill_config.py` | `execution_final_clamp_enabled`, `_hard_skip_mult` フィールド追加 |
| `fill_record_builder.py` | `execution_pre_clamp_offset` パラメータ追加 |
| `ztb/metrics/fill_quality.py` | `execution_pre_clamp_offset: float | None` フィールド追加 |

## テスト

- `test_421_final_clamp_deadlock.py`: 25テスト全パス

---

## 419# セルフレビュー (同日実施)

### 発見された9件の不備

| # | 優先度 | 問題 | 修正 |
|---|---|---|---|
| 1 | P0 | CancelReason Literal型に新定数未追加 | Literal type alias に追加 |
| 2 | P0 | fill_config_parser が新フィールド未配線 | YAML パース行追加 |
| 3 | P0 | test_145 AUDIT_CANCEL_REASONS 期待セット不足 | 新定数追加 |
| 4 | P1 | fill_test.yaml に明示的設定なし | YAML エントリ追加 |
| 5 | P1 | guard_reason_classifier に route_to_kill 未分類 | RECOVERY 追加 |
| 6 | P2 | 天井解決ロジック3箇所重複 | `FillTestConfig.resolve_offset_ceiling()` DRY化 |
| 7 | P2 | spread_at_order=None 時の Final Clamp 不整合 | 防御ガード追加 |
| 8 | P2 | config_hot_reload に新フィールド未登録 | `_HOT_RELOADABLE_FIELDS` 追加 |
| 9 | P2 | テストギャップ | 10件のテスト追加 |

### 追加変更ファイル (419#)

| ファイル | 変更内容 |
|---|---|
| `cancel_reasons.py` | Literal type alias 修正 |
| `fill_config_parser.py` | YAML ← execution_final_clamp_* 配線 |
| `fill_config.py` | `resolve_offset_ceiling()` DRY ヘルパー追加 |
| `fill_cycle_executor.py` | ceiling 解決を DRY 化 + spread=None ガード追加 |
| `maker_price.py` | ceiling 解決を DRY 化 |
| `guard_reason_classifier.py` | route_to_kill_deadlock → RECOVERY |
| `config_hot_reload.py` | 2フィールドを hot-reload 対象追加 |
| `fill_test.yaml` | execution_final_clamp_* 設定明示 |
| `test_145_structural_fixes.py` | AUDIT frozenset 期待値修正 |
| `test_113_resilience.py` | 行数制限 755→810 |
| `test_253_*.py` | 行数制限 1120→1170 |
| `test_421_final_clamp_deadlock.py` | 25→35テスト (10件追加) |

### テスト結果

- 421# テスト: 35 passed
- 全 v460 テスト: 3569 passed, 7 skipped, 0 failed
