# 174# Fresh Code Review — 新規バグ修正

> **日付**: 2026-02-27  
> **ベース**: `4ebde0326` (173# Code Review Sweep)  
> **コミット**: `ea20c41a4`  
> **テスト**: 54 passed (config/validation), 137 passed (regime/skip_gate)

---

## 1. 概要

173# で包括的コードレビューを実施した後、さらに別観点からの Fresh Code Review を行い、
**CRITICAL 1 件 / HIGH 5 件 / MED 1 件**の問題を発見・修正。
加えて未修正の追加対応推奨事項を識別。

---

## 2. 修正内容

### 2.1 CRITICAL — 戻り値欠落

| ファイル | 問題 | 修正 |
|---------|------|------|
| `fill_loop_orchestrator.py` | `_cancel_stale_orders()` が成功パスで `cancelled_count` を返さず `None` を返す | `return cancelled_count` 追加 |

呼び出し元が戻り値で分岐しているため、`None` が返ると cancel 成功判定が常に失敗していた。

### 2.2 HIGH — 監査セット・バリデーション

| # | ファイル | 問題 | 修正 |
|---|---------|------|------|
| H1 | `cancel_reasons.py` | `SKIP_GATE`, `SKIP_GATE_RULE_VELOCITY_SELL/BUY` が `AUDIT_CANCEL_REASONS` に欠落 → quarantine bypass 誤判定 | 3 メンバー追加 |
| H2 | `skip_gate_evaluator.py` | `_valid_regimes` に `trending_up` / `trending_down` 欠落 → 156# D-4 の方向別 regime が偽警告 | 2 メンバー追加 |
| H3 | `config_hot_reload.py` | side 別 fast_fill フィールド 4 件が `_HOT_RELOADABLE_FIELDS` に欠落 | 4 フィールド追加 |
| H4 | `config_hot_reload.py` | `post_fill_wait_sec` (base) が reloadable でない | 追加 |
| H5 | `fill_config.py` | `daily_drawdown_soft_limit_bps < hard_limit_bps` の順序逆転検出 | `__post_init__` バリデーション追加 |

### 2.3 MED — バリデーション追加

| ファイル | 問題 |
|---------|------|
| `fill_config.py` | `inventory_skewing_window < 0`, `sell_dynamic_kill_window < 1`, `sell_offset_floor_inv_discount ∉ [0,1]` を検出する `__post_init__` バリデーション追加 |

---

## 3. 追加対応推奨事項 (識別のみ、未修正)

| # | ファイル | 内容 |
|---|---------|------|
| 7 | `maker_price.py`, `order_monitor.py`, `skip_gate_evaluator.py`, `balance_checker.py` | `object` 型注釈 → Protocol 型化 |
| 8 | `skip_gate_evaluator.py` | `FillRecord` 重複 import 4 箇所 |
| 10 | `adapter.py` | `InsufficientFundsError` 検出が英語パターンのみ、日本語エラー未対応 |
| 12 | `order_monitor.py` | stale 検出の side 別セレクタ冗長 |
| 13 | `config_hot_reload.py` | stale 系 side 別フィールド 6 件が reloadable でない |

> 上記は 175# で部分的に対応済み (Protocol 型化、stale side 別フィールド追加等)。

---

## 4. 変更ファイル一覧

| # | ファイル | 優先度 |
|---|---------|--------|
| 1 | `scripts/v460/lib/fill_loop_orchestrator.py` | CRITICAL |
| 2 | `scripts/v460/lib/cancel_reasons.py` | HIGH |
| 3 | `scripts/v460/lib/skip_gate_evaluator.py` | HIGH |
| 4 | `scripts/v460/lib/config_hot_reload.py` | HIGH |
| 5 | `scripts/v460/lib/fill_config.py` | HIGH + MED |
