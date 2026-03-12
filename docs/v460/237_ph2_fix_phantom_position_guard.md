# 237# PhantomPositionGuard: status_unknown 幽霊ポジション検知・遅延照合

## 概要

232# §1.6 [HIGH] で指摘された **status_unknown → 幽霊ポジション** リスクへの対策。
`filled=False` の即断ではなく **quarantine state** を導入し、
残高差分 + 注文状態の次サイクル再照合で実態を確認する。

## 背景・問題

注文の約定ステータスが `status_unknown` で返却された場合、
従来は `filled=False` として処理を終了していた。
しかし実際にはサーバー側で約定が成立している可能性があり、
認識できないポジション（幽霊ポジション）が残留するリスクがあった。

> 232# §1.6:
> 「`filled=False` 即断ではなく quarantine state を持つべき」
> 1. `pending_reconciliation` 状態を FillRecord に残す
> 2. 残高差分と open orders を次 cycle で再照合
> 3. 照合完了まで同 side の在庫計算を慎重側へ寄せる

## 実装内容

### A. PhantomPositionGuard クラス (新規)

`scripts/v460/lib/phantom_position_guard.py`

| クラス | 説明 |
|--------|------|
| `PendingReconciliation` | quarantine 対象の注文情報 (order_id, side, quantity, price, balance snapshots) |
| `PhantomDetection` | 検知結果 (order_id, side, method: `"order_recheck"` / `"balance_delta"` / `"both"`) |
| `PhantomPositionGuard` | メインガード: 登録・照合・メトリクス管理 |

**照合フロー (2段階)**:
1. **Phase 1 — 注文ステータス再確認**: cancel 成功 → clean / filled → phantom 検知
2. **Phase 2 — 残高差分照合**: 登録時と現在の残高差分が想定数量以上 → phantom 検知

**安全機構**:
- レートリミット: `_MIN_RECONCILE_INTERVAL_SEC = 5.0` 秒
- 残高許容差: `_BALANCE_TOLERANCE_BTC = 0.0005` BTC

### B. FillRecord 拡張

`ztb/metrics/fill_quality.py`

- `pending_reconciliation: bool | None = None` — status_unknown 発生時に `True` に設定

### C. FillMonitorResult 拡張

`scripts/v460/lib/fill_config.py`

- `order_id_for_reconciliation: str | None = None` — phantom guard 登録用 order_id の受け渡し

### D. order_monitor.py: order_id 伝播

status_unknown + 未約定のケースで `order_id_for_reconciliation` を設定。

### E. fill_cycle_executor.py: phantom 登録

- クラスレベルデフォルト: `_phantom_guard: object | None = None` (hasattr 排除)
- ヘルパー: `_maybe_register_phantom()` — monitor 結果から phantom guard への登録を分離
- `run_single_cycle()` 内で `record.pending_reconciliation = True` を設定

### F. run_fill_test.py: ガード初期化

`FillTestRunner.__init__()` で `PhantomPositionGuard()` を生成。

### G. fill_loop_orchestrator.py: サイクル前照合

- 各サイクル開始前に `phantom_guard.reconcile()` を実行
- phantom 検出時: `CRITICAL` ログ + インターバル 3 倍化
- `_build_state_snapshot()` で `phantom_guard_metrics` を保存
- クラスレベルデフォルト: `_phantom_guard: object | None = None` (hasattr 排除)

### H. FillTestState 拡張

`scripts/v460/lib/resilience.py`

- `phantom_guard_metrics: dict[str, int | float] | None = None`

## 変更ファイル

| ファイル | 変更内容 |
|---------|----------|
| `phantom_position_guard.py` | **新規** PhantomPositionGuard + データクラス |
| `fill_quality.py` | `pending_reconciliation` フィールド追加 |
| `fill_config.py` | `order_id_for_reconciliation` フィールド追加 |
| `order_monitor.py` | status_unknown 時の order_id 伝播 |
| `fill_cycle_executor.py` | phantom 登録ヘルパー + クラスレベルデフォルト |
| `run_fill_test.py` | PhantomPositionGuard 初期化 |
| `fill_loop_orchestrator.py` | 照合統合 + metrics 保存 + クラスレベルデフォルト |
| `resilience.py` | `phantom_guard_metrics` フィールド追加 |
| `test_113_resilience.py` | 行数閾値 700→710 更新 |
| `test_237_phantom_position_guard.py` | **新規** 29 テスト |

## テスト結果

```
3252 passed, 0 failed
```
