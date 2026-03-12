# 238# 237# セルフレビュー: 型安全 + 残高スナップショット + TTL + サイドベトー

## 概要

237# PhantomPositionGuard のセルフレビューで発見した 8 件の問題を修正。
型安全性の向上（Protocol 導入）、dead code の解消、市場理論に基づくサイドベトー機構を追加。

## 発見した問題と対策

### CRITICAL (3件)

| ID | 問題 | 対策 |
|----|------|------|
| C-1 | `_phantom_guard: object \| None` — mypy がメソッド呼び出しを検証不能 | `PhantomPositionGuard` 型を TYPE_CHECKING conditional import で使用 |
| C-2 | `balance_btc`/`balance_jpy` が `register_unknown()` に未渡し — Phase 2 残高差分照合が本番 dead code | `BalanceChecker` に `last_btc_free` キャッシュを追加、executor から転送 |
| C-3 | `getattr(status, "status", "")` — 236# で排除した `hasattr` と同根の anti-pattern | `_OrderStatusResult` Protocol 導入、直接属性アクセスに書換 |

### SIGNIFICANT (4件)

| ID | 問題 | 対策 |
|----|------|------|
| S-1 | pending エントリに TTL なし — stale エントリが永続蓄積 | `_MAX_PENDING_AGE_SEC = 300.0` 自動パージ追加 |
| S-2 | 232# §1.6 "cautious side bias" 未実装 — phantom 検知後の反応なし | `_PHANTOM_VETO_CYCLES = 3` サイドベトー機構（Avellaneda-Stoikov §3.2 準拠） |
| S-3 | CRITICAL ログが reconcile_single + reconcile + orchestrator で 3 重出力 | `reconcile()` の CRITICAL ログ削除、orchestrator 側に一本化 |
| S-4 | `resolved` リストが常に全 pending と一致 — 不要な複雑性 | `self._pending.clear()` に簡素化 |

### LOW (1件)

| ID | 問題 | 対策 |
|----|------|------|
| L-1 | テストファイルの dead import `Optional` | 削除、`X \| None` 構文に統一 |

## 市場理論的根拠

phantom fill（意図しない約定）は **逆選択 (adverse selection)** の強い指標。
高ボラティリティ・高フロー期に発生しやすく、その後の同サイド注文も逆選択コストが高い。

**Avellaneda-Stoikov (2008) §3.2** の最適ビッド・アスク理論に基づき:
- phantom 検知 → 当該サイドを一時的にベトー（`_PHANTOM_VETO_CYCLES = 3` サイクル）
- 両サイドベトー → `CR.PHANTOM_SIDE_VETO` でサイクルスキップ
- ベトーはサイクルごとに `tick_veto()` でデクリメント、0 で自然解除

`_toxic_veto` と同パターンの設計で、既存のリスク管理フレームワークと整合。

## 変更ファイル

### A. phantom_position_guard.py

- `_OrderStatusResult` Protocol 追加（`.status: str` 必須）
- `getattr(status, "status", "")` → `status.status` 直接アクセス (C-3)
- TTL: `_MAX_PENDING_AGE_SEC = 300.0` — `reconcile()` 冒頭でパージ (S-1)
- サイドベトー: `_PHANTOM_VETO_CYCLES = 3`, `_side_veto: dict[str, int]` (S-2)
- `is_side_vetoed(side)`, `tick_veto()` メソッド追加
- `reconcile()`: 検知時に `_side_veto[detection.side] = _PHANTOM_VETO_CYCLES` 設定
- `reconcile()`: CRITICAL ログ削除 (S-3)
- `reconcile()`: `resolved` リスト廃止 → `self._pending.clear()` (S-4)
- `clear()`: `_side_veto` もクリア
- `get_metrics()`: `side_veto_active` フィールド追加

### B. balance_checker.py

- `_last_btc_free: float | None = None` キャッシュ追加
- `_last_jpy_free: float | None = None` キャッシュ追加
- `last_btc_free` プロパティ追加
- `_check_sell()`: `btc_free` をキャッシュ
- `_check_buy()`: `jpy_free` をキャッシュ

### C. fill_cycle_executor.py

- `_phantom_guard: object | None` → `PhantomPositionGuard | None` (C-1)
- TYPE_CHECKING conditional import 追加
- `_maybe_register_phantom()`: `balance_btc=self._balance_checker.last_btc_free` 転送 (C-2)

### D. fill_loop_orchestrator.py

- `_phantom_guard: object` → `PhantomPositionGuard` (C-1)
- TYPE_CHECKING conditional import 追加
- `tick_veto()` 呼び出し追加（サイクル冒頭）
- phantom サイドベトーブロック追加（toxic_veto の後、時間フィルターの前）:
  - 当該サイドベトー → 反対サイドに切替
  - 両サイドベトー → `CR.PHANTOM_SIDE_VETO` でスキップ

### E. cancel_reasons.py

- `PHANTOM_SIDE_VETO = "phantom_side_veto"` 定数追加
- `AUDIT_CANCEL_REASONS` / `CANCEL_REASONS_ALL` に追加

### F. テスト

- dead import `Optional` 削除 (L-1)
- 12 新テスト追加:
  - `TestTTLPurge` (2): stale パージ / fresh 保持
  - `TestSideVeto` (4): 初期状態 / phantom 設定 / tick デクリメント / clear リセット
  - `TestMetricsExtended` (2): side_veto_active フィールド
  - `TestBalanceCheckerCache` (2): 初期 None / check_sell 後キャッシュ
  - `TestCancelReasonPhantom` (2): 定数存在 / AUDIT セット所属
- `test_145_structural_fixes.py`: `CR.PHANTOM_SIDE_VETO` を期待セットに追加

## テスト結果

- 変更前: 3252 tests passed
- 変更後: **3264 tests passed** (+12 新規)
- mypy: エラーなし
