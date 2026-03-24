# 604# ログ可観測性改善 + 膠着診断スクリプト

## 背景
602# 調査で「滞留注文→BTC locked→両側膠着→SAFE_STOP」パターンの特定に
時間を要した。ログから原因を迅速に特定できるよう、可観測性を改善する。

## 過去のログ改善履歴
| # | 内容 |
|---|------|
| 487# | cancel_reason 導入 |
| 508# | basis_bps 可観測性 |
| 526# | cancel ログに order_id、残高コンテキスト |
| 534# | log_cycle_no、BTC 残高 |
| 592# | ログ再監査 |

## 変更内容

### 1. balance_checker.py — locked 残高キャッシュ
- `_last_btc_locked` / `_last_jpy_locked` フィールド追加
- `last_btc_locked` / `last_jpy_locked` プロパティ追加
- `_check_sell()`: `btc_balances` から `locked` 合算をキャッシュ
- `_check_buy()`: `jpy_balances` から `locked` 合算をキャッシュ
- Insufficient BTC/JPY ログに `locked=` を追加

### 2. orchestrator_balance.py — preflight skip 詳細ログ
- `_handle_preflight_failure()`: 毎回 `[preflight_skip]` WARNING を出力
  ```
  [preflight_skip] count=N/M btc_free=X, btc_locked=Y, jpy_free=Z, jpy_locked=W
  ```
- 602# recovery 例外ハンドラに `exc_info=True` 追加

### 3. fill_cycle_executor.py — age_cap order_id + exc_info
- `sell_age_cap exceeded` ログに `order_id=%s` 追加
- 603# cancel 例外ハンドラに `exc_info=True` 追加

### 4. diagnose_deadlock.py — 膠着診断スクリプト (新規)
```
python -m scripts.v460.analysis.diagnose_deadlock [--log PATH] [--tail N]
```
- テキストログから膠着パターンを自動検出
- 604# 新形式 + 旧形式 (preflight_pause/balance_shrink) の両方に対応
- DeadlockEvent: skip count, pause, SAFE_STOP, recovery cancel, age_cap orders, btc_locked, jpy_free
- 原因推定付き診断レポート出力

### 5. テスト — 603# バグ修正 + 604# テスト追加
- `TestAgeCapCancelOrder.test_cancel_precedes_break`: `idx_open` 未定義バグ修正
- `TestAgeCapCancelOrder`: 重複 `test_safe_stop_still_reachable` を `test_age_cap_cancel_exc_info` に置換
- `TestLoggingObservability604`: ソース契約テスト 8 件
- `TestDiagnoseDeadlock604`: analyze_log / format_report テスト 6 件

## 602# 調査での痛点と対応
| 痛点 | 対応 |
|------|------|
| preflight 失敗時に `btc_locked` が不明 | Insufficient ログ + preflight_skip ログに locked 追加 |
| age_cap exceeded に order_id なし | ログに order_id= 追加 |
| skip 毎の残高推移が追えない | 毎回 [preflight_skip] WARNING 出力 |
| 例外が `str(e)` のみ | `exc_info=True` でトレースバック付与 |
| 膠着パターンを手動 grep で調査 | diagnose_deadlock.py で自動検出 |
