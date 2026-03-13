# 345# プロアクティブ修正: warmup downweight 整合 / CircuitBreaker Py3.12+ 互換

## 概要

ボット稼働中に先行着手可能な課題を消化する。

- **A**: warmup 時の `forced_fill_pnl_downweight` 不整合修正
- **B**: `CircuitBreaker` sync メソッドの `asyncio.new_event_loop()` 排除 (Py3.12+ 互換)
- **C**: 324# §5 M-1/L-2 ステータス更新 (344# で完了済み)

## 1. A: warmup forced_fill_pnl_downweight 整合性

### 問題

`orchestrator_lifecycle._warmup_kill_managers_from_records()` は fill records を
replay して kill manager の PnL 履歴を復元するが、
343# で導入した `forced_fill_pnl_downweight` を適用していなかった。

| パス | forced fill の扱い |
|---|---|
| **ライブ** (`orchestrator_guards._track_fill_pnl`) | PnL × 0.5 で track |
| **warmup** (`orchestrator_lifecycle._warmup_kill_managers_from_records`) | PnL そのまま track (**不整合**) |

再起動後に kill manager の rolling mean がライブ時と異なる値に収束し、
kill 判定の精度が歪む要因となる。

### 修正

warmup でも `getattr(r, "balance_forced_switch", False)` をチェックし、
`self.config.forced_fill_pnl_downweight` を乗算。
`weight <= 0.0` の場合は完全除外 (旧 337# 挙動)。

ログに `skipped_forced` カウンタを追加。

## 2. B: CircuitBreaker sync メソッド Py3.12+ 互換

### 問題

`ztb/utils/circuit_breaker.py` の `_on_success_sync()` / `_on_failure_sync()` が
毎回 `asyncio.new_event_loop()` + `asyncio.set_event_loop()` を呼んでいた。

1. Python 3.12+ でのベストプラクティスに反する
2. `set_event_loop()` がスレッド全体のグローバル状態を変更し、他の async コードと干渉する可能性
3. asyncio.Lock は同期コンテキストでは不要 (GIL で十分)

### 修正

sync メソッドを `_on_success()` / `_on_failure()` のロジック直接実装に置き換え。
asyncio への依存を排除し、属性変更のみの軽量処理に変更。

## 3. C: 324# §5 ステータス更新

| ID | 内容 | 旧ステータス | 新ステータス |
|---|---|---|---|
| M-1/L-2 | velocity_ema_alpha | 保留 (データ蓄積待ち) | ✅ 344# で完了 (α=0.3) |

## 4. テスト結果

```
4259 passed, 0 failed (51.38s)
```

344# の 4227 passed から 32 テスト増加。

### 新規テスト (test_345_proactive_fixes.py)

| クラス | テスト数 | 内容 |
|---|---|---|
| TestWarmupDownweight | 7 | forced fill downweight 整合性 (通常/forced/zero/混在/古record/live一致) |
| TestCircuitBreakerSyncMethods | 10 | sync success/failure/threshold/record/no-event-loop |

## 5. 残存課題

| ID | 内容 | 優先度 | 備考 |
|---|---|---|---|
| S-2 | Sell Hour Boost | 保留 | post-310# データ分析が前提 |
| S-6 | buy ev_offset | 保留 | 分析先行 |
| 342#E | sell post_fill_wait_sec | LOW | 非対称性は意図的 |
| 342#F | velocity AS-aware | LOW | 複雑・リスク大 |

## 6. 変更ファイル一覧

| ファイル | 変更種別 |
|---|---|
| `scripts/v460/lib/orchestrator_lifecycle.py` | A: warmup downweight 適用 |
| `ztb/utils/circuit_breaker.py` | B: sync メソッド Py3.12+ 互換化 |
| `docs/v460/324_phg_fix_residual_tasks_and_regime_reuse.md` | C: M-1/L-2 ステータス更新 |
| `tests/unit/v460/test_345_proactive_fixes.py` | 新規: 17 テスト |
| `docs/v460/345_proactive_fixes.md` | 新規作成 |
| `docs/v460/index.md` | 345# エントリ追加 |
| `CHANGELOG.md` | 345# エントリ追加 |
