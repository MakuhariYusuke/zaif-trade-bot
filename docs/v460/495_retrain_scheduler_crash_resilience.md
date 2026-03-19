# 495# Retrain Scheduler Crash Resilience

**日付**: 2026-03-20
**分類**: P0 バグ修正 — 再学習スケジューラの繰り返しクラッシュ防止

---

## 1. 背景

再学習スケジューラ (`sac_retrain_scheduler.py`) が繰り返し無言で死亡する現象。
`sac_retrain_history.jsonl` の履歴 (5 件中 3 件がエラー) および `retrain_scheduler.log` の
PID 遷移 (`21168→71004→85352→73088→81112→86000`) が多数の再起動を示す。

### エラー履歴 (sac_retrain_history.jsonl)
| 日付 | status | エラー内容 |
|------|--------|-----------|
| 3/11 | error | DLL initialization (c10.dll) ×2 |
| 3/19 | error | Timestamp TypeError (修正済: 07487339a) |
| 3/19 | deployed | 正常デプロイ (1156s) |
| 3/19 | oos_failed | OOS validation 失敗 (731s) |

## 2. 根本原因分析

### 2.1 Silent Death (P0)
`main()` に try/except がなく、`load_config()` や `run_scheduler()` で発生する
未ハンドル例外がプロセスを無言終了させていた。Windows の PowerShell stdout/stderr
リダイレクト + Python バッファリングにより、クラッシュログが消失。

### 2.2 シグナルハンドラの遅延登録 (P1)
`_install_signal_handlers()` が `run_scheduler()` 内部でのみ呼ばれており、
`main()` 起動中 (config ロード等) に SIGTERM を受けるとデフォルトハンドラで即死。

### 2.3 ループ保護の欠如 (P1)
`run_scheduler()` の while ループ内で `trigger.should_retrain()`,
`trigger.record_result()`, `_append_history()` が try/except なしで呼ばれており、
これらの例外がメインループを破壊。

### 2.4 訓練タイムアウトなし (P2)
`model.learn()` にタイムアウトがなく、無限ハングの可能性。

## 3. 修正内容

### 3.1 main() 全体の例外保護 + 自動リスタート
```python
# 495# main() 改修:
# - _install_signal_handlers() を起動直後に呼出 (load_config 前)
# - try/except で致命エラーをキャッチしログ出力
# - run_scheduler() 異常終了時の自動リスタート (上限 _MAX_AUTO_RESTARTS=5)
# - finally で logging.shutdown() + stream flush (Windows バッファ消失防止)
```

### 3.2 run_scheduler() ループの個別例外保護
```python
# trigger.should_retrain() → try/except で保護、失敗時は次回 check まで待機
# trigger.record_result() → try/except で保護、ログ出力のみ
# _append_history() → try/except で保護、ログ出力のみ
```

### 3.3 訓練タイムアウト (threading ベース)
```python
# _TRAINING_TIMEOUT_SEC = 3600 (1時間)
# threading.Thread(daemon=True) で model.learn() を実行
# join(timeout=...) でタイムアウト検出 → TimeoutError raise
# Windows 環境対応 (SIGALRM 未対応のため threading ベース)
```

## 4. 変更ファイル

| ファイル | 変更概要 |
|---------|---------|
| `scripts/v460/ml/sac_retrain_scheduler.py` | main() try/except + 自動リスタート, シグナルハンドラ早期登録, ループ保護, 訓練タイムアウト |
| `tests/unit/v460/test_sac_retrain_scheduler.py` | 495# クラッシュ耐性テスト 6 件追加 |

## 5. テスト結果

```
38 passed in 3.34s
```

新規テスト (6 件):
- `test_trigger_exception_does_not_kill_loop` — trigger.should_retrain() 例外でループ継続
- `test_record_result_exception_does_not_kill_loop` — record_result() 例外でループ継続
- `test_main_auto_restart_on_scheduler_crash` — run_scheduler クラッシュ後の自動リスタート
- `test_main_auto_restart_limit` — リスタート上限で打ち切り
- `test_main_fatal_config_error_logged` — config エラーで main() が例外終了しない
- `test_training_timeout_raises` — 訓練タイムアウト時に error ステータス返却

## 6. Before / After

| 項目 | Before | After |
|------|--------|-------|
| main() 例外保護 | なし — 未ハンドル例外で即死 | try/except/finally で全捕捉 |
| シグナルハンドラ | run_scheduler() 内のみ | main() 起動直後 |
| ループ保護 | retrain_once() のみ | trigger/record/history も個別保護 |
| 自動リスタート | なし | 最大 5 回 (backoff 60s×n) |
| 訓練タイムアウト | なし | 3600s (1 時間) |
| ログ flush | なし | logging.shutdown + stream flush |
