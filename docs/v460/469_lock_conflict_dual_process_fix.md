# 469# lock_conflict 二重プロセス防止

## 概要

watchdog 自動再起動と手動起動が競合し、2つの `run_fill_test` プロセスが
同時稼働する問題を修正。

## 問題

### 発生経緯
1. PID A が balance deadlock (JPY/BTC 両方不足) で `preflight_skip_exceeded` 停止
2. watchdog が NOT_RUNNING を検出し PID B を自動起動
3. 手動で PID C を起動 (preflight tolerance 引き上げ済み)
4. PID B と PID C が同時稼働 → lock_conflict

### 根本原因
- lock_manager はロックファイルの有無のみで二重起動を判定
- ロックファイル未作成 (起動途中) のプロセスは検出不可
- watchdog も WMI クエリ結果のみで判定し、heartbeat の鮮度を考慮しない

## 修正内容

### 1. lock_manager.py: プロセススキャンによる二重起動検出

`acquire()` の冒頭で `psutil.process_iter()` により全プロセスを走査。
自 PID 以外に `run_fill_test` を含む cmdline のプロセスが存在すれば
`LockConflictError` を発生させる。

```python
def _check_running_fill_test(self) -> None:
    my_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        if proc.info["pid"] == my_pid:
            continue
        cmdline = " ".join(proc.info.get("cmdline") or [])
        if "run_fill_test" in cmdline:
            raise LockConflictError(...)
```

### 2. fill_test_watchdog.ps1: heartbeat fresh 時の再起動抑制

NOT_RUNNING 判定後の stale lock 処理で、PID dead でも heartbeat が
stale 閾値未満の場合は再起動を1サイクル (5分) 延期。
WMI / Get-Process の一時的な検出失敗への耐性を向上。

```powershell
if (-not $heartbeatStale) {
    Write-WatchdogLog "CAUTION" "Lock PID dead but heartbeat fresh. Waiting one cycle."
    exit 1
}
```

## テスト

`tests/unit/v460/test_286_comprehensive_resolution.py` に2テスト追加:
- `test_check_running_fill_test_blocks_duplicate`: 別プロセス検出時に LockConflictError
- `test_check_running_fill_test_ignores_self`: 自 PID は無視

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/lock_manager.py` | `_check_running_fill_test()` 追加 |
| `ops/windows/fill_test_watchdog.ps1` | heartbeat fresh 時の再起動抑制 |
| `tests/unit/v460/test_286_comprehensive_resolution.py` | 2テスト追加 |
