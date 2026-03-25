<#
.SYNOPSIS
    286#/458# fill_test ホットスワップ再起動スクリプト
.DESCRIPTION
    現在稼働中の fill_test を安全に停止し、最新コードで再起動する。
    IDE が落ちても独立動作するよう設計。

    手順:
      0. Pre-flight: config 存在確認 + git SHA 表示
      1. 現在の lock ファイルから稼働 PID を特定
      2. 全 run_fill_test プロセスを停止 (lock PID + 孤児プロセス)
      3. lock ファイルの stale 化を確認・除去
      4. 新プロセスを起動 (Start-Process, バックグラウンド)
      5. 起動確認 (lock ファイル + heartbeat 更新確認)

    使用例:
      powershell -ExecutionPolicy Bypass -File ops\windows\hot_swap_restart.ps1
      powershell -ExecutionPolicy Bypass -File ops\windows\hot_swap_restart.ps1 -Hours 24 -DryRun
.NOTES
    286# 2026-03-06 作成
    458# 2026-03-17 改善: CIM置換, 孤児プロセスkill, taskkillエラー処理, pre-flight, heartbeat検証
#>

param(
    [string]$ResultsDir = "results\v460\fill_test",
    [string]$Config = "configs/v460/fill_test.yaml",
    [string]$Exchange = "coincheck",
    [float]$Hours = 24.0,
    [switch]$DryRun,
    [int]$GracefulWaitSec = 30,
    [int]$StartupConfirmSec = 60
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = Resolve-Path (Join-Path $ScriptDir "..\..") | Select-Object -ExpandProperty Path
Set-Location $RootDir

$timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
$scriptStart = Get-Date
$gitSha = (git rev-parse --short HEAD 2>$null)
if (-not $gitSha) { $gitSha = "unknown" }
Write-Host "=========================================="
Write-Host " fill_test hot-swap restart (286#/458#)"
Write-Host " $timestamp  SHA: $gitSha"
Write-Host "=========================================="

# ======================================================================
# Step 0: Pre-flight チェック
# ======================================================================
$logDir = Join-Path $ResultsDir "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
$logPath = Join-Path $logDir "hot_swap.log"

function Log {
    param([string]$Level, [string]$Msg)
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "$ts [$Level] $Msg"
    Write-Host $line
    Add-Content -Path $logPath -Value $line
}

# Pre-flight: config 存在確認
if (-not (Test-Path $Config)) {
    Log "ERROR" "Config ファイルが見つかりません: $Config"
    exit 1
}
$pythonExe = Join-Path $RootDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Log "ERROR" "Python 実行ファイルが見つかりません: $pythonExe"
    exit 1
}
# NOTE: venv launcher はトランポリンとして子プロセスを spawn する。
# 起動後に launcher (親) を kill すると子 (実体) も死ぬため、
# post-startup のorphan cleanupは行わない。launcher は ~2MB でharmless。
Log "INFO" "Pre-flight OK: config=$Config, python=$pythonExe, sha=$gitSha"

# ======================================================================
# Step 1: 現在の稼働プロセスを特定
# ======================================================================
Log "INFO" "Step 1: 稼働プロセス特定..."

$lockPath = Join-Path $ResultsDir "fill_test.lock"
$oldPid = $null

if (Test-Path $lockPath) {
    $lockContent = Get-Content $lockPath -Raw
    $parts = $lockContent.Trim().Split("|")
    $oldPid = [int]$parts[0]
    $heartbeatEpoch = [long]$parts[3]
    $heartbeat = [DateTimeOffset]::FromUnixTimeSeconds($heartbeatEpoch).LocalDateTime
    $age = [int]((Get-Date) - $heartbeat).TotalSeconds
    Log "INFO" "  Lock PID: $oldPid, heartbeat: ${age}s ago"

    $procAlive = $null -ne (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)
    if (-not $procAlive) {
        Log "WARN" "  PID $oldPid は既に終了済。stale lock を除去します。"
        Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
        $oldPid = $null
    }
} else {
    Log "INFO" "  lock ファイルなし。プロセスは稼働していない可能性。"
}

# WMI でも確認 (458# Get-CimInstance に置換)
$wmiProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*run_fill_test*" }
if ($wmiProcs) {
    foreach ($p in $wmiProcs) {
        $cmdSnippet = $p.CommandLine.Substring(0, [Math]::Min(120, $p.CommandLine.Length))
        Log "INFO" "  CIM 発見: PID=$($p.ProcessId) CMD=$cmdSnippet"
    }
}

# ======================================================================
# Step 2: 旧プロセスを安全に停止
# ======================================================================
if ($oldPid) {
    Log "INFO" "Step 2: PID $oldPid を graceful 停止 (${GracefulWaitSec}s 待機)..."

    $proc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
    if ($proc) {
        # 458# taskkill エラー処理改善: native command の stderr を
        # $ErrorActionPreference="Stop" 下でも安全に捕捉
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $tkOutput = & taskkill /PID $oldPid 2>&1 | Out-String
        $ErrorActionPreference = $prevEAP
        if ($LASTEXITCODE -eq 0) {
            Log "INFO" "  taskkill 送信完了。graceful shutdown 待機中..."
        } else {
            Log "WARN" "  taskkill 失敗 (exit=$LASTEXITCODE): $($tkOutput.Trim())"
        }

        # 待機
        $waited = 0
        while ($waited -lt $GracefulWaitSec) {
            Start-Sleep -Seconds 2
            $waited += 2
            $still = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
            if (-not $still) {
                Log "INFO" "  PID $oldPid が graceful に終了しました (${waited}s)"
                break
            }
            if ($waited % 10 -eq 0) {
                Log "INFO" "  待機中... ${waited}s / ${GracefulWaitSec}s"
            }
        }

        # まだ生きていたら強制終了
        $still = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
        if ($still) {
            Log "WARN" "  graceful 期限切れ。PID $oldPid を強制終了します..."
            Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 3
            $still = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
            if ($still) {
                Log "ERROR" "  PID $oldPid を強制終了できませんでした。中断します。"
                exit 1
            }
            Log "INFO" "  PID $oldPid を強制終了しました。"
        }
    }
} else {
    Log "INFO" "Step 2: 停止対象プロセスなし。スキップ。"
}

# 458# 孤児プロセスの掃除: lock PID 以外の run_fill_test プロセスも停止
$orphanProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*run_fill_test*" -and $_.ProcessId -ne $oldPid }
if ($orphanProcs) {
    Log "WARN" "孤児 run_fill_test プロセスを発見。停止します..."
    foreach ($op in $orphanProcs) {
        try {
            Stop-Process -Id $op.ProcessId -Force -ErrorAction SilentlyContinue
            Log "INFO" "  孤児 PID $($op.ProcessId) 停止"
        } catch {
            Log "WARN" "  孤児 PID $($op.ProcessId) 停止失敗: $_"
        }
    }
    Start-Sleep -Seconds 2
}

# retrain_scheduler も停止 (同じコードベースを使うため)
# 491# fix: graceful shutdown — まず SIGTERM 相当の Ctrl+C → 待機 → Force
$retrainProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*retrain_scheduler*" }
if ($retrainProcs) {
    Log "INFO" "  retrain_scheduler を graceful 停止します..."
    foreach ($rp in $retrainProcs) {
        try {
            # まず通常停止を試行 (signal handler が _shutdown_event.set() する)
            Stop-Process -Id $rp.ProcessId -ErrorAction SilentlyContinue
        } catch {
            # 無視して Force へ
        }
    }
    # 15秒待機して graceful shutdown を待つ
    $waited = 0
    while ($waited -lt 15) {
        Start-Sleep -Seconds 3
        $waited += 3
        $remaining = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -like "*retrain_scheduler*" }
        if (-not $remaining) { break }
    }
    # まだ残っていれば Force kill
    $remaining = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*retrain_scheduler*" }
    if ($remaining) {
        Log "WARN" "  retrain_scheduler が graceful 停止しなかったため強制停止..."
        foreach ($rp in $remaining) {
            Stop-Process -Id $rp.ProcessId -Force -ErrorAction SilentlyContinue
            Log "INFO" "  retrain_scheduler PID $($rp.ProcessId) 強制停止"
        }
        Start-Sleep -Seconds 2
    } else {
        Log "INFO" "  retrain_scheduler graceful 停止完了"
    }
}

# ======================================================================
# Step 3: stale lock 除去
# ======================================================================
Log "INFO" "Step 3: stale lock 確認..."

if (Test-Path $lockPath) {
    Log "INFO" "  stale lock を除去します。"
    Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
}

# .os_lock も除去 (286# portalocker 用)
$osLockPath = Join-Path $ResultsDir "fill_test.os_lock"
if (Test-Path $osLockPath) {
    Remove-Item $osLockPath -Force -ErrorAction SilentlyContinue
    Log "INFO" "  .os_lock も除去しました。"
}

# 474# retrain_scheduler.lock も除去 (多重起動防止用)
$retrainLockPath = Join-Path $RootDir "logs\retrain_scheduler.lock"
if (Test-Path $retrainLockPath) {
    Remove-Item $retrainLockPath -Force -ErrorAction SilentlyContinue
    Log "INFO" "  retrain_scheduler.lock も除去しました。"
}

# ======================================================================
# Step 4: 新プロセス起動
# ======================================================================
Log "INFO" "Step 4: 新プロセス起動..."

$arguments = @(
    "-m", "scripts.v460.run_fill_test",
    "--hours", $Hours,
    "--config", $Config,
    "--exchange", $Exchange
)
if ($DryRun) { $arguments += "--dry-run" }

Log "INFO" "  CMD: $pythonExe $($arguments -join ' ')"

$stdoutLog = Join-Path $logDir "fill_test_stdout.log"
$stderrLog = Join-Path $logDir "fill_test_stderr.log"

$newProc = Start-Process -FilePath $pythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $RootDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

$newPid = $newProc.Id
Log "INFO" "  新プロセス起動: PID=$newPid"

# ======================================================================
# Step 5: 起動確認
# ======================================================================
Log "INFO" "Step 5: 起動確認 (最大 ${StartupConfirmSec}s)..."

$confirmed = $false
$waited = 0
while ($waited -lt $StartupConfirmSec) {
    Start-Sleep -Seconds 5
    $waited += 5

    # プロセス生存確認 (親 or 子 PID)
    $alive = ($null -ne (Get-Process -Id $newPid -ErrorAction SilentlyContinue))
    if (-not $alive) {
        # 親PIDが dead でも lock ファイルの子PIDが生きている場合がある
        if (Test-Path $lockPath) {
            $checkLock = Get-Content $lockPath -Raw -ErrorAction SilentlyContinue
            if ($checkLock) {
                $checkPid = $checkLock.Trim().Split("|")[0]
                $alive = $null -ne (Get-Process -Id $checkPid -ErrorAction SilentlyContinue)
            }
        }
    }
    if (-not $alive) {
        Log "ERROR" "  新プロセス PID=$newPid が起動直後に終了しました！"
        Log "ERROR" "  stderr ログを確認: $stderrLog"
        if (Test-Path $stderrLog) {
            $errContent = Get-Content $stderrLog -Tail 20 -ErrorAction SilentlyContinue
            if ($errContent) {
                Log "ERROR" "  === stderr tail ==="
                $errContent | ForEach-Object { Log "ERROR" "  $_" }
            }
        }
        exit 1
    }

    # lock ファイル確認
    # NOTE: Start-Process の PID とlock の PID は異なる場合がある
    # (Python ランチャーが子プロセスを spawn するため)
    # lock PID が alive であれば成功とみなす
    if (Test-Path $lockPath) {
        $newLock = Get-Content $lockPath -Raw
        $newParts = $newLock.Trim().Split("|")
        $newLockPid = $newParts[0]
        $lockPidAlive = $null -ne (Get-Process -Id $newLockPid -ErrorAction SilentlyContinue)
        Log "INFO" "  lock 確認: PID=$newLockPid (alive=$lockPidAlive) [${waited}s]"
        if ($lockPidAlive) {
            $newPid = [int]$newLockPid  # 実際の PID に更新

            # 458# heartbeat 更新確認: lock が作られただけでなく、実際に動作しているか確認
            if ($newParts.Length -ge 4) {
                $hbEpoch = [long]$newParts[3]
                $hbAge = [int]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $hbEpoch)
                Log "INFO" "  heartbeat age: ${hbAge}s"
                if ($hbAge -gt 120) {
                    Log "WARN" "  heartbeat が ${hbAge}s 前。プロセスがハングしている可能性。"
                }
            }
            $confirmed = $true
            break
        }
    } else {
        Log "INFO" "  lock 未作成... (${waited}s)"
    }
}

if ($confirmed) {
    $elapsed = [int]((Get-Date) - $scriptStart).TotalSeconds
    Log "INFO" "=========================================="
    Log "INFO" " Hot-swap 完了！"
    Log "INFO" " 新 PID: $newPid"
    Log "INFO" " SHA: $gitSha"
    Log "INFO" " Config: $Config"
    Log "INFO" " Exchange: $Exchange"
    Log "INFO" " Hours: $Hours"
    Log "INFO" " 所要時間: ${elapsed}s"
    Log "INFO" "=========================================="

    # 426# P1: retrain_scheduler の自動再起動
    # 491# fix: fill_test config ではなく SAC 訓練用 config を使用
    $retrainScript = Join-Path $PSScriptRoot "retrain_scheduler.ps1"
    $retrainConfig = "configs/v460/experiments/g2_sac_train.yaml"
    if (Test-Path $retrainScript) {
        Log "INFO" "retrain_scheduler を再起動します... (config: $retrainConfig)"
        & $retrainScript -Action start -Config $retrainConfig
    }
} else {
    Log "WARN" "  ${StartupConfirmSec}s 以内に lock 確認できず。プロセスは起動中ですが要確認。"
    Log "WARN" "  PID $newPid  alive=$(($null -ne (Get-Process -Id $newPid -ErrorAction SilentlyContinue)))"
}

# events.jsonl に記録
$eventsPath = Join-Path $ResultsDir "fill_test_events.jsonl"
$record = @{
    timestamp = (Get-Date).ToUniversalTime().ToString("o")
    event     = "hot_swap_restart"
    run_id    = $null
    git_sha   = $gitSha
    pid       = $newPid
    reason    = "286# hot-swap deploy"
    details   = @{
        old_pid  = $oldPid
        new_pid  = $newPid
        hours    = $Hours
        config   = $Config
        exchange = $Exchange
        dry_run  = $DryRun.IsPresent
    }
} | ConvertTo-Json -Compress
Add-Content -Path $eventsPath -Value $record

exit 0
