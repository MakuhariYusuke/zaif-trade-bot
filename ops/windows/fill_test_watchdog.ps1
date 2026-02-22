<#
.SYNOPSIS
    fill_test プロセス死活監視スクリプト (147# P2-A)
.DESCRIPTION
    fill_test.py の実行状態を監視し、停止を検出した場合にログ出力・通知を行う。
    タスクスケジューラで 5-10 分間隔で実行することを想定。
.NOTES
    148# P0/P1 実装で fill_test_events.jsonl にイベントが記録されるようになったため、
    このスクリプトは主に「プロセス自体が存在しない」ケースの検出に使用。
#>

param(
    [string]$ResultsDir = "results\v460\fill_test",
    [switch]$Notify,
    [string]$WebhookUrl = $env:DISCORD_WEBHOOK
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = Resolve-Path (Join-Path $ScriptDir "..\..") | Select-Object -ExpandProperty Path
Set-Location $RootDir

# ======================================================================
# 1. プロセス存在確認
# ======================================================================
$fillTestProcess = $null
try {
    # WMI で CommandLine を含めてプロセス検索
    $fillTestProcess = Get-WmiObject Win32_Process -Filter "Name='python.exe'" | 
        Where-Object { $_.CommandLine -like "*run_fill_test*" } |
        Select-Object -First 1

    if ($null -eq $fillTestProcess) {
        # プロセスが存在しない
        $status = "NOT_RUNNING"
        $message = "[watchdog] fill_test プロセスが見つかりません"
    } else {
        $status = "RUNNING"
        $procId = $fillTestProcess.ProcessId
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($proc) {
            $startTime = $proc.StartTime
            $uptime = (Get-Date) - $startTime
            $message = "[watchdog] fill_test 稼働中: PID=$procId, uptime=$($uptime.ToString('d\.hh\:mm\:ss'))"
        } else {
            $message = "[watchdog] fill_test 稼働中: PID=$procId"
        }
    }
} catch {
    $status = "UNKNOWN"
    $message = "[watchdog] プロセス確認エラー: $_"
}

# ======================================================================
# 2. lock ファイル確認 (heartbeat 鮮度)
# フォーマット: PID|created_ts|run_id|heartbeat_ts
# ======================================================================
$lockPath = Join-Path $ResultsDir "fill_test.lock"
$heartbeatStale = $false
$heartbeatAge = $null

if (Test-Path $lockPath) {
    try {
        $lockContent = Get-Content $lockPath -Raw
        $parts = $lockContent.Trim().Split("|")
        if ($parts.Count -ge 4) {
            $heartbeatEpoch = [long]$parts[3]
            $heartbeat = [DateTimeOffset]::FromUnixTimeSeconds($heartbeatEpoch).LocalDateTime
            $heartbeatAge = ((Get-Date) - $heartbeat).TotalSeconds
            
            # 148# P0: 60s 周期更新、300s で stale 判定
            if ($heartbeatAge -gt 300) {
                $heartbeatStale = $true
                $message += " | heartbeat STALE ($([int]$heartbeatAge)s ago)"
            } else {
                $message += " | heartbeat OK ($([int]$heartbeatAge)s ago)"
            }
        } else {
            $message += " | lock format unexpected"
        }
    } catch {
        $message += " | lock parse error: $_"
    }
} elseif ($status -eq "RUNNING") {
    $message += " | lock file missing (unexpected)"
}

# ======================================================================
# 3. 最新 fill_record 確認
# ======================================================================
$today = (Get-Date).ToString("yyyyMMdd")
$todayRecordPath = Join-Path $ResultsDir "fill_records_$today.jsonl"
$lastFillAge = $null

if (Test-Path $todayRecordPath) {
    $lastLine = Get-Content $todayRecordPath -Tail 1 -ErrorAction SilentlyContinue
    if ($lastLine) {
        try {
            $lastRecord = $lastLine | ConvertFrom-Json
            $lastFillTime = [DateTime]::Parse($lastRecord.timestamp)
            $lastFillAge = ((Get-Date) - $lastFillTime).TotalMinutes
            if ($lastFillAge -gt 30) {
                $message += " | last fill $([int]$lastFillAge)min ago (stale?)"
            } else {
                $message += " | last fill $([int]$lastFillAge)min ago"
            }
        } catch {
            # JSON parse error - ignore
        }
    }
}

# ======================================================================
# 4. ログ出力
# ======================================================================
$timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
$logLine = "$timestamp $status $message"
Write-Host $logLine

$logDir = Join-Path $ResultsDir "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
$logPath = Join-Path $logDir "watchdog.log"
Add-Content -Path $logPath -Value $logLine

# ======================================================================
# 5. アラート条件判定 + 通知
# ======================================================================
$shouldAlert = ($status -eq "NOT_RUNNING") -or $heartbeatStale

if ($shouldAlert) {
    Write-Warning $message
    
    # fill_test_events.jsonl にイベント記録
    $eventsPath = Join-Path $ResultsDir "fill_test_events.jsonl"
    $event = @{
        timestamp = (Get-Date).ToString("o")
        event = "watchdog_alert"
        run_id = $null
        git_sha = $null
        reason = $status
        details = @{
            heartbeat_stale = $heartbeatStale
            heartbeat_age_sec = $heartbeatAge
            last_fill_age_min = $lastFillAge
        }
    } | ConvertTo-Json -Compress
    Add-Content -Path $eventsPath -Value $event
    
    # Discord webhook 通知 (オプション)
    if ($Notify -and $WebhookUrl) {
        $payload = @{
            content = "⚠️ **fill_test watchdog alert**`n$message"
        } | ConvertTo-Json
        try {
            Invoke-RestMethod -Uri $WebhookUrl -Method Post -Body $payload -ContentType "application/json"
            Write-Host "[watchdog] Discord notification sent"
        } catch {
            Write-Warning "[watchdog] Discord notification failed: $_"
        }
    }
    
    exit 1
}

exit 0
