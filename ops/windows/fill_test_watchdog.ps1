<#
.SYNOPSIS
    fill_test プロセス死活監視 + 自動再起動スクリプト (147# P2-A / 150# P2-B)
.DESCRIPTION
    fill_test.py の実行状態を監視し、停止を検出した場合にログ出力・通知を行う。
    -AutoRestart を指定すると、停止検出時に自動再起動を実行する。
    タスクスケジューラで 5 分間隔で実行することを想定。

    150# 設計: イベント契約 (schema)
      watchdog_alert  : 監視異常検出 (通知のみ、再起動なし)
      watchdog_restart: 自動再起動実行

    crash loop 防止: CooldownMinutes 内に MaxRestarts 回以上再起動済みなら抑止。
    TOCTOU 防止: restart.lock (短寿命) で再起動処理を排他。
.NOTES
    148# P0/P1 実装で fill_test_events.jsonl にイベントが記録されるようになったため、
    このスクリプトは主に「プロセス自体が存在しない」ケースの検出に使用。
#>

param(
    [string]$ResultsDir = "results\v460\fill_test",
    [switch]$Notify,
    [string]$WebhookUrl = $env:DISCORD_WEBHOOK,
    # 150# P2-B: 自動再起動パラメータ
    [switch]$AutoRestart,
    [int]$MaxRestarts = 3,
    [int]$CooldownMinutes = 60,
    [float]$Hours = 168,
    [string]$Config = "configs/v460/fill_test.yaml",
    [string]$Exchange = "coincheck",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = Resolve-Path (Join-Path $ScriptDir "..\..") | Select-Object -ExpandProperty Path
Set-Location $RootDir

# ======================================================================
# Helper: YAML から lock_stale_heartbeat_sec を読み取る (150# §9 #2)
# ======================================================================
function Get-StaleHeartbeatSec {
    param([string]$ConfigPath)
    $default = 300
    if (-not (Test-Path $ConfigPath)) { return $default }
    try {
        $content = Get-Content $ConfigPath -Raw
        if ($content -match 'lock_stale_heartbeat_sec:\s*([\d.]+)') {
            return [int][double]$Matches[1]
        }
    } catch { }
    return $default
}

# ======================================================================
# Helper: events.jsonl にイベント記録
# ======================================================================
function Write-EventLog {
    param(
        [string]$Event,
        [string]$Reason = $null,
        [hashtable]$Details = @{}
    )
    $eventsPath = Join-Path $ResultsDir "fill_test_events.jsonl"
    if (-not (Test-Path (Split-Path $eventsPath))) {
        New-Item -ItemType Directory -Path (Split-Path $eventsPath) -Force | Out-Null
    }
    $record = @{
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        event     = $Event
        run_id    = $null
        git_sha   = $null
        pid       = $PID
        reason    = $Reason
        details   = $Details
    } | ConvertTo-Json -Compress
    Add-Content -Path $eventsPath -Value $record
}

# ======================================================================
# Helper: watchdog ログ出力
# ======================================================================
function Write-WatchdogLog {
    param([string]$Status, [string]$Message)
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $logLine = "$timestamp $Status $Message"
    Write-Host $logLine

    $logDir = Join-Path $ResultsDir "logs"
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    $logPath = Join-Path $logDir "watchdog.log"
    Add-Content -Path $logPath -Value $logLine
}

# ======================================================================
# Helper: Discord 通知
# ======================================================================
function Send-DiscordNotify {
    param([string]$Message)
    if ($Notify -and $WebhookUrl) {
        $payload = @{
            content = $Message
        } | ConvertTo-Json
        try {
            Invoke-RestMethod -Uri $WebhookUrl -Method Post -Body $payload -ContentType "application/json"
            Write-Host "[watchdog] Discord notification sent"
        } catch {
            Write-Warning "[watchdog] Discord notification failed: $_"
        }
    }
}

# ======================================================================
# Helper: crash loop 判定 (150# §3.5)
# ======================================================================
function Test-CrashLoop {
    $eventsPath = Join-Path $ResultsDir "fill_test_events.jsonl"
    if (-not (Test-Path $eventsPath)) { return $false }
    $cutoff = (Get-Date).AddMinutes(-$CooldownMinutes).ToUniversalTime()
    $count = 0
    try {
        Get-Content $eventsPath | ForEach-Object {
            try {
                $evt = $_ | ConvertFrom-Json
                if ($evt.event -eq "watchdog_restart") {
                    $ts = [DateTime]::Parse($evt.timestamp).ToUniversalTime()
                    if ($ts -gt $cutoff) { $count++ }
                }
            } catch { }
        }
    } catch { }
    return ($count -ge $MaxRestarts)
}

# ======================================================================
# Helper: events.jsonl から最新 start イベントの起動パラメータを復元 (150# §3.4)
# ======================================================================
function Get-LastStartArgs {
    $eventsPath = Join-Path $ResultsDir "fill_test_events.jsonl"
    if (-not (Test-Path $eventsPath)) { return $null }
    $lastStart = $null
    try {
        Get-Content $eventsPath | ForEach-Object {
            try {
                $evt = $_ | ConvertFrom-Json
                if ($evt.event -eq "start") { $lastStart = $evt }
            } catch { }
        }
    } catch { }
    if ($lastStart -and $lastStart.details -and $lastStart.details.args) {
        return $lastStart.details.args
    }
    return $null
}

# ======================================================================
# 1. restart.lock 取得 (TOCTOU 防止, 150# §3.3)
# ======================================================================
$restartLockPath = Join-Path $ResultsDir "restart.lock"
$restartLockAcquired = $false

if ($AutoRestart) {
    if (Test-Path $restartLockPath) {
        # 120秒以上前の restart.lock は stale とみなす (360# OPS-4)
        $lockAge = ((Get-Date) - (Get-Item $restartLockPath).LastWriteTime).TotalSeconds
        if ($lockAge -gt 120) {
            Remove-Item $restartLockPath -Force -ErrorAction SilentlyContinue
            Write-Host "[watchdog] stale restart.lock removed (${lockAge}s old)"
        } else {
            Write-WatchdogLog "SKIP" "[watchdog] restart.lock held by another watchdog instance (${lockAge}s old)"
            exit 0
        }
    }
    # 取得
    try {
        [System.IO.File]::WriteAllText($restartLockPath, "$PID|$((Get-Date).ToString('o'))")
        $restartLockAcquired = $true
    } catch {
        Write-WatchdogLog "SKIP" "[watchdog] restart.lock acquisition failed: $_"
        exit 0
    }
}

try {

# ======================================================================
# 2. プロセス存在確認
# ======================================================================
$fillTestProcess = $null
$status = "UNKNOWN"
$message = ""
try {
    # WMI で CommandLine を含めてプロセス検索
    $fillTestProcess = Get-WmiObject Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*run_fill_test*" } |
        Select-Object -First 1

    if ($null -eq $fillTestProcess) {
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
# 3. lock ファイル確認 (heartbeat 鮮度)
# フォーマット: PID|created_ts|run_id|heartbeat_ts
# ======================================================================
$lockPath = Join-Path $ResultsDir "fill_test.lock"
$heartbeatStale = $false
$heartbeatAge = $null
$lockPidAlive = $false
$staleHeartbeatSec = Get-StaleHeartbeatSec (Join-Path $RootDir $Config)

if (Test-Path $lockPath) {
    try {
        $lockContent = Get-Content $lockPath -Raw
        $parts = $lockContent.Trim().Split("|")
        if ($parts.Count -ge 4) {
            $lockPid = [int]$parts[0]
            $heartbeatEpoch = [long]$parts[3]
            $heartbeat = [DateTimeOffset]::FromUnixTimeSeconds($heartbeatEpoch).LocalDateTime
            $heartbeatAge = ((Get-Date) - $heartbeat).TotalSeconds

            # lock の PID が生きているか確認
            $lockPidAlive = $null -ne (Get-Process -Id $lockPid -ErrorAction SilentlyContinue)

            # 150# §9 #2: stale 閾値を YAML から読む
            if ($heartbeatAge -gt $staleHeartbeatSec) {
                $heartbeatStale = $true
                $message += " | heartbeat STALE ($([int]$heartbeatAge)s ago, threshold=${staleHeartbeatSec}s)"
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
# 4. 最新 fill_record 確認
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
        } catch { }
    }
}

# ======================================================================
# 5. ステータスログ出力
# ======================================================================
Write-WatchdogLog $status $message

# ======================================================================
# 6. RUNNING の場合 → 正常終了
# ======================================================================
if ($status -eq "RUNNING") {
    exit 0
}

# ======================================================================
# 7. NOT_RUNNING / UNKNOWN → アラート
# ======================================================================
$shouldAlert = ($status -eq "NOT_RUNNING") -or ($status -eq "UNKNOWN") -or $heartbeatStale

if ($shouldAlert) {
    Write-Warning $message

    # watchdog_alert イベント記録
    Write-EventLog -Event "watchdog_alert" -Reason $status -Details @{
        heartbeat_stale   = $heartbeatStale
        heartbeat_age_sec = $heartbeatAge
        last_fill_age_min = $lastFillAge
    }

    Send-DiscordNotify "⚠️ **fill_test watchdog alert**`n$message"
}

# ======================================================================
# 8. AutoRestart: 自動再起動 (150# P2-B)
# ======================================================================
if (-not $AutoRestart) {
    if ($shouldAlert) { exit 1 }
    exit 0
}

if ($status -eq "UNKNOWN") {
    # §11 #1 (379# fix): UNKNOWN でも lock PID が死亡 & heartbeat stale なら NOT_RUNNING として扱う
    # 元のロジックは WMI "Call cancelled" 時に再起動を永久スキップしてしまうバグがあった
    if ($heartbeatStale -and -not $lockPidAlive) {
        Write-WatchdogLog "ESCALATE" "[watchdog] status=UNKNOWN but lock PID dead + heartbeat STALE → treating as NOT_RUNNING"
        $status = "NOT_RUNNING"
    } else {
        Write-WatchdogLog "SKIP" "[watchdog] status=UNKNOWN, restart skipped (state unconfirmed)"
        exit 1
    }
}
if ($status -ne "NOT_RUNNING") {
    Write-WatchdogLog "SKIP" "[watchdog] status=$status, AutoRestart skipped (not NOT_RUNNING)"
    exit 1
}

# 8a. crash loop 判定 (150# §3.5)
if (Test-CrashLoop) {
    $loopMsg = "[watchdog] crash loop detected: $MaxRestarts+ restarts in ${CooldownMinutes}min. Restart suppressed."
    Write-WatchdogLog "CRASH_LOOP" $loopMsg
    Write-EventLog -Event "watchdog_alert" -Reason "crash_loop" -Details @{
        max_restarts     = $MaxRestarts
        cooldown_minutes = $CooldownMinutes
    }
    Send-DiscordNotify "🔴 **fill_test crash loop detected**`n$loopMsg"
    exit 2
}

# 8b. stale lock 処理 (150# §3.3 step 4)
if (Test-Path $lockPath) {
    if ($lockPidAlive) {
        # WMI で見つからないが PID は alive → 矛盾 (検出漏れ?)
        Write-WatchdogLog "CONFLICT" "[watchdog] lock PID alive but WMI found nothing. Skipping restart."
        exit 1
    }
    # 469#: PID dead だが heartbeat が fresh な場合、1 サイクル待機
    # WMI/Get-Process が一時的に失敗するケースに対応
    if (-not $heartbeatStale) {
        Write-WatchdogLog "CAUTION" "[watchdog] Lock PID dead but heartbeat fresh ($([int]$heartbeatAge)s ago). Waiting one cycle before restart."
        exit 1
    }
    # PID dead AND heartbeat stale → stale lock → 削除
    Write-Host "[watchdog] Removing stale lock (PID not alive, heartbeat stale)"
    Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
}

# 8c. 起動パラメータ決定 (明示パラメータ優先、なければ最新 start event から復元)
$effectiveHours = $Hours
$effectiveConfig = $Config
$effectiveExchange = $Exchange
$effectiveDryRun = $DryRun.IsPresent

# 明示されていない場合、最新 start event から復元を試みる (§11 #4: hours/config も復元対象)
$lastArgs = Get-LastStartArgs
if ($lastArgs) {
    Write-Host "[watchdog] Last start event args found: $($lastArgs | ConvertTo-Json -Compress)"
    # デフォルト値の場合のみ start event から復元 (明示パラメータ優先)
    if ($lastArgs.hours -and $Hours -eq 168) {
        $effectiveHours = $lastArgs.hours
    }
    if ($lastArgs.config -and $Config -eq "configs/v460/fill_test.yaml") {
        $effectiveConfig = $lastArgs.config
    }
    if ($lastArgs.exchange -and $Exchange -eq "coincheck") {
        $effectiveExchange = $lastArgs.exchange
    }
    if ($lastArgs.dry_run -eq $true -and -not $DryRun.IsPresent) {
        $effectiveDryRun = $true
    }
}

# 8d. 再起動実行 (150# §3.4)
$pythonExe = Join-Path $RootDir ".venv\Scripts\python.exe"
$arguments = @(
    "-m", "scripts.v460.run_fill_test",
    "--hours", $effectiveHours,
    "--config", $effectiveConfig,
    "--exchange", $effectiveExchange
)
if ($effectiveDryRun) { $arguments += "--dry-run" }

Write-Host "[watchdog] Restarting fill_test: $pythonExe $($arguments -join ' ')"

# stdout/stderr リダイレクト先
$logDir = Join-Path $ResultsDir "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
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

# OPS-6: 起動確認 (360# §6.3)
$lockWaitMax = 30
$lockWaitInterval = 2
$lockWaitElapsed = 0
while ($lockWaitElapsed -lt $lockWaitMax) {
    if (Test-Path $lockPath) {
        Write-WatchdogLog "INFO" "[watchdog] fill_test.lock detected after ${lockWaitElapsed}s — startup confirmed"
        break
    }
    Start-Sleep -Seconds $lockWaitInterval
    $lockWaitElapsed += $lockWaitInterval
}
if ($lockWaitElapsed -ge $lockWaitMax) {
    Write-WatchdogLog "WARN" "[watchdog] fill_test.lock not found after ${lockWaitMax}s — startup may have failed"
}

# watchdog_restart イベント記録
Write-EventLog -Event "watchdog_restart" -Reason "auto_restart" -Details @{
    new_pid     = $newPid
    hours       = $effectiveHours
    config      = $effectiveConfig
    exchange    = $effectiveExchange
    dry_run     = $effectiveDryRun
    from_event  = if ($lastArgs) { $true } else { $false }
}

$restartMsg = "[watchdog] fill_test restarted: PID=$newPid, hours=$effectiveHours, exchange=$effectiveExchange"
Write-WatchdogLog "RESTARTED" $restartMsg
Send-DiscordNotify "🟢 **fill_test auto-restarted**`n$restartMsg"

exit 0

} finally {
    # restart.lock 解放
    if ($restartLockAcquired -and (Test-Path $restartLockPath)) {
        Remove-Item $restartLockPath -Force -ErrorAction SilentlyContinue
    }
}
