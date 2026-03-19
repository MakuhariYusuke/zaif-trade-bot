<#
.SYNOPSIS
    SAC retrain scheduler の起動・停止・状態確認
.DESCRIPTION
    430# で発見された sac_retrain_scheduler.py を起動する ops スクリプト。
    hot_swap_restart.ps1 で retrain_scheduler が巻き込み停止されるため、
    fill_test 再起動後に本スクリプトで retrain_scheduler も再起動する。
.PARAMETER Action
    start | stop | status | restart
.PARAMETER Config
    YAML config パス（デフォルト: configs/v460/experiments/g2_sac_train.yaml）
.PARAMETER Seed
    シード上書き（省略時はconfig内の値）
.EXAMPLE
    .\retrain_scheduler.ps1 -Action start
    .\retrain_scheduler.ps1 -Action stop
    .\retrain_scheduler.ps1 -Action status
    .\retrain_scheduler.ps1 -Action restart
#>
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "status", "restart")]
    [string]$Action,

    [string]$Config = "configs/v460/experiments/g2_sac_train.yaml",
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $ProjectRoot

$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$LogFile = Join-Path $LogDir "retrain_scheduler.log"
$PidFile = Join-Path $LogDir "retrain_scheduler.pid"
$Python  = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Script  = "scripts/v460/ml/sac_retrain_scheduler.py"

function Log([string]$Level, [string]$Msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts [$Level] $Msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Get-RetrainProcess {
    Get-WmiObject Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*retrain_scheduler*" }
}

function Stop-Retrain {
    $procs = Get-RetrainProcess
    if (-not $procs) {
        Log "INFO" "retrain_scheduler は起動していません"
        return
    }
    foreach ($p in $procs) {
        Log "INFO" "retrain_scheduler PID $($p.ProcessId) を停止します..."
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 3
    $remaining = Get-RetrainProcess
    if ($remaining) {
        Log "WARN" "一部プロセスが残存。強制停止を再試行..."
        foreach ($p in $remaining) {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
    Log "INFO" "retrain_scheduler 停止完了"
}

function Start-Retrain {
    $existing = Get-RetrainProcess
    if ($existing) {
        Log "WARN" "retrain_scheduler は既に起動中 (PID: $($existing[0].ProcessId))"
        return
    }

    if (-not (Test-Path $Python)) {
        Log "ERROR" "Python not found: $Python"
        return
    }
    if (-not (Test-Path (Join-Path $ProjectRoot $Config))) {
        Log "ERROR" "Config not found: $Config"
        return
    }

    $args = @($Script, "--config", $Config)
    if ($Seed -gt 0) {
        $args += @("--seed", $Seed.ToString())
    }

    $stdOut = Join-Path $LogDir "retrain_scheduler_stdout.log"
    $stdErr = Join-Path $LogDir "retrain_scheduler_stderr.log"

    # ztb パッケージ解決: Start-Process では sys.path[0] がスクリプト
    # ディレクトリになるため、プロジェクトルートを PYTHONPATH に追加
    $env:PYTHONPATH = $ProjectRoot
    $proc = Start-Process -FilePath $Python `
        -ArgumentList $args `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdOut `
        -RedirectStandardError $stdErr `
        -PassThru

    if ($proc -and $proc.Id) {
        Set-Content -Path $PidFile -Value $proc.Id -Encoding UTF8
        Log "INFO" "retrain_scheduler 起動成功 (PID: $($proc.Id))"
        Log "INFO" "  Config: $Config"
        Log "INFO" "  stdout: $stdOut"
        Log "INFO" "  stderr: $stdErr"
        Log "INFO" "  Signal: cache/sidecar_signal.json"
    } else {
        Log "ERROR" "retrain_scheduler 起動失敗"
    }
}

function Show-Status {
    $procs = Get-RetrainProcess
    if ($procs) {
        foreach ($p in $procs) {
            Log "INFO" "RUNNING - PID: $($p.ProcessId)"
            Log "INFO" "  CommandLine: $($p.CommandLine)"
        }
    } else {
        Log "INFO" "NOT_RUNNING"
    }

    # Signal ファイルの状態
    $signalPath = Join-Path $ProjectRoot "cache\sidecar_signal.json"
    if (Test-Path $signalPath) {
        $mtime = (Get-Item $signalPath).LastWriteTime
        $age = (Get-Date) - $mtime
        $ageStr = "{0:hh\:mm\:ss}" -f $age
        Log "INFO" "  Signal: $signalPath (age: $ageStr)"
        if ($age.TotalSeconds -gt 7800) {
            Log "WARN" "  Signal is STALE (> 2h10m TTL)"
        }
    } else {
        Log "INFO" "  Signal: NOT_FOUND"
    }

    # History の最終行
    $historyPath = Join-Path $ProjectRoot "logs\sac_retrain_history.jsonl"
    if (Test-Path $historyPath) {
        $lastLine = Get-Content $historyPath | Select-Object -Last 1
        if ($lastLine) {
            Log "INFO" "  Last history: $lastLine"
        }
    }
}

# ---- Main ----
switch ($Action) {
    "start"   { Start-Retrain }
    "stop"    { Stop-Retrain }
    "status"  { Show-Status }
    "restart" { Stop-Retrain; Start-Sleep -Seconds 2; Start-Retrain }
}
