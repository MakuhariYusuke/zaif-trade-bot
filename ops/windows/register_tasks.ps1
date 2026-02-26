<#
.SYNOPSIS
    タスクスケジューラに 3 つの定期タスクを登録するスクリプト。
.DESCRIPTION
    168# §4.3 の運用自動化:
      1. ZTB-Watchdog       : fill_test 死活監視 (5分間隔)
      2. ZTB-DailyHealth    : 日次ヘルスチェック (毎日 15:00 JST = 06:00 UTC)
      3. ZTB-WeeklyAnalysis : 週次分析バッチ (毎週月曜 09:00 JST = 00:00 UTC)

    管理者権限で実行すること。
.EXAMPLE
    .\ops\windows\register_tasks.ps1
    .\ops\windows\register_tasks.ps1 -Unregister   # 登録解除
#>

param(
    [switch]$Unregister,
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
)

$ErrorActionPreference = "Stop"

$tasks = @(
    @{
        Name        = "ZTB-Watchdog"
        Description = "fill_test process watchdog (168#)"
        Script      = "ops\windows\fill_test_watchdog.ps1"
        Args        = "-Notify -AutoRestart"
        Trigger     = "New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)"
    },
    @{
        Name        = "ZTB-DailyHealth"
        Description = "Daily health check + KPI batch (168#)"
        Script      = "ops\windows\daily_health_check.ps1"
        Args        = ""
        Trigger     = "New-ScheduledTaskTrigger -Daily -At '15:00'"
    },
    @{
        Name        = "ZTB-WeeklyAnalysis"
        Description = "Weekly hindsight + Monte Carlo analysis (168#)"
        Script      = "ops\windows\weekly_analysis.ps1"
        Args        = ""
        Trigger     = "New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At '09:00'"
    }
)

foreach ($t in $tasks) {
    if ($Unregister) {
        if (Get-ScheduledTask -TaskName $t.Name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $t.Name -Confirm:$false
            Write-Host "Unregistered: $($t.Name)" -ForegroundColor Yellow
        } else {
            Write-Host "Not found: $($t.Name)" -ForegroundColor DarkGray
        }
        continue
    }

    $scriptPath = Join-Path $ProjectRoot $t.Script
    if (-not (Test-Path $scriptPath)) {
        Write-Warning "Script not found: $scriptPath — skipping $($t.Name)"
        continue
    }

    $argStr = "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
    if ($t.Args) { $argStr += " $($t.Args)" }

    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument $argStr `
        -WorkingDirectory $ProjectRoot

    $trigger = Invoke-Expression $t.Trigger

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2)

    # Unregister existing if present
    if (Get-ScheduledTask -TaskName $t.Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $t.Name -Confirm:$false
        Write-Host "  Replaced existing: $($t.Name)" -ForegroundColor DarkGray
    }

    Register-ScheduledTask `
        -TaskName $t.Name `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description $t.Description | Out-Null

    Write-Host "Registered: $($t.Name)" -ForegroundColor Green
    Write-Host "  Script : $scriptPath" -ForegroundColor DarkGray
    Write-Host "  Trigger: $($t.Trigger)" -ForegroundColor DarkGray
}

if (-not $Unregister) {
    Write-Host "`n=== Registered Tasks ===" -ForegroundColor Cyan
    Get-ScheduledTask -TaskName "ZTB-*" | Format-Table TaskName, State, @{N="NextRun";E={$_.Triggers[0].StartBoundary}} -AutoSize
}
