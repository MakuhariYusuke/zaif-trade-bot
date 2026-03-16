# 168# §4.3: 週次分析バッチ — hindsight_filter + PnL Monte Carlo
# タスクスケジューラで毎週月曜 09:00 JST (00:00 UTC) に実行想定
#
# Usage:
#   .\ops\windows\weekly_analysis.ps1
#   .\ops\windows\weekly_analysis.ps1 -SkipMonteCarlo
#   .\ops\windows\weekly_analysis.ps1 -Window 14
#
# タスクスケジューラ登録例:
#   $Action = New-ScheduledTaskAction -Execute "powershell.exe" `
#       -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$PWD\ops\windows\weekly_analysis.ps1`"" `
#       -WorkingDirectory $PWD
#   $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "09:00"
#   Register-ScheduledTask -TaskName "ZTB-WeeklyAnalysis" -Action $Action -Trigger $Trigger `
#       -Description "168# weekly hindsight + MC analysis"

param(
    [switch]$SkipMonteCarlo,
    [switch]$SkipHindsight,
    [int]$Window = 7,
    [string]$OutputDir = "analysis_results\weekly"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
Push-Location $ProjectRoot

try {
    $date = Get-Date -Format "yyyy-MM-dd"
    $endDate = $date
    $startDate = (Get-Date).AddDays(-$Window).ToString("yyyy-MM-dd")

    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    }

    Write-Host "=== Weekly Analysis ($startDate ~ $endDate) ===" -ForegroundColor Cyan

    # ---- 1. Hindsight Filter ----
    if (-not $SkipHindsight) {
        $hindsightOutput = Join-Path $OutputDir "hindsight_${date}.json"
        Write-Host "--- Hindsight Filter (${Window}d window) ---" -ForegroundColor Yellow

        & ".venv\Scripts\python.exe" -m scripts.v460.analysis.hindsight_filter `
            --start $startDate --end $endDate `
            --output $hindsightOutput 2>&1 | Write-Host

        if ($LASTEXITCODE -eq 0 -and (Test-Path $hindsightOutput)) {
            Write-Host "  -> $hindsightOutput" -ForegroundColor Green
        } else {
            Write-Host "  Hindsight filter returned exit code $LASTEXITCODE" -ForegroundColor Red
        }
    }

    # ---- 2. PnL Monte Carlo ----
    if (-not $SkipMonteCarlo) {
        $mcOutput = Join-Path $OutputDir "pnl_mc_${date}.json"
        Write-Host "--- PnL Monte Carlo ---" -ForegroundColor Yellow

        & ".venv\Scripts\python.exe" scripts/v460/run_pnl_monte_carlo.py `
            --sensitivity --output $mcOutput 2>&1 | Write-Host

        if ($LASTEXITCODE -eq 0 -and (Test-Path $mcOutput)) {
            Write-Host "  -> $mcOutput" -ForegroundColor Green
        } else {
            Write-Host "  Monte Carlo returned exit code $LASTEXITCODE" -ForegroundColor Red
        }
    }

    # ---- 3. Cleanup: 30日以上古い結果を削除 ----
    if (Test-Path $OutputDir) {
        $removed = Get-ChildItem $OutputDir -Filter "*.json" |
            Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) }
        if ($removed) {
            $removed | Remove-Item -Force
            Write-Host "Cleaned up $($removed.Count) old files" -ForegroundColor DarkGray
        }
    }

    Write-Host "=== Weekly Analysis Complete ===" -ForegroundColor Cyan
    exit 0

} catch {
    Write-Host "Weekly analysis failed: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}
