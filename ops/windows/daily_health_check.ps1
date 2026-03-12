# 146# P2-04/P3-04: 日次ヘルスチェック + KPI バッチランナー
# タスクスケジューラで毎日 15:00 JST (06:00 UTC) に実行想定
#
# Usage:
#   .\ops\windows\daily_health_check.ps1
#   .\ops\windows\daily_health_check.ps1 -SkipMonteCarlo
#   .\ops\windows\daily_health_check.ps1 -OutputDir "reports\daily"

param(
    [switch]$SkipMonteCarlo,
    [switch]$SkipOracle,
    [string]$OutputDir = "reports\daily"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
Push-Location $ProjectRoot

try {
    $date = Get-Date -Format "yyyy-MM-dd"
    $outputPath = Join-Path $OutputDir "$date.json"

    Write-Host "=== Daily Health Check ($date) ===" -ForegroundColor Cyan

    $args = @("scripts\v460\daily_health_check.py", "--output", $outputPath)
    if ($SkipMonteCarlo) { $args += "--skip-monte-carlo" }
    if ($SkipOracle) { $args += "--skip-oracle" }

    & ".venv\Scripts\python.exe" @args
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "Health check exited with code $exitCode" -ForegroundColor Red
    }

    # 168# §8 #8: Stopgap daily report + Side regime dashboard
    $stopgapOutput = Join-Path $OutputDir "stopgap_$date.json"
    Write-Host "--- Stopgap Daily Report ---" -ForegroundColor Cyan
    & ".venv\Scripts\python.exe" scripts/v460/analysis/stopgap_daily_report.py `
        --json --output $stopgapOutput 2>&1 | Write-Host

    $dashboardOutput = Join-Path $OutputDir "dashboard_$date.json"
    Write-Host "--- Side Regime Dashboard ---" -ForegroundColor Cyan
    # side_regime_dashboard.py は --output 未対応のため stdout リダイレクト
    & ".venv\Scripts\python.exe" scripts/v460/analysis/side_regime_dashboard.py `
        --json 2>&1 | Out-File -FilePath $dashboardOutput -Encoding utf8

    # 7日以上古いレポートを削除
    if (Test-Path $OutputDir) {
        Get-ChildItem $OutputDir -Filter "*.json" |
            Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
            Remove-Item -Force
    }

    exit $exitCode
} finally {
    Pop-Location
}
