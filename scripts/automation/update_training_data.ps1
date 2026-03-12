param(
    [int]$Days = 7
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $repoRoot

$pythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host "[ERROR] $pythonExe not found."
    Write-Host "[HINT] Create venv and install dependencies first."
    exit 1
}

Write-Host "[INFO] Updating BTC/JPY training data (source=all, days=$Days)"
& $pythonExe "scripts\v456\update_data_comprehensive.py" --source all --days $Days
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Data update failed."
    exit $LASTEXITCODE
}

Write-Host "[OK] Data update completed."

