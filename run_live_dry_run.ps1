# Run Live Trading in Dry Run Mode with Iteration 12 Model

$ModelPath = "models/sac_v450_phase6_hft.zip"
$Algorithm = "sac"
$Venue = "coincheck" # or 'sim'
$DurationHours = 24

# Ensure the virtual environment is activated
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Activating virtual environment..."
    . .venv/Scripts/Activate.ps1
}

Write-Host "Starting Dry Run with Model: $ModelPath"
Write-Host "Algorithm: $Algorithm"
Write-Host "Venue: $Venue"
Write-Host "Duration: $DurationHours hours"

python -m ztb.trading.live_trader.main `
    --model-path $ModelPath `
    --algorithm $Algorithm `
    --venue $Venue `
    --duration-hours $DurationHours `
    --dry-run `
    --log-level INFO

Write-Host "Dry Run Completed."
