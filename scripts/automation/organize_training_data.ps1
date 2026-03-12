param(
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Message,
        [scriptblock]$Action
    )
    Write-Host "[STEP] $Message"
    if ($DryRun) {
        Write-Host "       (dry-run)"
        return
    }
    & $Action
}

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Move-IfExists {
    param(
        [string]$SourcePath,
        [string]$DestinationDir
    )
    if (-not (Test-Path -LiteralPath $SourcePath)) { return }
    Ensure-Dir $DestinationDir
    Invoke-Step "Move $SourcePath -> $DestinationDir/" {
        Move-Item -LiteralPath $SourcePath -Destination $DestinationDir -Force
    }
}

function Move-DirIfExists {
    param(
        [string]$SourceDir,
        [string]$DestinationDir
    )
    if (-not (Test-Path -LiteralPath $SourceDir)) { return }
    Ensure-Dir (Split-Path -Parent $DestinationDir)
    Invoke-Step "Move dir $SourceDir -> $DestinationDir" {
        Move-Item -LiteralPath $SourceDir -Destination $DestinationDir -Force
    }
}

Write-Host "== Organize training data files =="
Write-Host "DryRun: $DryRun"

# 1) Backups to archive
Get-ChildItem -LiteralPath "data" -File -Filter "*.bak" -ErrorAction SilentlyContinue | ForEach-Object {
    Move-IfExists $_.FullName "data/archives/datasets/backups"
}

# 2) Legacy snapshots / one-off exports
$legacySnapshotFiles = @(
    "data/btc_jpy_1m_dataset",
    "data/btc_jpy_1m_dataset_pre_long",
    "data/btc_jpy_1m_dataset_expanded.csv",
    "data/btc_jpy_1m_latest_7d_20251213_155436.csv",
    "data/btc_jpy_1m_latest_7d_20251215_073955.csv",
    "data/btc_jpy_1m_latest_7d_20251215_074136.csv",
    "data/btc_jpy_1m_yahoo_20251207_090329.csv",
    "data/btc_jpy_yahoo_real_20251021.csv",
    "data/btc_jpy_yahoo_real_20251021_corrected.csv",
    "data/btc_jpy_yahoo_real_20251021_fixed.csv",
    "data/btc_jpy_yahoo_real_20251021_fixed_featured.csv"
)
foreach ($f in $legacySnapshotFiles) {
    Move-IfExists $f "data/datasets/legacy/root_snapshots"
}

# 3) Synthetic range datasets to dedicated legacy bucket
$syntheticRangeFiles = @(
    "data/range_choppy.csv",
    "data/range_choppy_featured.csv",
    "data/range_medium.csv",
    "data/range_medium_featured.csv",
    "data/range_tight.csv",
    "data/range_tight_featured.csv",
    "data/range_wide.csv",
    "data/range_wide_featured.csv"
)
foreach ($f in $syntheticRangeFiles) {
    Move-IfExists $f "data/datasets/legacy/synthetic_ranges"
}

# 4) Debug/test one-off outputs
$testOutputFiles = @(
    "data/debug_sell_bias_output.csv",
    "data/test_featured.csv",
    "data/btc_jpy_15m_from_test_minute.csv",
    "data/btc_jpy_5m_from_test_minute.csv"
)
foreach ($f in $testOutputFiles) {
    Move-IfExists $f "data/datasets/legacy/test_outputs"
}

# 5) Root-level backtest artifact file
Move-IfExists "data/backtest" "data/archives/backtest"

# 6) Nested accidental dump directory
if (Test-Path -LiteralPath "data/data") {
    if (-not (Test-Path -LiteralPath "data/archives/workspace_dump/data_data_legacy")) {
        Move-DirIfExists "data/data" "data/archives/workspace_dump/data_data_legacy"
    }
    else {
        $suffix = (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-DirIfExists "data/data" "data/archives/workspace_dump/data_data_legacy_$suffix"
    }
}

Write-Host "== Done =="
Write-Host "Recommended checks:"
Write-Host "  1) Get-ChildItem data -File"
Write-Host "  2) Get-ChildItem data/datasets/legacy -Recurse -File"
