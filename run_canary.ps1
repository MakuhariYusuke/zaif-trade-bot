# Canary test script for Windows
# Runs a short replay simulation to verify system integrity

param(
    [int]$DurationMinutes = 3,
    [string]$Policy = "sma_fast_slow",
    [string]$OutputDir,
    [switch]$EnableRisk
)

Write-Host "Running canary test (Windows)..."

# Set environment variables
$env:QUIET = "1"
$env:PYTHONPATH = "$PSScriptRoot"

# Create temp directory
if ($OutputDir) {
    $tempDir = $OutputDir
} else {
    $tempDir = "$env:TEMP\ztb_canary_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
}

# Copy venues directory to temp dir
Copy-Item -Path "$PSScriptRoot\venues" -Destination $tempDir -Recurse

try {
    # Run paper trader replay
    $arguments = @(
        "-m", "ztb.live.paper_trader",
        "--mode", "replay",
        "--policy", $Policy,
        "--duration-minutes", $DurationMinutes,
        "--output-dir", $tempDir
    )

    if ($EnableRisk) {
        $arguments += "--enable-risk"
    }

    $process = Start-Process -FilePath "python" -ArgumentList $arguments -WorkingDirectory $tempDir -NoNewWindow -Wait -PassThru

    # Check exit code
    if ($process.ExitCode -ne 0) {
        Write-Error "Canary test failed with exit code $($process.ExitCode)"
        exit 1
    }

    # Find artifacts directory (created by paper trader)
    $artifactsDir = Get-ChildItem $tempDir -Directory | Select-Object -First 1
    if (-not $artifactsDir) {
        Write-Error "No artifacts directory found"
        exit 1
    }

    # Verify artifacts
    $expectedFiles = @(
        "run_metadata.json",
        "orders.csv",
        "stats.json"
    )

    foreach ($file in $expectedFiles) {
        if (-not (Test-Path "$tempDir\$artifactsDir\$file")) {
            Write-Error "Missing expected artifact: $file"
            exit 1
        }
    }

    Write-Host "Canary test passed!"
    exit 0

} catch {
    Write-Error "Canary test failed: $($_.Exception.Message)"
    exit 1
} finally {
    # Cleanup only if we created the temp dir
    if (-not $OutputDir -and (Test-Path $tempDir)) {
        Remove-Item -Path $tempDir -Recurse -Force
    }
}