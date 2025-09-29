# Canary test script for Windows
# Runs a short replay simulation to verify system integrity

param(
    [int]$DurationMinutes = 3,
    [string]$Policy = "sma_fast_slow",
    [switch]$EnableRisk
)

Write-Host "Running canary test (Windows)..."

# Set environment variables
$env:QUIET = "1"
$env:PYTHONPATH = "$PSScriptRoot\..\.."

# Create temp directory
$tempDir = "$env:TEMP\ztb_canary_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    # Run paper trader replay
    $arguments = @(
        "-m", "ztb.live.paper_trader",
        "--mode", "replay",
        "--policy", $Policy,
        "--duration-minutes", $DurationMinutes
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

    # Verify artifacts
    $expectedFiles = @(
        "run_metadata.json",
        "orders.csv",
        "stats.json"
    )

    foreach ($file in $expectedFiles) {
        if (-not (Test-Path "$tempDir\$file")) {
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
    # Cleanup
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force
    }
}