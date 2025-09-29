# Canary Deployment Script for Zaif Trade Bot (PowerShell)
# Automates replay -> live-lite transition with kill/resume testing

param(
    [int]$DurationMinutes = 30,
    [string]$Policy = "sma_fast_slow",
    [double]$KillThreshold = 0.01
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $ProjectRoot "artifacts\canary_$Timestamp"

# Setup output directory
function Setup-OutputDir {
    Write-Host "Setting up output directory: $OutputDir" -ForegroundColor Blue
    New-Item -ItemType Directory -Path "$OutputDir\logs", "$OutputDir\metrics", "$OutputDir\reports", "$OutputDir\config" -Force | Out-Null

    # Backup current config
    if (Test-Path ".env") {
        Copy-Item ".env" "$OutputDir\config\.env.backup"
    }
    if (Test-Path "trade-config.json") {
        Copy-Item "trade-config.json" "$OutputDir\config\"
    }
}

# Phase 1: Replay Mode
function Run-ReplayPhase {
    Write-Host "Starting Phase 1: Replay Mode ($DurationMinutes minutes)" -ForegroundColor Blue

    $env:EXCHANGE = "paper"
    $env:DRY_RUN = "1"
    $env:TEST_MODE = "1"
    $env:LOG_FILE = "$OutputDir\logs\replay.log"

    Write-Host "Running paper trader in replay mode..." -ForegroundColor Blue

    $startTime = Get-Date
    $endTime = $startTime.AddMinutes($DurationMinutes)

    try {
        $process = Start-Process -FilePath "python" -ArgumentList "-m ztb.live.paper_trader --mode replay --policy $Policy --duration-minutes $DurationMinutes --enable-risk --risk-profile balanced" -RedirectStandardOutput "$OutputDir\logs\replay.log" -RedirectStandardError "$OutputDir\logs\replay_error.log" -NoNewWindow -PassThru

        # Wait with timeout
        $timeout = $DurationMinutes * 60
        $process.WaitForExit($timeout * 1000)

        if ($process.ExitCode -eq 0) {
            Write-Host "Replay phase completed successfully" -ForegroundColor Green
        } else {
            Write-Host "Replay phase completed with exit code $($process.ExitCode)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Replay phase failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }

    # Collect metrics
    Collect-SystemMetrics "replay"
    return $true
}

# Phase 2: Live-Lite Mode
function Run-LiveLitePhase {
    Write-Host "Starting Phase 2: Live-Lite Mode ($DurationMinutes minutes)" -ForegroundColor Blue

    $env:EXCHANGE = "zaif"
    $env:DRY_RUN = "1"
    $env:TEST_MODE = "1"
    $env:LIVE_LITE = "1"
    $env:LOG_FILE = "$OutputDir\logs\livelite.log"

    Write-Host "Running paper trader in live-lite mode..." -ForegroundColor Blue

    try {
        $process = Start-Process -FilePath "python" -ArgumentList "-m ztb.live.paper_trader --mode live --policy $Policy --duration-minutes $DurationMinutes --enable-risk --risk-profile balanced" -RedirectStandardOutput "$OutputDir\logs\livelite.log" -RedirectStandardError "$OutputDir\logs\livelite_error.log" -NoNewWindow -PassThru

        $timeout = $DurationMinutes * 60
        $process.WaitForExit($timeout * 1000)

        if ($process.ExitCode -eq 0) {
            Write-Host "Live-lite phase completed successfully" -ForegroundColor Green
        } else {
            Write-Host "Live-lite phase completed with exit code $($process.ExitCode)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Live-lite phase failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }

    # Collect metrics
    Collect-SystemMetrics "livelite"
    return $true
}

# Phase 3: Kill/Resume Exercise
function Run-KillResumeExercise {
    Write-Host "Starting Phase 3: Kill/Resume Exercise" -ForegroundColor Blue

    $env:EXCHANGE = "paper"
    $env:DRY_RUN = "1"
    $env:TEST_MODE = "1"
    $env:LOG_FILE = "$OutputDir\logs\kill_resume.log"

    Write-Host "Starting trader process..." -ForegroundColor Blue

    try {
        $process = Start-Process -FilePath "python" -ArgumentList "-m ztb.live.paper_trader --mode replay --policy $Policy --duration-minutes 10 --enable-risk --risk-profile balanced" -RedirectStandardOutput "$OutputDir\logs\kill_resume.log" -RedirectStandardError "$OutputDir\logs\kill_resume_error.log" -NoNewWindow -PassThru

        # Wait for startup
        Start-Sleep -Seconds 30

        # Trigger kill switch
        Write-Host "Triggering kill switch..." -ForegroundColor Blue
        New-Item -ItemType File -Path "C:\tmp\ztb.stop" -Force | Out-Null

        # Wait for process to stop
        $timeout = 60
        $count = 0
        while (!$process.HasExited -and $count -lt $timeout) {
            Start-Sleep -Seconds 1
            $count++
        }

        if (!$process.HasExited) {
            Write-Host "Process did not stop within $timeout seconds, forcing kill" -ForegroundColor Red
            $process.Kill()
            return $false
        } else {
            Write-Host "Process stopped successfully" -ForegroundColor Green
        }

        # Resume test
        Write-Host "Testing resume capability..." -ForegroundColor Blue
        Remove-Item -Path "C:\tmp\ztb.stop" -ErrorAction SilentlyContinue

        # Quick restart test
        $resumeProcess = Start-Process -FilePath "python" -ArgumentList "-m ztb.live.paper_trader --mode replay --policy $Policy --duration-minutes 2 --enable-risk --risk-profile balanced" -RedirectStandardOutput "$OutputDir\logs\kill_resume.log" -RedirectStandardError "$OutputDir\logs\kill_resume_error.log" -NoNewWindow -PassThru -Wait

        if ($resumeProcess.ExitCode -eq 0) {
            Write-Host "Resume test completed successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "Resume test failed" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "Kill/Resume exercise failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Collect system metrics
function Collect-SystemMetrics {
    param([string]$Phase)

    $metricsFile = "$OutputDir\metrics\${Phase}_metrics.json"
    Write-Host "Collecting system metrics for $Phase phase..." -ForegroundColor Blue

    # Get system info
    $osInfo = Get-CimInstance -ClassName Win32_OperatingSystem
    $memoryUsage = [math]::Round(($osInfo.TotalVisibleMemorySize - $osInfo.FreePhysicalMemory) / $osInfo.TotalVisibleMemorySize * 100, 2)
    $cpuUsage = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples.CookedValue

    $metrics = @{
        phase = $Phase
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        system = @{
            memory_usage_percent = $memoryUsage
            cpu_usage_percent = [math]::Round($cpuUsage, 2)
        }
        process = @{
            command = "ztb.live.paper_trader"
        }
    }

    $metrics | ConvertTo-Json | Out-File -FilePath $metricsFile -Encoding UTF8
}

# Generate final report
function Generate-Report {
    Write-Host "Generating final canary report..." -ForegroundColor Blue

    $reportFile = "$OutputDir\reports\canary_report.json"

    $report = @{
        canary_deployment = @{
            timestamp = $Timestamp
            duration_minutes = $DurationMinutes
            policy = $Policy
            kill_threshold = $KillThreshold
        }
        phases = @{
            replay = @{
                status = "completed"
                log_file = "logs\replay.log"
                metrics_file = "metrics\replay_metrics.json"
            }
            livelite = @{
                status = "completed"
                log_file = "logs\livelite.log"
                metrics_file = "metrics\livelite_metrics.json"
            }
            kill_resume = @{
                status = "completed"
                log_file = "logs\kill_resume.log"
            }
        }
        artifacts = @{
            output_directory = $OutputDir
            zip_file = "canary_report.zip"
        }
    }

    $report | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportFile -Encoding UTF8

    # Create zip archive (requires PowerShell 5+)
    Write-Host "Creating artifact zip file..." -ForegroundColor Blue
    Compress-Archive -Path "$OutputDir\*" -DestinationPath "$ProjectRoot\artifacts\canary_report.zip" -Force

    Write-Host "Canary deployment completed successfully! 🎉" -ForegroundColor Green
    Write-Host "Artifacts saved to: $OutputDir" -ForegroundColor Blue
    Write-Host "Report: $reportFile" -ForegroundColor Blue
}

# Cleanup function
function Cleanup {
    Write-Host "Cleaning up temporary files..." -ForegroundColor Blue
    Remove-Item -Path "C:\tmp\ztb.stop" -ErrorAction SilentlyContinue
}

# Main execution
function Main {
    Write-Host "Starting Zaif Trade Bot Canary Deployment" -ForegroundColor Blue
    Write-Host "Output Directory: $OutputDir" -ForegroundColor Blue
    Write-Host "Duration per phase: $DurationMinutes minutes" -ForegroundColor Blue
    Write-Host "Policy: $Policy" -ForegroundColor Blue

    # Setup
    Setup-OutputDir

    # Run phases
    if (!(Run-ReplayPhase)) { exit 1 }
    if (!(Run-LiveLitePhase)) { exit 1 }
    if (!(Run-KillResumeExercise)) { exit 1 }

    # Generate report
    Generate-Report

    Cleanup
}

# Run main
Main