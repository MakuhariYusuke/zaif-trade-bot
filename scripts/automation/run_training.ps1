# PowerShell script for v4XX unified training system
# Provides easy access to training and analysis commands

param(
    [string]$Action = "help",
    [string]$Version = "v435",
    [string]$Config = "",
    [switch]$SaveConfig,
    [switch]$ValidateOnly
)

$pythonCmd = "python"
$venvPath = "venv311\Scripts\activate.ps1"

# Activate virtual environment if it exists
if (Test-Path $venvPath) {
    & $venvPath
    $pythonCmd = "python"
}

function Show-Help {
    Write-Host "v4XX Unified Training System - PowerShell Interface"
    Write-Host "=================================================="
    Write-Host ""
    Write-Host "Usage: .\run_training.ps1 -Action <action> [options]"
    Write-Host ""
    Write-Host "Actions:"
    Write-Host "  train     - Train a model"
    Write-Host "  analyze   - Analyze results"
    Write-Host "  convert   - Convert configuration"
    Write-Host "  help      - Show this help"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Version <version>    - Version to use (v427, v435, v437, v440)"
    Write-Host "  -Config <path>        - Configuration file path"
    Write-Host "  -SaveConfig           - Save converted configuration"
    Write-Host "  -ValidateOnly         - Only validate configuration"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run_training.ps1 -Action train -Version v435"
    Write-Host "  .\run_training.ps1 -Action analyze -Version v440"
    Write-Host "  .\run_training.ps1 -Action convert -Config config\sac_v427_default_config.json"
}

function Invoke-Training {
    param([string]$Version, [string]$Config, [switch]$SaveConfig)

    $scriptMap = @{
        "v427" = "scripts/training/train_sac_v437_unified.py"
        "v435" = "v435\train_sac_v435_7a.py"
        "v437" = "scripts/training/train_sac_v437_unified.py"
        "v440" = "scripts/training/train_sac_v440_unified.py"
    }

    if (-not $scriptMap.ContainsKey($Version)) {
        Write-Error "Unknown version: $Version"
        return
    }

    $script = $scriptMap[$Version]
    $args = @()

    if ($Config) {
        $args += "--config", $Config
    }

    if ($SaveConfig) {
        $args += "--save-config"
    }

    Write-Host "Starting training for $Version using $script"
    & $pythonCmd $script @args
}

function Invoke-Analysis {
    param([string]$Version)

    $resultMap = @{
        "v427" = "backtest_experiments\v437.1"
        "v435" = "v435\backtest_results_v435.json"
        "v437" = "backtest_experiments\v437.1"
        "v440" = "results\v440\backtest_results_v440.json"
    }

    if (-not $resultMap.ContainsKey($Version)) {
        Write-Error "Unknown version: $Version"
        return
    }

    $results = $resultMap[$Version]

    Write-Host "Analyzing results for $Version from $results"
    & $pythonCmd -c "from ztb.analysis.v4xx_unified_analyzer import analyze_v4xx_results; analyze_v4xx_results('$results', version='$Version'.replace('v', ''))"
}

function Invoke-ConfigConversion {
    param([string]$Config)

    if (-not $Config) {
        Write-Error "Configuration file path required for convert action"
        return
    }

    Write-Host "Converting configuration: $Config"
    & $pythonCmd -c "from ztb.utils.v4xx_config_converter import convert_config_file; convert_config_file('$Config')"
}

# Main execution
switch ($Action) {
    "train" {
        Invoke-Training -Version $Version -Config $Config -SaveConfig:$SaveConfig
    }
    "analyze" {
        Invoke-Analysis -Version $Version
    }
    "convert" {
        Invoke-ConfigConversion -Config $Config
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error "Unknown action: $Action"
        Show-Help
    }
}
