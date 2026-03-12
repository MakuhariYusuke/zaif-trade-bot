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

function Merge-Dir {
    param(
        [string]$Src,
        [string]$Dst
    )
    if (-not (Test-Path -LiteralPath $Src)) { return }
    Ensure-Dir $Dst
    Invoke-Step "Merge $Src -> $Dst" {
        $null = robocopy $Src $Dst /E /MOVE /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
        if ($LASTEXITCODE -gt 7) {
            throw "robocopy failed for $Src -> $Dst (exit=$LASTEXITCODE)"
        }
        if (Test-Path -LiteralPath $Src) {
            Remove-Item -LiteralPath $Src -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Move-FileToArchived {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    Ensure-Dir "archived"
    Invoke-Step "Move file $Path -> archived/" {
        Move-Item -LiteralPath $Path -Destination "archived" -Force
    }
}

Write-Host "== Repo cleanup finalize =="
Write-Host "DryRun: $DryRun"

# 1) root temporary files
$rootTempRegex = '^(action_analysis_.*|all_objects.*|big_objects.*|blob_sizes_.*|stash_diff.*|syntax_.*|temp_.*|test_d0_.*|training_.*_log.*|training_.*txt|scan_.*|test_.*|training_.*|backtest_gate_log.*|backtest_trades_.*|test_synthetic_dataset.*|tmp_.*\.py|temp_.*\.py)$'
Get-ChildItem -LiteralPath . -File | Where-Object { $_.Name -match $rootTempRegex } | ForEach-Object {
    Invoke-Step "Delete root file $($_.Name)" {
        Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
    }
}

# 2) move root scripts to archived/
$fixedScripts = @(
    "alert_system.py", "circuit_breaker.py", "emergency_stop.py", "health_checker.py",
    "inspect_model.py", "market_data_simulator.py", "paper_trading_manager.py",
    "performance_monitor.py", "performance_validator.py", "real_time_metrics.py",
    "recovery_system.py", "result_comparator.py", "risk_based_allocator.py",
    "rollback_manager.py", "sac.py", "virtual_portfolio_manager.py",
    "test_reward_simplified.py", "test_scale_verification.py", "test_short_step_training.py"
)
foreach ($f in $fixedScripts) { Move-FileToArchived $f }

$wildcardScripts = @("analyze_*.py", "backtest_v45*.py", "debug_*.py", "diagnose_*.py")
foreach ($pattern in $wildcardScripts) {
    Get-ChildItem -LiteralPath . -File -Filter $pattern -ErrorAction SilentlyContinue | ForEach-Object {
        Move-FileToArchived $_.Name
    }
}

# 3) directory merges
Merge-Dir "analysis_results" "results/analysis"
Merge-Dir "backtest_results" "results/backtest"
Merge-Dir "backtest_analysis_plots" "results/backtest/plots"
Merge-Dir "experiment_plots" "results/experiments/plots"
Merge-Dir "optimization_results" "results/optimization"
Merge-Dir "phase3_comparison_results" "results/phase3"
Merge-Dir "statistical_sampling_results" "results/statistical"
Merge-Dir "test_backtest_results" "results/test_backtest"
Merge-Dir "test_results" "results/test"
Merge-Dir "training_results" "results/training"
Merge-Dir "coverage" "results/coverage"

Merge-Dir "test_checkpoints" "checkpoints/test"
Merge-Dir "test_checkpoints_phase2" "checkpoints/test_phase2"
Merge-Dir "best_model" "checkpoints/best"
Merge-Dir "models" "checkpoints/models"
Merge-Dir "models_test" "checkpoints/models_test"

Merge-Dir "eval_logs" "logs/eval"
Merge-Dir "sac_action_test_logs" "logs/sac_action_test"
Merge-Dir "tensorboard" "logs/tensorboard"

Merge-Dir "backtest_experiments" "archived/backtest_experiments"
Merge-Dir "config" "configs"
Merge-Dir "schema" "configs/schema"
Merge-Dir "jsonschema" "configs/jsonschema"
Merge-Dir "_stable_baselines3_shim" "ztb/compat/sb3_shim"
Merge-Dir "utils" "ztb/utils"
Merge-Dir "websockets" "ztb/api/websockets"
Merge-Dir "venues" "ztb/api/venues"
Merge-Dir "python" "archived/python"
Merge-Dir "src" "archived/src"

if (Test-Path -LiteralPath "bundles") {
    Invoke-Step "Move bundles -> archived/bundles_legacy" {
        Move-Item -LiteralPath "bundles" -Destination "archived/bundles_legacy" -Force
    }
}

# 4) removable directories
$removeDirs = @(
    ".tmp", ".tmp-strategies", ".tmp-utils-stats", ".hypothesis", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", ".benchmarks", "htmlcov", "build", "zaif_trade_bot.egg-info", "__pycache__",
    "node_modules", "venv", "venv311", "venv311_new", ".venv311", "zaif-trade-bot-mirror",
    "git-filter-repo", "v435", "temp_scripts", "temp_model", "stable_baselines3", "sb3_contrib"
)
foreach ($d in $removeDirs) {
    if (Test-Path -LiteralPath $d) {
        Invoke-Step "Delete dir $d" {
            Remove-Item -LiteralPath $d -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "== Done =="
Write-Host "Next:"
Write-Host "  1) git status --short"
Write-Host "  2) python -m pytest tests/ -x --timeout=60"
