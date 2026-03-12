<#
Runs the repository checks within the local Python environment (Windows PowerShell)
This will detect an existing .venv and use it; otherwise it'll create a venv at .venv
#>
param(
    [switch]$CreateVenv = $false,
    [switch]$InstallProd = $false,
    [string]$LogFile = "logs/ci-local-windows-$(Get-Date -Format yyyyMMddHHmmss).log"
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $repoRoot

$logPath = $LogFile
if (-not [System.IO.Path]::IsPathRooted($LogFile)) {
    $logPath = Join-Path $repoRoot $LogFile
}

if (-not (Test-Path -Path (Join-Path $repoRoot '.venv\Scripts\Activate.ps1'))) {
    if ($CreateVenv) {
        Write-Host "Creating venv at .venv..."
        python -m venv .venv
    } else {
        Write-Warning "No .venv found. Using existing Python environment. Set -CreateVenv to create a local venv."
    }
}

$activate = (Join-Path $repoRoot '.venv\Scripts\Activate.ps1')
if (Test-Path $activate) {
    # If the existing .venv's python points to a non-existent executable (e.g., removed Python version), recreate the venv
    $pycfg = Join-Path $repoRoot '.venv\pyvenv.cfg'
    $recreate = $false
    if (Test-Path $pycfg) {
        $content = Get-Content $pycfg -Raw
        if ($content -match 'executable\s*=\s*(.+)') {
            $exe = $matches[1].Trim()
            if (-not (Test-Path $exe)) { $recreate = $true }
        }
    }
    if ($recreate -or $CreateVenv) {
        Write-Host "Recreating .venv with current python executable..."
        python -m venv .venv --clear
    }
    . $activate
}

New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot 'logs') | Out-Null

Write-Host "Installing dev requirements (production deps skipped by default)..."
python -m pip install --upgrade pip
if ($InstallProd) {
    Write-Host "Installing production requirements as well (this may be slow and include heavy native packages)..." | Tee-Object -FilePath $logPath -Append
    python -m pip install -r config/requirements/requirements.txt -r config/requirements/requirements-dev.txt | Tee-Object -FilePath $logPath -Append
} else {
    # Only a minimal dev set to ensure the tools run locally on Windows; this avoids heavy or platform-constrained packages.
    # Add lightweight scientific packages needed for basic test collection (numpy/pandas).
    # Use --prefer-binary to prefer wheels when available and avoid building from source.
    python -m pip install --prefer-binary mypy flake8 ruff pytest pytest-cov pyyaml pydantic numpy pandas psutil scipy gymnasium scikit-learn | Tee-Object -FilePath $logPath -Append
}

Write-Host "Running mypy..." | Tee-Object -FilePath $logPath -Append
python -m mypy ztb/ --ignore-missing-imports 2>&1 | Tee-Object -FilePath $logPath -Append

Write-Host "Running flake8..." | Tee-Object -FilePath $logPath -Append
python -m flake8 ztb/ --max-line-length=100 --extend-ignore=E203,W503 2>&1 | Tee-Object -FilePath $logPath -Append

Write-Host "Running ruff..." | Tee-Object -FilePath $logPath -Append
python -m ruff check ztb/ 2>&1 | Tee-Object -FilePath $logPath -Append

Write-Host "Running a small pytest subset (skip heavy modules)" | Tee-Object -FilePath $logPath -Append
# Use a lightweight, cross-platform subset of tests that avoid heavy imports
python -m pytest tests/test_position_scaling.py tests/test_trailing_stop_placeholder.py -v --tb=short 2>&1 | Tee-Object -FilePath $logPath -Append

Write-Host "Validation complete. Logs written to $logPath"
