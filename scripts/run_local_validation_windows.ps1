<#
Runs the repository checks within the local Python environment (Windows PowerShell)
This will detect an existing .venv and use it; otherwise it'll create a venv at .venv
#>
param(
    [switch]$CreateVenv = $false,
    [switch]$InstallProd = $false,
    [string]$LogFile = "logs/ci-local-windows-$(Get-Date -Format yyyyMMddHHmmss).log"
)

if (-not (Test-Path -Path (Join-Path $PSScriptRoot '..\.venv\Scripts\Activate.ps1'))) {
    if ($CreateVenv) {
        Write-Host "Creating venv at .venv..."
        python -m venv .venv
    } else {
        Write-Warning "No .venv found. Using existing Python environment. Set -CreateVenv to create a local venv."
    }
}

$activate = (Join-Path $PSScriptRoot '..\.venv\Scripts\Activate.ps1')
if (Test-Path $activate) {
    # If the existing .venv's python points to a non-existent executable (e.g., removed Python version), recreate the venv
    $pycfg = Join-Path $PSScriptRoot '..\.venv\pyvenv.cfg'
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

New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot '..\logs') | Out-Null

Write-Host "Installing dev requirements (production deps skipped by default)..."
python -m pip install --upgrade pip
if ($InstallProd) {
    Write-Host "Installing production requirements as well (this may be slow and include heavy native packages)..." | Tee-Object -FilePath $LogFile -Append
    python -m pip install -r config/requirements/requirements.txt -r config/requirements/requirements-dev.txt | Tee-Object -FilePath $LogFile -Append
} else {
    # Only dev requirements to speed up local runs and avoid heavy packages like torch
    python -m pip install -r config/requirements/requirements-dev.txt | Tee-Object -FilePath $LogFile -Append
}

Write-Host "Running mypy..." | Tee-Object -FilePath $LogFile -Append
python -m mypy ztb/ --ignore-missing-imports 2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Host "Running flake8..." | Tee-Object -FilePath $LogFile -Append
python -m flake8 ztb/ --max-line-length=100 --extend-ignore=E203,W503 2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Host "Running ruff..." | Tee-Object -FilePath $LogFile -Append
python -m ruff check ztb/ 2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Host "Running a small pytest subset (skip heavy modules)" | Tee-Object -FilePath $LogFile -Append
python -m pytest tests/test_backtest.py tests/test_risk.py -v --tb=short 2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Host "Validation complete. Logs written to $LogFile"
