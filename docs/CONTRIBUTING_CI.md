# Local CI Validation (Alternative to GitHub Actions)

This repository historically ran a number of CI checks via GitHub Actions. If you prefer not to use Actions, you can locally reproduce the CI steps using Docker and the scripts provided in the `scripts/` folder.

Prerequisites
- Docker / Docker Desktop or a Linux VM
- A machine with internet access to install Python packages

Quickstart (PowerShell)
```powershell
.\scripts\run_local_validation.ps1
```

Quickstart (Bash)
```bash
./scripts/run_local_validation.sh
```

What the local CI runs
- `mypy ztb/ --ignore-missing-imports` (type checking)
- `flake8 ztb/` (linting)
- `ruff check ztb/` (optional, ruff linting)
- `pytest tests/test_backtest.py tests/test_risk.py` (subset of unit tests)

How to extend the script
- To build the Docker image for a single Python version: `docker build --build-arg PYTHON_VERSION=3.11 -t ztb-ci:3.11 -f docker/ci.Dockerfile .`
- To run a shell in the container (interactive):
  `docker run --rm -it -v "$(pwd)":/workspace -w /workspace ztb-ci:3.11 bash`

Note
- The `Dockerfile` will attempt to install both `requirements.txt` and `requirements-dev.txt` -- if some packages fail to build, you can iterate by installing smaller sets or adding Debian packages in the Dockerfile. This gives a reproducible environment for CI checks without using GitHub Actions.
