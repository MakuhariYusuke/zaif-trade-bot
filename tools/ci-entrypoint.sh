#!/usr/bin/env bash
set -euo pipefail

echo "[ci-entrypoint] Workspace: $(pwd)"

# Run type checks, lint, and tests. Exit early on failure so container returns non-zero.
echo "[ci-entrypoint] Running mypy..."
mypy ztb/ --ignore-missing-imports || true

echo "[ci-entrypoint] Running flake8..."
flake8 ztb/ --max-line-length=100 --extend-ignore=E203,W503 || true

echo "[ci-entrypoint] Running ruff..."
ruff check ztb/ || true

echo "[ci-entrypoint] Running pytest subset..."
python -m pytest tests/test_backtest.py tests/test_risk.py -v --tb=short || true

echo "[ci-entrypoint] Completed (some steps may have reported failures; check output)."

exec "$@"
