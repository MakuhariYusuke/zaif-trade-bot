#!/usr/bin/env bash
set -e
echo "🔍 Running 1M checklist verification..."
pytest -q tests/test_checklist.py