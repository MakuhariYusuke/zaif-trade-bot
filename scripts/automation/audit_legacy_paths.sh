#!/usr/bin/env bash
set -euo pipefail

echo "== Legacy path/import audit =="
echo

echo "[1] legacy imports"
rg -n "from utils\\.|import utils\\b|_stable_baselines3_shim|from websockets\\b|import websockets\\b|from venues\\b|import venues\\b" \
  backtest scripts ops tests ztb archived --glob '!scripts/v459/**' --glob '!**/__pycache__/**' || true
echo

echo "[2] legacy directories/paths in code"
rg -n "analysis_results|backtest_results|training_results|test_results|eval_logs|sac_action_test_logs|tensorboard|test_checkpoints|test_checkpoints_phase2|best_model|models_test|config/" \
  backtest scripts ops tests ztb --glob '!scripts/v459/**' --glob '!**/__pycache__/**' || true
echo

echo "[3] direct schema path usage (candidate: configs/schema)"
rg -n "schema/|jsonschema/" ztb scripts ops tests backtest --glob '!scripts/v459/**' --glob '!**/__pycache__/**' || true
echo

echo "== Done =="
