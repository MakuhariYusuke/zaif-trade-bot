# Phase 3: Execution Realism Verification

This directory contains experiments related to verifying the impact of realistic execution constraints on the trading model.

## Files

- `run_execution_comparison.py`: The main script that trains a model in an Ideal environment and evaluates it in both Ideal and Realistic environments.
- `test_execution_model.py`: Unit tests for the `RealisticExecutionModel` class.

## Results (2025-12-07)

The comparison confirmed a massive "Realism Gap".

| Metric | Ideal | Realistic | Gap |
|---|---|---|---|
| Mean Reward | 70,860 | -21,369 | -92,229 |

The model failed catastrophically in the realistic environment, triggering an emergency stop. This indicates the need for **Execution-Aware Training** (Phase 4).
