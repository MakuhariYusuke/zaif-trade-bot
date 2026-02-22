# v450 Dynamic Thresholding Implementation Plan

## Overview
This document outlines the implementation of Z-score based dynamic thresholding in `ThresholdManager`. This addresses the issue where the agent's actions are suppressed by fixed thresholds, especially during early learning or low volatility periods.

## Objectives
1.  **Dynamic Adaptation**: Allow the agent to trade if its signal is statistically significant relative to its recent history, even if the absolute value is low.
2.  **Z-Score Mechanism**: Use Z-score (Standard Score) to determine the significance of an action.
3.  **Backward Compatibility**: Maintain support for existing fixed and volatility-based thresholds.

## Implementation Details

### Status: Implemented
- **Date**: 2025-12-07
- **Files Modified**:
    - `ztb/trading/environment/components/threshold_manager.py`: Added Z-score logic and fixed negative threshold volatility adjustment.
    - `ztb/utils/data/outlier_detection.py`: Added `calculate_z_score_single` helper and centralized z-score logic.
    - `ztb/trading/environment/heavy_env/core.py`: Integrated dynamic thresholding into `step` method.
- **Tests**: `tests/unit/trading/components/test_threshold_manager_v450.py` (Passed)
- **Tests**: `tests/unit/trading/components/test_threshold_manager_v450.py` (Passed), `tests/unit/utils/test_outlier_detection.py` (Passed)

### Additional improvements and notes
- `heavy_env` (`ztb/trading/environment/heavy_env/core.py`) now updates the `ThresholdManager` action history *after* threshold calculation and action validation to avoid look-ahead bias (current action excluded from z-score history).
- `ThresholdManager` supports both standard Z-score (`method='std'`) and robust MAD-based z-score (`method='mad'`) via the `z_score_method` config option. This helps control sensitivity to outliers in history.
 - `ThresholdManager` supports both standard Z-score (`method='std'`) and robust MAD-based z-score (`method='mad'`) via the `z_score_method` config option. This helps control sensitivity to outliers in history.
 - `MarketRegimeDetector` now supports relative percentile-based regime detection (`use_relative`), which can be enabled via `regime_detection_config` in your `EnvironmentConfig` (e.g. `regime_detection_config: {"use_relative": true, "reference_window": 1000, "percentile_threshold": 0.8}`).

### Modified Class: `ThresholdManager`
Location: `ztb/trading/environment/components/threshold_manager.py`

#### New Attributes
- `action_history`: A `deque` (max length N, e.g., 100) to store recent raw model outputs (absolute values).
- `z_score_threshold`: The Z-score value above which an action is considered significant (e.g., 2.0).
- `min_std`: A minimum standard deviation to prevent division by zero or extreme sensitivity in flat periods.

#### New Methods
- `update_action_stats(raw_action_value: float)`: Updates the history with the latest model output.
- `get_dynamic_threshold(raw_action_value: float) -> float`: Calculates the effective threshold for the current step.
    - If Z-score based mode is enabled:
        - Calculate $\mu$ (mean) and $\sigma$ (std) of `action_history`.
        - Calculate $z = \frac{|a| - \mu}{\sigma}$.
        - If $z > K$ (threshold), allow the trade (return a threshold slightly lower than $|a|$).
        - Otherwise, return the standard threshold.

### Configuration Updates
The `config` dictionary passed to `ThresholdManager` will support:
- `dynamic_threshold_mode`: "z_score" | "volatility" | "fixed" (default: "fixed")
- `z_score_window`: Window size for history (default: 100)
- `z_score_threshold`: Critical Z-score (default: 2.0)

## Testing Strategy
1.  **Unit Tests**:
    - Verify history updates correctly.
    - Verify Z-score calculation.
    - Test edge cases: empty history, zero variance, extremely small values.
    - Verify fallback to fixed thresholds.

## Directory Structure
No new directories needed for the core logic. Tests will be placed in `tests/unit/trading/components/`.
