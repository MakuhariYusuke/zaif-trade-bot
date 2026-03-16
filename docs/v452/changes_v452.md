# v452 Changes Summary

## 1. ThresholdManager Improvements
*   **Validation Fix:** `_validate_config` now raises a `ValueError` instead of clamping `base_threshold` when it is out of bounds. This prevents silent configuration errors.
*   **Refactoring:** Extracted regime adjustment logic into `_apply_regime_adjustment` method for better readability and maintainability.

## 2. MarketRegimeDetector Enhancements
*   **Relative Evaluation:** Changed default `use_relative` to `True`. This allows the detector to adapt to changing market volatility conditions by using percentile-based thresholds instead of fixed absolute values.
*   **Hurst Exponent:** Implemented `_calculate_hurst` method and integrated it into `_classify_regime`.
    *   **Logic:** If Hurst Exponent < 0.45, the market is classified as having a "Strong Mean Reversion" tendency, which overrides other classifications to `MODERATE_VOLATILITY_RANGING`. This helps in identifying ranging markets more accurately.

## 3. Configuration Verification
*   Verified that `max_action_threshold` is correctly set to `1.0` in `ztb/trading/environment/utils/config.py` and `ztb/trading/backtest/adapters.py`, ensuring the "clamping bug" is resolved.
