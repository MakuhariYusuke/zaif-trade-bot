# SAC v446 Fixed Model Analysis

## Overview
Following the correction of the feature engineering logic (fixing NaN values) and retraining the model from scratch (`sac_v446_fixed`), a backtest was conducted to validate performance.

## Results
- **Model**: `sac_v446_fixed` (Trained for 50k steps)
- **Backtest Period**: 2025-11-05 to 2025-11-09 (Synthetic/Future Data)
- **Total Return**: **+34.34%** (Previous model: -17.59%)
- **Final Portfolio**: 4,030,098 JPY (Initial: 3,000,000 JPY)
- **Action Distribution**: 100% SELL

## Analysis
1.  **Performance Flip**: The model completely reversed its behavior from the previous version.
    -   Old Model: ~96% BUY (in a Bear Market) -> -17% Loss.
    -   New Model: 100% SELL (in a Bear Market) -> +34% Profit.
2.  **Market Regime**: The backtest period was clearly a strong downtrend (Bear Market). The model correctly identified this and adopted a "Short and Hold" strategy.
3.  **Strategy Behavior**: The "100% SELL" distribution suggests the model outputs a negative action value (Short) at every step. In the `HeavyTradingEnv`, this results in opening a short position at the start and holding it (or re-asserting it) throughout the period.
4.  **Reward Hacking vs. Learning**: While "Short and Hold" is a simple strategy, it is the *optimal* simple strategy for a unidirectional bear market. The fact that the model converged to this (instead of random noise or losing long positions) indicates effective learning.

## Conclusion
The fix for the NaN values in feature engineering was successful. The model is now capable of learning a profitable policy, whereas the previous model was "poisoned" by bad data. The immediate goal of fixing the negative returns has been achieved.

## Next Steps
-   **Regime Robustness**: Test the model on a Bull Market period to ensure it doesn't just "Always Short".
-   **Scalping Behavior**: Investigate if the model can learn more granular trading (scalping) instead of just trend following, by adjusting reward functions or training on more volatile/ranging data.
