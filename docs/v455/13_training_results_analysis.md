# Training Results Analysis (v455 Main Run)

## Overview
- **Run ID**: `v455_hft_main`
- **Timesteps**: 300,000
- **Model**: SAC (Stable Baselines 3)
- **Configuration**:
    - Balance: 100,000 JPY
    - Max Position: 0.01 BTC (~1.37x Leverage)
    - `min_edge_mult`: 1.5
    - `vol_floor`: 0.002

## Key Metrics (Final 50 Episodes)
Based on `monitor.csv` analysis:

1.  **Profitability**:
    -   Final balances range from **89,000 JPY to 121,000 JPY**.
    -   Several episodes achieved >20% profit (120k+).
    -   This confirms the agent *can* be profitable with the strict reward settings.

2.  **Stability**:
    -   **Episode Length**: Average ~880 steps (out of 1000).
    -   **Survival**: The agent survives the full episode in many cases.
    -   **Drawdown**: The 10% drawdown limit is still hit occasionally (approx. 1 in 5 episodes), acting as a hard stop. This is acceptable for a high-risk HFT strategy as long as the winners cover the losers.

3.  **Cost Management**:
    -   **Trade Costs**: Generally low (< 50 JPY per episode in stable periods).
    -   **Edge Shortfall**: Significantly reduced compared to previous runs, indicating the agent is respecting the `min_edge_mult` constraint.

## Conclusion
The optimization successfully stabilized the agent. It is no longer "bleeding" money aimlessly. It waits for volatility (as seen by `vol_ratio` checks) and takes trades that generally cover their costs.

## Next Steps
1.  **Backtest**: Run `backtest_v451_optimized.py` (or a new v455 version) using the saved model `models/v455_hft_main/sac_hft_final.zip` on *unseen* data (e.g., a different month or the next chunk of data) to verify generalization.
2.  **Visualization**: Plot the learning curve to see if it was still improving at 300k steps.
