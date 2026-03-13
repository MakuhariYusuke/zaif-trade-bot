# v455 Phase Summary & Handover Report

## 1. Overview
**Objective**: Stabilize the High-Frequency Trading (HFT) agent and resolve critical issues related to "Instant Death" (rapid bankruptcy) and "Bleeding" (slow loss due to fees/spread).

**Status**: **Completed (Stability Achieved)**
- The agent no longer crashes or bankrupts instantly.
- Risk management and cost awareness have been successfully learned.
- Profitability is not yet positive (-9.3% in backtest), indicating a need for better predictive signals in the next phase.

## 2. Key Achievements & Changes

### 2.1. Stability Fixes
- **Leverage Reduction**: Adjusted `max_position` from 1.0 BTC to 0.01 BTC (relative to 100k JPY balance), reducing leverage from ~10x to ~1.37x. This eliminated the "Instant Death" loop.
- **Memory Optimization**: Fixed memory leaks in `FastIntradayEnv` by explicitly deleting large DataFrames and optimizing `OnlineScaler` with batch updates.

### 2.2. Reward Engineering (`ztb/trading/rewards/fast_intraday.py`)
Implemented strict penalties to force the agent to respect market microstructure constraints:
- **`min_edge_mult` (1.5)**: Penalizes trades where the expected price move is less than 1.5x the transaction cost.
- **`vol_floor` (0.002)**: Penalizes trading in low-volatility environments where the spread eats up potential profits.
- **Time Decay**: Penalizes holding positions too long without profit.

### 2.3. Parameter Optimization
Conducted a Grid Search Sensitivity Analysis (`scripts/v455/run_sensitivity_analysis.py`) to find the optimal combination of reward parameters.
- **Optimal Config**: `min_edge_mult=1.5`, `vol_floor=0.002`.

## 3. Key Scripts & Artifacts

| Script/File | Description |
| :--- | :--- |
| `scripts/v455/train_hft.py` | Main training script (SAC). Configured for 300k steps. Includes custom logging callbacks. |
| `scripts/v455/run_sensitivity_analysis.py` | Grid search tool for tuning reward parameters. |
| `scripts/v455/backtest_hft.py` | Backtesting script using `SequentialFastIntradayEnv` on unseen data. |
| `scripts/v455/analyze_learning_curve.py` | Visualization tool for analyzing `monitor.csv` logs. |
| `ztb/trading/environment/fast_intraday_env.py` | The HFT environment. Optimized for speed and memory. |
| `ztb/trading/rewards/fast_intraday.py` | The custom reward function logic. |
| `models/v455_hft_main/sac_hft_final.zip` | The final trained model. |
| `logs/v455_hft_main/` | Training logs and TensorBoard data. |

## 4. Results Summary

### Training (300k Steps)
- **Survival**: The agent consistently survives episodes (Avg length ~925/1000 steps).
- **Cost Control**: Trade costs are low, and the agent avoids "churning".
- **Performance**: Average final balance ~95k JPY (-5%). Variance is high, with some episodes reaching +20%.

### Backtest (Last 20k Steps)
- **Result**: **-9.3% Loss** (100k -> 90.6k JPY).
- **Analysis**: The agent is stable but lacks the "Alpha" (predictive power) to consistently overcome the ~0.1% fee + spread barrier on a 1-minute timeframe using only basic features.

## 5. Handover to v456

The foundation is solid. The agent is safe and controllable. The next step is to inject **Intelligence**.

### Recommendations for v456
Refer to `docs/v456/00_improvement_proposal.md` for the detailed plan.

1.  **Multi-Timeframe (MTF)**: Integrate 5m/15m/1h trends to filter 1m signals.
2.  **Integrated Signal System**: Combine RL output with robust technical indicators (Ichimoku, Dow Theory).
3.  **Feature Engineering**: Use `ztb.features.multi_timeframe` and `ztb.features.trend` to provide better inputs to the agent.

**Note**: The current training pipeline (`train_hft.py`) is robust and can be reused for v456 by simply updating the `FastIntradayEnv` to include the new features.
