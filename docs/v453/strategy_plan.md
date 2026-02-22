# v453 Strategy Plan: Hybrid Regime Filtering & Time-Based Optimization

## 1. Context & Motivation
- **v451 (Regime Adaptation)**: Failed due to excessive trading in ranging markets (Bankruptcy).
- **v452 (Volatility Suppression)**: Achieved stability (PF 1.01, MaxDD -7.4%) but lacked profitability. It successfully "stopped the bleeding" but didn't "start the winning".
- **Phase 6 HFT (Iter 12)**: Showed promise (PF 1.12) in Low Volatility but failed specifically in "Medium-Low" volatility (trend transitions) and at specific hours (14:00, 17:00, 01:00).

## 2. The Hypothesis
The current RL model (SAC) is likely a "Mean Reversion" specialist. It fails when the market transitions from "Low Volatility" (Range) to "Medium Volatility" (Trend Start).
Instead of retraining the whole model (which is unstable), we should wrap the existing model with a **"Meta-Strategy Layer"** that filters its signals based on known weaknesses.

## 3. Proposed Changes (v453)

### A. Time-Based Filtering (The "Siesta" Logic)
Based on Phase 6 analysis, specific hours are consistently unprofitable.
- **Action**: Implement a `TimeFilter` in `PositionManager` or `HeavyTradingEnv`.
- **Logic**:
    - If Hour == 14 or 17 or 01:
        - `multiplier = 0.0` (No Entry) OR `multiplier = 0.5` (Reduced Size).
    - Rationale: Avoid market open/close chop or liquidity gaps specific to the exchange/pair.

### B. "Danger Zone" Avoidance (Medium-Low Volatility)
The "Medium-Low" volatility regime is identified as the killer. This is likely the "Fakeout" or "Trend Initiation" zone where mean reversion gets crushed.
- **Action**: Refine `DynamicPositionSizer` or `MarketAdaptationManager`.
- **Logic**:
    - Define `VOLATILITY_DANGER_ZONE` (e.g., 0.005 to 0.015 daily vol).
    - If current volatility is in this zone:
        - **Block Entries** (Wait for clear Trend or clear Range).
        - OR **Tighten Stop Loss** (Assume any trade here is high risk).

### C. Trend-Direction Confirmation (Simple Alpha)
To help the model in trending markets without a full trend model:
- **Action**: Add a simple EMA filter.
- **Logic**:
    - Calculate `EMA_Short` (e.g., 20) and `EMA_Long` (e.g., 50).
    - If `Regime == TRENDING`:
        - Only allow Long if `Price > EMA_Long`.
        - Only allow Short if `Price < EMA_Long`.
    - This prevents the "Mean Reversion" model from catching falling knives in a strong trend.

## 4. Implementation Plan
1.  **Modify `HeavyTradingEnv`**: Add `hour` to the state or check it in `step()`.
2.  **Update `PositionManager`**: Add `apply_time_filter` and `apply_trend_filter`.
3.  **Configuration**: Create `v453_hybrid_config.json` with these parameters enabled.

## 5. Expected Outcome
- **Win Rate**: Increase (by removing low-probability trades).
- **Trade Count**: Decrease (filtering out bad times/regimes).
- **Profit Factor**: Increase > 1.2 (Goal).
- **Drawdown**: Remain low (inherited from v452).

This approach moves from "Pure RL" to "Hybrid RL + Heuristics", which is often the most practical path to profitability in live trading.
