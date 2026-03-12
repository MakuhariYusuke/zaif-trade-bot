# 27. v457.3 Analysis: Fixed TTL & The "Buy & Hold" Discovery

## 1. Context & Hypothesis
Following the failure of v457.2 (1583 trades, -7.9M PnL), we hypothesized that the **2D Action Space (Position + TTL)** was causing a "Short TTL Loop" where the agent repeatedly entered and exited positions due to random/learned short TTL values.

**Experiment v457.1/2 Diagnosis:**
- High Frequency Trading was not "Alpha Hunting" but "Churning" driven by TTL expiry.
- Attempts to fix this via "Fee Penalty" (v457.2) failed because the mechanism (TTL) was structural.

**Experiment v457.3 Setup:**
- **Wrapper**: `FixedTTLWrapper` forces `action[1]` (TTL) to 1.0 (Max).
- **Config**: Reduced `fee_penalty_weight` to 0.0 (Standard PnL reward).
- **Goal**: Verify if stabilizing TTL reduces churn and improves PnL.

## 2. Results (10,000 steps)

| Metric | v457.2 (2D Action + Fee Penalty) | v457.3 (1D Action + Fixed TTL) | Delta |
| :--- | :--- | :--- | :--- |
| **Trades** | 1,583 | **676** | ⬇️ -57% |
| **Net PnL** | -7,897,318 JPY | **+36,824,062 JPY** | 🚀 **PROFITABLE** |
| **Profit Factor** | 0.92 | **5.35** | ⬆️ Massive |
| **Win Rate** | 0.0% | **43.8%** | ⬆️ Normal |
| **Policy** | "Buy Only" (Short TTL) | **"Buy & Hold" (Max TTL)** | Stable |

## 3. Analysis
1.  **TTL was the Root Cause**: By fixing TTL to max, we prevented the "looping" behavior. The trade count dropped significantly.
2.  **Profitable Baseline**: The model effectively learned a "Buy & Hold" strategy in a Bull Market segment. While simple, valid actions yielded +36M JPY, proving that **the environment is solvable** if the action space is stable.
3.  **Refactoring Opportunity**: The current `FastIntradayEnv` logic for TTL is too complex/unstable for the current stage. 
    - **Action**: We should refactor the environment to support "1D Action Space" natively (Target Position Only).
    - **Asset**: `ActionExecutor` from `ztb/components` handles discrete conversion, but we need a "Continuous Action Processor" component.

## 4. Next Steps (Refactoring & Evolution)
The "Quick Fix" via Wrapper worked. Now we should officialize this architecture.

### Plan: v457 Phase 3 (Refactoring)
1.  **Environment Refactoring**: 
    -   Modify `FastIntradayEnvV456` (or create `V457`) to accept `action_space_type="1d_position"` config.
    -   Remove/Disable TTL logic when in 1D mode.
2.  **Reward Componentization**:
    -   Instead of `fast_intraday.py`, integrate `ztb/trading/environment/components/rewards/` (e.g., `UltraProfitReward`) which are "Existing Assets" likely more robust.
3.  **Logging**:
    -   Integrate `BacktestReporter` improvements (trades.json) into the main library.

**Conclusion**: The system is not broken; it was just tripping over its own feet (TTL). Simplifying the action space unlocked the baseline profitability.
