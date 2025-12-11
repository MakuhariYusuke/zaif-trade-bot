# Phase 4.1 Stabilization Analysis Report

## 1. Objective
The primary goal of Phase 4.1 was to address the "model collapse" observed in Phase 4 (where the agent stopped trading after step 6,000) and to verify if introducing stricter risk management constraints and a lower learning rate would stabilize the training process.

## 2. Implementation Details

### 2.1 Data Expansion
- **Source:** Yahoo Finance (BTC-JPY)
- **Volume:** Expanded from ~7,000 rows to ~11,000 rows (merged recent 7 days of 1-minute data).
- **Purpose:** To provide the model with more recent market context and reduce overfitting to a small dataset.

### 2.2 Environment Enhancements (`HeavyTradingEnv`)
Two critical safety mechanisms were implemented to prevent "gambler's ruin" scenarios:
1.  **Bankruptcy Threshold:**
    -   **Logic:** If Portfolio Value < 2,000 JPY (1% of initial), the episode terminates immediately with `done=True`.
    -   **Goal:** Prevent the agent from digging an infinite hole and learning from "dead" states.
2.  **Drawdown Penalty:**
    -   **Logic:** If drawdown exceeds 20%, a penalty is applied proportional to the excess drawdown.
    -   **Goal:** Discourage high-volatility strategies that risk blowing up the account.

### 2.3 Algorithm Configuration (SAC)
- **Action Space:** Explicitly set to `Continuous` (Box) to satisfy SAC requirements.
- **Learning Rate:** Reduced to `1e-4` (from `3e-4`) to prevent catastrophic forgetting and unstable updates.
- **Hyperparameters:**
    -   `batch_size`: 512
    -   `buffer_size`: 100,000
    -   `ent_coef`: "auto"

## 3. Experiment Results

### 3.1 Execution Summary
- **Status:** Completed successfully (50,000 steps).
- **Training Time:** ~57 minutes.
- **Final Reward:** -0.042 (Average per step).

### 3.2 Key Observations
1.  **Safety Mechanisms Verified:**
    -   **Bankruptcy:** The logs confirm episodes terminating early: `Episode terminated due to bankruptcy: PV=1977.26 < 2000.0`.
    -   **Drawdown:** Warnings and penalties were triggered: `High drawdown warning at step 1059: drawdown 5.9%`.
    -   **Emergency Stop:** The training loop correctly handled extreme drawdown cases.

2.  **Action Distribution (Stabilized):**
    Unlike Phase 4, where the agent froze (100% HOLD), the Phase 4.1 agent maintained a healthy activity level throughout the 50,000 steps:
    -   **HOLD:** 49.3%
    -   **BUY:** 25.2%
    -   **SELL:** 25.4%
    -   **Mean Action:** ~0.0 (Centered, not stuck at extremes).

3.  **Performance Analysis:**
    -   **Early Training (Steps 0-2500):** High losses (-1300 to -1600 reward).
    -   **Mid Training (Steps 3000-6000):** Significant improvement (-80 reward).
    -   **Late Training:** Stabilized around -100 to -200 reward.
    -   **Outcome:** The agent is "losing money slowly" rather than "losing everything instantly" or "doing nothing".

## 4. Conclusion
The "Stabilization" hypothesis is **partially verified**.
-   **Success:** The model no longer collapses into a "zombie" state (100% HOLD). The lower learning rate and continuous action space configuration kept the policy active and distributed.
-   **Success:** The safety rails (bankruptcy/drawdown) function correctly, terminating bad runs before they pollute the replay buffer with meaningless data.
-   **Challenge:** The agent is still not profitable. It trades actively but consistently loses value, eventually hitting the bankruptcy threshold.

## 5. Recommendations for Phase 5
1.  **Reward Shaping:** The current penalty for drawdown might be too late. Consider a small negative reward for *any* decrease in portfolio value (volatility penalty) to encourage smoother equity curves.
2.  **Feature Engineering:** The agent might be "churning" (trading random noise). Review the input features to ensure they contain predictive signal (e.g., add more advanced momentum indicators or order book imbalance if available).
3.  **Curriculum Learning:** Start with a simplified environment (no fees, no slippage) to let the agent learn basic market mechanics, then gradually introduce fees and penalties.
