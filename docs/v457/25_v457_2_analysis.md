# 25. v457.2 Analysis Report: Profit-First Strategy Verification

## 1. Overview
v457.2 "Profit-First" Retraining was executed to address the "Death by Fees" issue where the model exhibited high frequency trading with negligible net profit.
- **Goal**: Minimize unnecessary trading by penalizing fees heavily in the reward function.
- **Training**: 10,000 steps with `fee_penalty_weight: 2.0`.
- **Model**: `sac_v457_2_final_1768713451.zip`.

## 2. Backtest Results (First 10,000 steps)
The backtest was run on the same dataset used for training to verify behavior change.

| Metric | v457.1 (Previous) | v457.2 (Current) | Difference |
| :--- | :--- | :--- | :--- |
| **Total Steps** | 10,000 | 10,000 | - |
| **Total Trades** | ~150 (Frequency Control) | **1,583** | ⬆️ +900% |
| **Action** | Bang-Bang (±1.0) | **Buy Only (100%)** | ⚠️ Frozen |
| **Net PnL** | -1M JPY | **-7.9M JPY** | ⬇️ Worse |
| **Gross PnL** | Positive | -69k JPY | ⬇️ Negative |
| **Fee Cost** | High | -7.5M JPY | ⚠️ Extreme |

## 3. Analysis of Failure
The model has collapsed into a **"Buy Only" Policy**.
- **Observation**: `buy: 10000 (100.0%)`. The model output `Action=1.0` at every single step.
- **Mechanism**: 
  - Despite the high fee penalty, the model learned that "Holding Long" might be slightly better than "Changing Position" (which incurs cost) or "Shorting" (in a bull run).
  - However, in this specific implementation, it seems the model is **constantly re-entering or trying to increase position**, or simply stuck at +1.0 output.
  - Since the `FastIntradayEnv` executes a trade if `target_position != current_position`, and if the logic interprets +1.0 repeatedly as "maintain max long", it shouldn't trade every step unless the `position` variable isn't persisting or updating correctly, OR if the environment logic treats continuous +1.0 action as "Buy More" (rebalancing) excessively.
  - **Critical Finding**: 1,583 trades in 10,000 steps = 15.8% trade rate. This is extremely high. The "Profit First" penalty backfired or wasn't learned in just 10k steps.

## 4. Why 10k steps failed?
- **Insufficient Exploration**: 10,000 steps is likely too short for SAC to map the "Fee Penalty" to specific actions. The initial random exploration or high entropy (`ent_coef=0.05`) likely caused a lot of random trades, accumulating massive negative rewards.
- **Policy Collapse**: Faced with massive penalties from random trading, the policy might have collapsed to a single edge (+1.0) hoping it's a local optimum, rather than finding the "Hold (0.0)" optimal Strategy.
- **Reward Scale**: With `reward_scale=10.0` and massive fees, the negative reward signals are huge. This can cause gradients to explode or vanish, pushing weights to extremes.

## 5. Next Steps
The "Short Training" approach with "Heavy Penalty" has resulted in a degenerate policy (Always Buy) and catastrophic accumulation of fees.
- **Immediate Action**: Stop v457.2 line.
- **Recommendation**:
  1.  **Resume Phase 2 Tuning (Frequency Control)**: The previous approach (Case F) had 153 trades. We should optimize *that* rather than forcing a heavy-handed reward penalty that breaks learning stability in short runs.
  2.  **Longer Training**: If we pursue v457.2, we need 100k+ steps for the critic to learn that "Doing Nothing" is the best way to avoid fees. 10k steps is just noise.

**Conclusion**: The hypothesis that "Heavy Fee Penalty in Reward" would *quickly* fix over-trading is **rejected** for short training durations. It forces the model into a standard "Bang-Bang" failure mode faster.
