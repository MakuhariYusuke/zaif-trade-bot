# Review Request: v457 "Resurrect & Simplify" Strategy

You are an expert Algorithmic Trading Architect. We are currently rebooting a crypto trading bot project (version `v457`) after a period of over-engineering (`v456`) led to performance degradation.

Please review our strategy and code changes, and specifically look for "Lost Technology" in our legacy versions that we should retrieve.

## 1. Project Context
- **Goal**: High-frequency/Scalping crypto bot (BTC/JPY).
- **Current State**: v456 failed because we added too many "safety" features (Hold Penalty, Regime Filters, Complex Rewards), causing the agent to stop trading or overfit.
- **New Direction (v457)**: "Reset to PnL". We are stripping away the complex reward logic and going back to a pure PnL-based reward function that worked in older versions (v451).

## 2. Our Strategy
We have analyzed past versions and identified `v451` as the "Golden Era".
- **Key Discovery**: v451 used `Gamma=0.8` (short-sighted) and **Zero Hold Penalty**. It traded aggressively and profitably.
- **Action Taken**: 
    - Created `config/v457/base/config.yaml` to replicate these "Golden" parameters.
    - Implemented `V457RewardCalculator` (20 lines) to bypass the existing `RewardCalculator` (2000 lines of spaghetti code).
    - We are removing "Regime Filters" that were silencing the bot.

## 3. Materials for Review
Please look at the following files in the workspace (conceptually):
1.  `docs/v457/00_V457_RESET_PLAYBOOK_20260116.md` (Our roadmap)
2.  `docs/v457/01_legacy_asset_analysis.md` (Our analysis of why v451 was good)
3.  `ztb/trading/environment/components/v457_reward.py` (The new minimal reward class)
4.  `config/v457/base/config.yaml` (The new configuration)

## 4. Your Task (The Review)

### A. Strategic Critique
- Is the decision to revert to `Gamma=0.8` (very low) valid for crypto scalping?
- Is bypassing the complex RewardCalculator completely (`V457RewardCalculator`) a sound architectural decision, or are we throwing away something valuable (like `ExecutionModel` or `Validation`)?
- Are we falling into a "Survivorship Bias" trap by cherry-picking v451?

### B. "Artifact Hunting" (The Asset Discovery)
We suspect there are other "Lost Technologies" buried in `v440`~`v450` that we missed.
**Please search the file list and file contents for the following known patterns and tell us if we should revive them:**
1.  **"Dynamic Thresholds" (v450?)**: Did we use Z-Score based action thresholds? Was it effective?
2.  **"Execution Model"**: Did older versions have a better slippage model that we accidentally removed?
3.  **"Feature Engineering"**: Compare `features.yaml` (v457) with what you see in `v444` (`sac_v444_6_optimized_config.json`). Did we drop any critical indicators (like `Hurst` or `Kalman`)?

### C. Suggestions
- What is the ONE feature from the "Complex Era" (v456) that is actually worth keeping in this simple v457 model?
