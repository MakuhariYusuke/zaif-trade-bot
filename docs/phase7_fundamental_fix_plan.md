# Phase 7: Fundamental Fix - Temporal & Regime Awareness (v451)

## 1. Problem Identification (Root Cause Analysis)
Based on the Phase 6 analysis (`phase6_weakness_report.md`), the model exhibits specific weaknesses:
1.  **Time-Specific Failures:** Consistently loses money at 14:00, 17:00, and 01:00.
2.  **Regime-Specific Failures:** Fails during "Medium-Low" volatility transitions.
3.  **Risk/Reward Imbalance:** Avg Loss > Avg Win.

**Root Cause:** The model (Iter 12 / v450) was trained on a feature set that **lacks explicit awareness of Time and Volatility Regimes**. It is effectively "blind" to the time of day and the broader market context, forcing it to learn a "global average" policy that fails in specific sub-conditions.

## 2. Proposed Solution: v451 (Phase 7)
Instead of applying post-hoc filters (symptomatic treatment), we will fundamentally upgrade the model's perception (State Space).

### A. Feature Engineering Upgrade (`SACv451FeatureEngineer`)
We have created a new feature engineering pipeline that adds:
1.  **Cyclical Time Encoding:**
    - `hour_sin`, `hour_cos`: Allows the model to learn continuous daily patterns (e.g., "14:00 is dangerous").
    - `day_sin`, `day_cos`: Weekly seasonality.
    - `minute_sin`, `minute_cos`: Intraday microstructure timing.
2.  **Explicit Volatility Regime Features:**
    - `vol_rank`: Rolling percentile of current volatility (0.0 - 1.0).
    - `regime_low`, `regime_med_low`, `regime_med_high`, `regime_high`: One-hot encoded regime flags.
    - `vol_ratio`: Ratio of short-term to long-term volatility (Expansion/Contraction signal).

### B. Learning Strategy
- **Algorithm:** SAC (Soft Actor-Critic) - Retained for its entropy-regularized exploration.
- **Hyperparameters:**
    - `gamma`: 0.80 (HFT/Scalping focus).
    - `ent_coef`: 0.05 (High exploration to discover new time-based strategies).
    - `loss_penalty`: 1.2x (Asymmetric reward to fix Risk/Reward ratio).
- **Training Data:** Pre-generated `btc_jpy_1m_v451.csv` containing the new features.

## 3. Execution Plan
1.  **Data Generation:** Generate `data/btc_jpy_1m_v451.csv` using `SACv451FeatureEngineer`. (✅ Done)
2.  **Training:** Run `experiments/v451/run_training_v451.py`.
3.  **Evaluation:** Compare v451 vs v450 (Iter 12) specifically on:
    - Performance at 14:00/17:00.
    - Performance in Med-Low Volatility.

This approach aligns with the "vXXX series" philosophy of iterative, fundamental improvements to the agent's capabilities rather than heuristic patching.
