# Phase 3: Execution Realism Gap Analysis & Roadmap

**Date:** 2025-12-07
**Status:** Analysis Complete / Roadmap Defined

## 1. The "Realism Gap" Discovery

In Phase 3, we conducted a controlled experiment to verify the impact of realistic execution constraints (Slippage, Latency, Fees) on the v450 model. The results were conclusive and severe.

### Experiment Results
| Metric | Ideal Environment | Realistic Environment | Gap |
|---|---|---|---|
| **Mean Reward** | **70,860** | **-21,369** | **-92,229** |
| **Outcome** | Profitable | **Emergency Stop (Drawdown)** | Catastrophic Failure |

### Interpretation
The current model (trained in an "Ideal" environment) has overfitted to the absence of friction. It likely employs a high-frequency scalping strategy that captures micro-profits. In the "Realistic" environment, these micro-profits are eroded by:
1.  **Slippage (ATR-based):** The price moves against the trade before execution.
2.  **Latency (50ms+):** The opportunity vanishes before the order reaches the matching engine.
3.  **Fees:** The cost of trading exceeds the captured spread.

## 2. Technical Findings & "Gotchas"

During the implementation of Phase 3, several technical issues were identified that hinder rapid experimentation.

### A. `UnifiedTrainer` Rigidity
The `UnifiedTrainer` class is designed for standardized training pipelines but proved brittle for comparative experiments.
- **Issue:** It was difficult to override specific environment parameters (like `feature_set` or `execution_model`) for the evaluation environment independently of the training environment without modifying the global config object.
- **Consequence:** We had to bypass `UnifiedTrainer` and use `stable_baselines3.SAC` directly in `run_execution_comparison.py` to ensure correct configuration propagation.
- **Action Item:** Refactor `UnifiedTrainer` to accept explicit `eval_env_config` overrides.

### B. Observation Space Fragility
A critical bug occurred where the training environment defaulted to a minimal feature set (10 dims) while the evaluation environment used the full set (143 dims), causing a crash.
- **Insight:** Configuration objects (`EnvironmentConfig`) must be explicitly passed and validated at every stage. Implicit defaults in the library code can lead to silent failures or dimension mismatches.

## 3. Roadmap to Bridge the Gap

To create a model that survives in the Realistic environment, we must incorporate these constraints into the training process.

### Strategy A: Execution-Aware Training (Immediate Next Step)
Instead of training on "Ideal" and evaluating on "Realistic", we will **train directly on the Realistic environment**.
- **Mechanism:** Enable `RealisticExecutionModel` during the training phase.
- **Hypothesis:** The agent will learn that high-frequency trading is costly and will naturally converge towards lower-frequency, higher-conviction trades (Trend Following / Swing).

### Strategy B: Curriculum Learning (Refinement)
If Strategy A fails to converge (too hard initially), we will use Curriculum Learning.
- **Stage 1:** Ideal Environment (Learn basic mechanics).
- **Stage 2:** Low Friction (Small fees, no slippage).
- **Stage 3:** High Friction (Full realistic model).

### Strategy C: Reward Shaping
Explicitly penalize behaviors that are dangerous in realistic settings.
- **Turnover Penalty:** Penalize high volume of transactions.
- **Holding Period Bonus:** Reward holding positions for longer durations to capture larger moves.

## 4. Next Actions

1.  **Create Phase 4 Experiment:** `experiments/v450/phase4_execution_aware_training`.
2.  **Implement Training Script:** Create a training script that utilizes `RealisticExecutionModel` from step 0.
3.  **Compare:** Run the comparison again. The goal is to see a positive result in the Realistic column, even if the Ideal result is lower than before.
