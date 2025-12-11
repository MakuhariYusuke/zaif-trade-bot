# Phase 4: Execution-Aware Training

**Objective:** Train the agent directly in a "Realistic" environment (with slippage, latency, and fees) to close the "Realism Gap" identified in Phase 3.

## Hypothesis
By exposing the agent to execution costs during training, it will learn to:
1.  Avoid high-frequency "noise" trading where transaction costs > profit.
2.  Target larger price movements (Trend Following / Swing) that survive slippage.
3.  Develop a more robust policy that remains profitable in production.

## Experiment Setup
- **Training Environment:** Realistic (Slippage enabled, Latency enabled, Fees enabled).
- **Evaluation Environment:** Realistic (Same as training).
- **Algorithm:** SAC (v450).
- **Trainer:** `UnifiedTrainer` (using the newly implemented `evaluation` config support).

## Success Criteria
- **Positive Reward** in the Realistic Evaluation environment.
- **Lower Trade Frequency** compared to the Phase 3 "Ideal" model.
- **No Emergency Stops** (Drawdown controlled).
