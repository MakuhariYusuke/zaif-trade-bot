# Backtest & Learning Curve Analysis (v455)

## 1. Learning Curve Analysis
Based on the training logs (`logs/v455_hft_main/monitor.csv`), we analyzed the agent's performance over 300,000 steps.

### Key Statistics (Final 50 Episodes)
- **Average Reward**: -299
- **Average Episode Length**: 925 steps (Max 1000). The agent consistently survives most of the episode.
- **Average Final Balance**: **95,019 JPY** (Starting 100,000 JPY).
    - The agent is, on average, losing about 5% per episode.
    - However, variance is high: some episodes reached **120,000 JPY (+20%)**, while others hit the drawdown limit.
- **Trade Cost**: ~20 JPY per episode (Very low). The agent is not churning.
- **Edge Shortfall**: ~19. This indicates the agent still occasionally takes trades that don't meet the strict `min_edge_mult=1.5` criteria, likely due to exploration or sudden market moves.

### Interpretation
The agent has successfully learned **survival** and **cost avoidance**. It is no longer "bleeding" money rapidly. However, it has not yet mastered **consistent alpha generation**. It tends to slowly lose money (spread/fees) or take small losses, rather than finding enough profitable trades to overcome the costs.

## 2. Backtest Results (Unseen Data)
We ran a backtest on the last 20,000 steps of the dataset (approx. 14 days), which serves as a proxy for a test set.

- **Period**: Steps 7,011 to 27,011 (Last segment of data).
- **Initial Balance**: 100,000 JPY.
- **Final Balance**: **90,692 JPY**.
- **Result**: **-9.3% Loss**.

### Observations
- The backtest performance (-9.3%) is consistent with the average training performance (-5%).
- The agent did not crash (survived the full 20k steps or chunked episodes).
- The loss is gradual, suggesting the agent is struggling to overcome the **bid-ask spread** and **fees** in a realistic environment.
- The "Initial Balance" in the log (94k) suggests a significant drawdown occurred very early in the backtest, or is an artifact of the logging (reporting balance after the first few trades).

## 3. Conclusion & Recommendations

### Status
- **Stability**: ✅ Achieved. The agent survives and manages risk.
- **Profitability**: ❌ Not yet achieved. The strategy is slightly net negative.
- **Generalization**: ⚠️ The agent performs similarly on test data (slight loss), indicating it hasn't overfit wildly, but hasn't learned a robust winning strategy either.

### Root Causes
1.  **Lack of Predictive Power**: The current features (`clv`, `vol_pressure`, etc.) might not contain enough signal to predict price moves > spread + fees on a 1-minute timeframe.
2.  **High Cost Barrier**: The 0.1% fee + spread is a high hurdle for a 1-minute HFT strategy. The required move to break even is significant.
3.  **Reward Sparsity**: The strict penalties (`min_edge_mult`) might have made the agent too risk-averse, preventing it from learning complex profitable patterns.

### Next Steps
1.  **Feature Engineering**: Introduce more predictive features (e.g., Order Book Imbalance if available, or more advanced microstructure features).
2.  **Timeframe**: Consider moving to a slightly longer timeframe (e.g., 5-minute) where the signal-to-noise ratio is higher and fees are a smaller percentage of the move.
3.  **Algorithm**: Try **PPO** instead of SAC. PPO is often more stable for continuous control in finance and might handle the stochastic nature better.
4.  **Curriculum Learning**: Start with lower fees/spreads to let the agent learn the mechanics, then gradually increase them.
