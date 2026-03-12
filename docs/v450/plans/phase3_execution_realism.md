# Phase 3: Execution Realism (Implementation Plan)

## Objective
Bridge the gap between backtest results and live trading performance by introducing realistic market friction and execution constraints.

## Key Components

### 1. Advanced Slippage Model
Current implementation likely uses a fixed percentage. We need:
- **Volatility-Adjusted Slippage:** Slippage increases when ATR (Average True Range) is high.
- **Volume-Adjusted Slippage:** (If volume data available) Larger orders incur more slippage.
- **Spread Simulation:** Explicit bid-ask spread modeling.

### 2. Latency Simulation
- **Network Latency:** Random delay between signal generation and order placement (e.g., 50ms - 500ms).
- **Processing Latency:** Time taken for the exchange to match the order.

### 3. Order Book Pressure (Impact)
- Simulate the market moving against us as we trade (Market Impact).

## Implementation Steps

1.  **Analyze Existing Execution Logic:** Review `ztb/trading/environment` and `ztb/trading/cost`. (Done)
2.  **Design `ExecutionModel` Interface:** Create a flexible interface for different execution simulations. (Done: `ztb/trading/execution/model.py`)
3.  **Implement `RealisticExecutionModel`:**
    - Integrate ATR for dynamic slippage. (Done: `ztb/trading/execution/realistic.py`)
    - Add random latency delays in the environment step. (Done)
4.  **Update `EnvironmentConfig`:** Add parameters for `execution_model`, `base_latency`, `slippage_multiplier`. (Done)
5.  **Create Phase 3 Experiments:** `experiments/v450/phase3/` to compare "Ideal" vs "Realistic" performance. (Next Step)

## Success Metrics
- "Realism Gap": Difference between "Ideal Backtest" and "Realistic Backtest" should be quantified.
- Robustness: Strategy should remain profitable even with 2x expected slippage.
