# v454 Hybrid Strategy Analysis: The "High Win Rate / Low Return" Paradox

## 1. Overview
**Date**: 2025-12-15
**Model**: SAC v454 (Inverse Confidence)
**Configuration**: `config/v454/sac_v454_config.json` + `hybrid_config` (v453 port)

The v454 model has achieved **stability** (eliminating the -3.67% baseline loss) but has entered a state of **paralysis** (0% return).

## 2. Performance Metrics

| Metric | Baseline (v454 Hybrid) | Threshold Relaxed (0.38) | Filter Relaxed (Allow High Vol) |
| :--- | :--- | :--- | :--- |
| **Total Return** | **-0.01%** | -0.06% | **-4.74%** |
| **Win Rate** | **97.5%** | N/A | N/A |
| **Total Trades** | 48 | 52 | **369** |
| **Drawdown** | Negligible | Negligible | High |

## 3. Analysis of the Paradox

### 3.1. The "Safety First" Success
The Hybrid Strategy (ported from v453) successfully identified `high_volatility_ranging` as a "Kill Zone".
- **With Filter**: The model avoids this regime and preserves capital (Return ~0%).
- **Without Filter**: The model trades aggressively in this regime (369 trades) but gets slaughtered (-4.74%).

**Conclusion**: The Regime Filter is **working correctly** and is essential for survival.

### 3.2. The "Profitability" Failure
The problem is that the model **only** seems to want to trade in the dangerous `high_volatility_ranging` regime.
- In "Safe" regimes (Trends, Low Volatility), the model is **silent** (only 48 trades in the test period).
- Lowering the confidence threshold (`continuous_to_discrete_threshold`) from `0.5` to `0.38` barely increased volume (48 -> 52).

**Root Cause**: The model is likely **overfitted to volatility**. It interprets "High Volatility" as "Opportunity" (false positive), and "Low Volatility" as "Nothing to do".

## 4. Comparison with v453
v453 achieved +8% with similar filters. This implies:
1.  v453 had better **feature sensitivity** in Safe Regimes.
2.  v453 could extract profit from Trends/Consolidation, whereas v454 ignores them.

## 5. Recommendations for Next Steps

### 5.1. Immediate Fix: "Soft Filter" Implementation
Instead of a binary "Ban" on High Volatility, implement a **Dynamic Risk Adjustment**:
- **Regime**: `high_volatility_ranging`
- **Action**: Allow trades, but with **0.1x Position Size** or **Higher Confidence Threshold** (e.g., 0.8).
- **Goal**: Capture the few good trades in this regime without exposing full capital to the noise.

### 5.2. Strategic Fix: Re-training (Curriculum Learning)
The model needs to learn that "Safe Regimes" are profitable.
- **Action**: Re-train v454 with `high_volatility_ranging` periods **masked out** or **penalized** in the reward function.
- **Goal**: Force the optimizer to find gradients in the Low Volatility/Trend regimes.

### 5.3. Code Adjustments
- **`ActionValidator`**: Ensure `logger.debug` is used for high-frequency logs to speed up backtests.
- **`StatisticsCalculator`**: Ensure `ACTION_SELL` is correctly aggregated (already verified).
- **`EnvironmentConfig`**: Expose `risk_management` parameters for easier tuning.

## 6. Experiment 2: Soft Filter Implementation (2025-12-16)

### 6.1. Configuration Changes
- **Mode**: `soft` (Regime Filter)
- **Constraints**:
    - `high_volatility_ranging`: **Restricted** (0.2x Position, +0.2 Threshold)
    - `extreme_volatility`: **Deny**
    - `strong_bear_trend`: **Deny**
- **Code Updates**:
    - `PositionManager`: Implemented `position_multiplier` for restricted regimes.
    - `ThresholdManager`: Implemented `confidence_threshold_modifier`.
    - `ActionValidator`: Aligned action masks with `deny` regimes.

### 6.2. Results
| Metric | Baseline (Paralysis) | Exp 2 (Soft Filter) |
| :--- | :--- | :--- |
| **Total Return** | -0.01% | **-3.00%** |
| **Win Rate** | 97.5% | 95.3% |
| **Total Trades** | 48 | 36 |
| **Drawdown** | Negligible | Moderate |

### 6.3. Analysis
- **Regime Activity**:
    - `high_volatility_ranging`: 6 trades (Restricted size ~0.002 BTC).
    - `extreme_volatility`: 0 trades (Successfully blocked).
    - Other regimes (`buy_breakout`, `sell_breakdown`) were active and likely unrestricted.
- **Performance Drop**:
    - The return dropped to -3.00% despite the "Soft Filter".
    - Since `high_volatility_ranging` trades were small (0.2x), the bulk of the losses likely came from **other regimes** that were left unrestricted (e.g., `sell_breakdown`, `buy_breakout`) or the restricted trades were still too toxic.
    - The "Safety Paradox" continues: Opening the door even slightly (or leaving other doors open) leads to losses.

### 6.4. Next Steps
- **Expand Restrictions**: The "Restricted" mode works technically (reduced size), but needs to be applied more broadly.
- **Target Regimes**: `sell_breakdown`, `buy_breakout`, `sell_volume_surge`, `buy_volume_surge`.
- **Action**: Apply `restricted` permission to these regimes to dampen volatility impact while maintaining market presence.

## 7. Experiment 3: Phase 2 - Targeted Restrictions & Optimization (2025-12-16)

### 7.1. Hypothesis & Changes
- **Hypothesis**: The "High Win Rate" is an illusion caused by step-based rewards (positive reward for holding even if the trade overall is negative). `high_volatility_ranging` is the dominant regime and cannot be ignored.
- **Changes**:
    - **Expanded Restrictions**: Applied `restricted` (0.2x) to `sell_breakdown`, `buy_breakout`, `sell_volume_surge`, `buy_volume_surge`.
    - **High Volatility Logic**: Implemented specific logic for `high_volatility_ranging`:
        - Entry Z-Score Threshold: 1.0 (Mean Reversion)
        - Stop Loss: 1.5%
        - Take Profit: 1.0%
    - **Code Fixes**: Fixed off-by-one error in regime detection and `min_trade_size` handling in `PositionManager`.

### 7.2. Results
| Metric | Baseline (Paralysis) | Exp 2 (Soft Filter) | Exp 3 (Targeted) |
| :--- | :--- | :--- | :--- |
| **Total Return** | -0.01% | -3.00% | **-0.66%** |
| **Win Rate** | 97.5% | 95.3% | **N/A** |
| **Total Trades** | 48 | 36 | **443** |

### 7.3. Analysis
- **Improvement**: Loss reduced significantly from -3.00% to -0.66%. Trade volume increased to 443, indicating the model is active.
- **Remaining Issue**: Still negative expectancy. The `high_volatility_ranging` strategy (Mean Reversion with fixed TP/SL) is close to break-even but slightly negative.
- **Reward Function**: The user suspects the reward function is misaligned (step-based vs trade-based), contributing to the "High Win Rate" illusion.

### 7.4. Next Steps
- **Parameter Tuning**: Grid search for `entry_zscore_threshold`, `stop_loss_pct`, and `take_profit_pct` in `high_volatility_ranging`.
- **Reward Function Review**: Analyze `RewardCalculator` to ensure it incentivizes *realized* profit over *unrealized* step gains.

## 8. Experiment 4: Phase 3 - Grid Search & Strategy Finalization (2025-12-16)

### 8.1. Key Discovery: "TP/SL is ineffective if the model micro-exits"
The existing implementation applied TP/SL as a *forced exit* safety net, but the SAC policy often **closed positions early**. As a result, TP/SL rarely triggered and tuning `stop_loss_pct` / `take_profit_pct` had little impact.

**Fix**: Introduce a regime-level exit override to hold positions until TP/SL triggers.
- Config key: `exit_action_source: "tp_sl"`
- Code: `HeavyTradingEnv` now supports regime-specific exit override tracking the *entry regime* of the open position.

### 8.2. Profitability Breakthrough: "Entry must be Z-Score-driven (not model-driven)"
Even with TP/SL exits fixed, model-driven entries remained slightly negative. Switching to **pure Z-Score mean-reversion entries** in `high_volatility_ranging` finally flipped expectancy positive.
- Config key: `entry_action_source: "zscore"`

### 8.3. Grid Search Result (Winning Combo)
Final tuned parameters for `high_volatility_ranging`:
- `entry_zscore_threshold`: **1.3**
- `stop_loss_pct`: **0.8%** (`0.008`)
- `take_profit_pct`: **1.3%** (`0.013`)

**Backtest Result (v454 Hybrid)**:
- **Total Return**: **+0.39%**
- **Trades**: ~413 (dominant regime: `high_volatility_ranging`)

## 9. Reward Function Review (Phase 3)
The environment reward is step-based by default (`step_pnl = trade_pnl + Δunrealized_pnl`), which can diverge from trade-level outcome perception and create a "high win-rate / negative return" illusion under clipping and shaping.

**Update**: `RewardCalculator.calculate_reward(...)` now accepts `trade_pnl` and supports configurable PnL sources:
- `pnl_mode: "step"` (default): mark-to-market per-step PnL
- `pnl_mode: "trade"`: realized/trade-only PnL (sparser, trade-outcome aligned)
- `pnl_mode: "hybrid"`: weighted combination (with optional close-only application)

This enables re-training with realized-PnL-focused rewards without affecting backtest portfolio accounting.
