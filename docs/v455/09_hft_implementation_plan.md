# Fast Intraday / HFT Implementation Plan (Revised v2)

## 1. Overview
We will create a specialized, lightweight Gym environment (`FastIntradayEnv`) focused on the "Fast Intraday" strategy. This environment will decouple from the complex `HeavyTradingEnv` to ensure the specific HFT logic (Target Position, TTL, specialized Reward) is implemented cleanly and efficiently.

## 2. Components

### A. Feature Engineering (`ztb/features/hft_proxies.py`)
We will implement a utility module to generate the microstructure proxies from 1m OHLCV data.
- **Functions**:
    - `add_hft_features(df: pd.DataFrame) -> pd.DataFrame`
- **Features**:
    - `clv` (Close Location Value)
    - `vol_pressure` (Signed Volume Pressure)
    - `impact_proxy` (Range / Volume)
    - `vol_regime` (ATR / Close)
    - `trend_persistence` (EMA divergence)
- **Leakage Prevention**: All features must be calculated using rolling windows on past data only. No `shift(-1)` or full-series normalization.

### B. Reward Function (`ztb/trading/rewards/fast_intraday.py`)
We will implement the specialized reward function as a standalone, pure function for testability.
- **Function**: `compute_hft_reward(...)`
- **Inputs**: PnL, Fee, Slippage, Position Change, Holding Time, Inventory Risk.
- **Logic**: Implements the formula: $r_t = pnl_{norm} - costs - \alpha \cdot |pos\_chg| - \beta \cdot hold - \gamma \cdot inv\_risk$
- **Normalization**: Use ATR or Price to make reward unitless.

### C. Environment (`ztb/trading/environment/fast_intraday_env.py`)
A new Gym environment class inheriting from `gym.Env`.

#### 1. State & Observation
- **Observation Space**: `Box` containing:
    - **Market Features**: Scaled via `ztb.processing.online_scaler.OnlineScaler` to prevent leakage.
    - **Account State**: `current_position` (normalized), `remaining_ttl` (normalized), `last_step_cost` (normalized).
- **Scaling Strategy**:
    - **Initialization**: `OnlineScaler` must be pre-warmed with `N` steps of past data before the episode start index to ensure stable statistics.
    - **Update**: Update scaler statistics step-by-step during the episode.

#### 2. Action Space
- **Space**: `Box(low=[-1, 0], high=[1, 1])` (Target Position, TTL Fraction).
- **Logic**:
    - `target_position`: Desired position size (fraction of max).
    - `ttl_fraction`: Desired Time-To-Live for the position.

#### 3. Execution Logic (Step)
1.  **TTL Management**:
    -   Update `position_ttl` **only** if `sign(target) != sign(current_position)` (Reversal/Entry).
    -   Decrement `position_ttl` every step.
    -   If `position_ttl <= 0`: Force `target_position = 0`.
    -   **Cooldown**: If TTL expires, enforce `cooldown_steps` where no new entry is allowed.
2.  **Target Transition**:
    -   Calculate `delta = target_position - current_position`.
    -   **Deadband**: If `abs(delta) < min_delta`, treat as 0 change to avoid churn.
    -   **Clipping**: `delta = clip(delta, -max_delta_per_step, max_delta_per_step)` to prevent instant full reversals.
3.  **Cost Calculation**:
    -   **Fee**: Use `ztb.utils.fee_model.ExchangeFeeModel` (same instance as Gate).
    -   **Slippage**: `slippage_cost = impact_proxy * volatility_scale * trade_size`. Ensure units match PnL (JPY).
4.  **Reward**: Calculate using `compute_hft_reward`.
5.  **Risk Management**:
    -   `max_position`: Hard limit on position size.
    -   `drawdown_limit`: Terminate episode if drawdown exceeds threshold.

#### 4. Performance Optimization
-   **Data Access**: Convert DataFrame columns to Numpy arrays (`float32`) at initialization to avoid `iloc` overhead in `step()`.

## 3. Integration & Testing

### A. Unit Test (`tests/unit/environment/test_fast_intraday_env.py`)
- Verify Action Space mapping and TTL logic (entry vs update).
- Verify `OnlineScaler` updates and transforms correctly (check for future leakage).
- Verify Reward calculation returns reasonable values.
- Verify Risk Management (drawdown kill switch).
- **Reproducibility**: Verify `reset(seed=X)` produces identical trajectories.

### B. Integration Script (`scripts/v455/test_hft_env.py`)
- Load `data/btc_jpy_1m_v454.csv`.
- Apply `add_hft_features`.
- Initialize `FastIntradayEnv`.
- Run a short episode with random actions.
- Plot/Log results to verify behavior (TTL expiration, Cost accumulation).

### C. Training & Tuning Scripts
-   **Training**: `scripts/v455/train_hft.py` - Standard SAC training loop using `FastIntradayEnv`.
-   **Tuning**: `scripts/v455/tune_hft.py` - Optuna-based hyperparameter optimization for Reward params (alpha, beta) and SAC params.

## 4. Execution Steps
1.  Create Feature Engineering module (`ztb/features/hft_proxies.py`).
2.  Create Reward Function module (`ztb/trading/rewards/fast_intraday.py`).
3.  Create Environment class (`ztb/trading/environment/fast_intraday_env.py`) integrating `OnlineScaler` and `ExchangeFeeModel`.
4.  Create and run Unit Tests.
5.  Create and run Integration Script.
6.  Create Training Script (`scripts/v455/train_hft.py`).
7.  Create Tuning Script (`scripts/v455/tune_hft.py`).
