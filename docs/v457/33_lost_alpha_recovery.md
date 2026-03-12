# Lost Alpha Recovery Report (v457.5)

## Overview
Following the stabilization of seed performance via Curriculum Learning (v457.4), a deep audit of the codebase revealed several "proposals" from v450-v456 that were either partially implemented or completely broken. This "Lost Alpha" has been restored in the environment.

## Restored Features

### 1. Cyclical Time Features (6 Dimensions)
- **Status Before**: Features `hour_sin`, `hour_cos`, etc. were defined in `ztb/features/time/cyclical.py` but the Environment (`FastIntradayEnvV456`) was **zero-filling** them (Placeholder Implementation).
- **Fix**: Modified `FastIntradayEnvV456` to pre-calculate these values in `__init__` using the DataFrame index.
- **Impact**: The agent now explicitly knows the "Time of Day" and "Day of Week". This allows it to learn intraday volatility patterns (e.g., "Don't open longs at 3 AM low liquidity").

### 2. Multi-Timeframe (MTF) Context (27 Dimensions)
- **Status Before**: The Factory (`factory_v456.py`) was generating "MTF" columns (e.g., `mtf_rsi_5m`) by applying indicators to **1-minute** data without resampling.
    - Result: `rsi_5m` was identical to `rsi_15m` and `rsi_1h` (and often identical to Base `rsi_14`).
    - Consequence: 27 dimensions of the observation space were redundant noise.
- **Fix**: Modified `EnvironmentFactory.calculate_mtf_features` to implement proper Pandas `resample().agg()` logic.
    - Indicators are now calculated on true 5m/15m/1h candles.
    - Values are forward-filled back to the 1m timeline.
- **Impact**: The agent can now see the "Bigger Picture". It can distinguish a "dip in an uptrend" (1h RSI > 60, 1m RSI < 30) from a "crash" (1h RSI < 30).

## Technical Verification
A verification script (`test_fix_verification.py`) confirmed:
1.  **MTF Integrity**: `mean(abs(rsi_5m - rsi_15m))` > 9.0 (Previously ~0.0).
2.  **Cyclical Integrity**: Arrays are populated with sin/cos values instead of zeros.

## Next Steps
- Re-run the Seed Stability Test ( Seeds 42, 123, 777) with these features active.
- Expect higher convergence rate and better generalization due to reduced observation noise and added context.
