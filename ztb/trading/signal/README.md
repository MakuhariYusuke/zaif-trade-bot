Signal Guidance and Scoring Conventions
======================================

This module centralizes the `Signal Guidance` design choices and constants to reduce
confusion about margin thresholds and scoring polarity across the project.

Key constants:
- HIGH_SCORE_IS_BUY: Boolean. If true, a higher signal quality score (0-100) implies a BUY (
  this is the default for the `SignalQualityScorer`).
- BACKTEST_HIGH_SCORE_IS_SELL: Boolean. Backtesting/legacy scripts historically used the
  opposite parity (high score == SELL) — this is retained for backward compatibility.
- DEFAULT_FALLBACK_THRESHOLD: Default threshold used in `SignalGuidanceSystem` fallback
  conversions from continuous actions (value `0.2`).
- DEFAULT_BUY_THRESHOLD / DEFAULT_SELL_THRESHOLD: defaults for the `SignalQualityScorer`.

Design intentions:
- Use the constants in `ztb/trading/signal/constants.py` to avoid hard-coded thresholds in
  the code and to make parity changes intentional and centralized.
- When changing the default scoring parity (i.e., `HIGH_SCORE_IS_BUY`), update both the
  SignalQuality scoring code and any backtests that assume a different mapping.

Implementation notes:
- We added constant-based behavior and tests to avoid fragile logic duplication across modules.
- Backtests preserve historical behavior via `BACKTEST_HIGH_SCORE_IS_SELL` flag.
