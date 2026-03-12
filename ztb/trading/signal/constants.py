"""
Signal guidance constants to centralize parity and thresholds.
This file provides canonical constants for the sign conventions used across
signal scoring, guidance systems and backtests to reduce future confusion.
"""

# If True, HIGH SCORE -> BUY in the signal quality scorer (default behaviour)
HIGH_SCORE_IS_BUY = True

# For backtests that historically treated high score as SELL, set this flag
BACKTEST_HIGH_SCORE_IS_SELL = False

# Default fallback conversion threshold used by SignalGuidanceSystem
DEFAULT_FALLBACK_THRESHOLD = 0.2

# Default thresholds used by SignalQualityScorer (0-100 scale)
DEFAULT_BUY_THRESHOLD = 75
DEFAULT_SELL_THRESHOLD = 25
DEFAULT_HOLD_THRESHOLD = 45

# Scaling for continuous action to score (continuous_action -1..1 -> 0..100)
CONTINUOUS_TO_SCORE_SCALE = 50
