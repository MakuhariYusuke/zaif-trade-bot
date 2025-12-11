"""Trading action constants.

This module defines constants for trading actions used throughout the system.
Using constants instead of magic numbers improves code readability and maintainability.
"""

# Trading action constants
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = -1

# For convenience, export as a tuple
ALL_ACTIONS = (ACTION_HOLD, ACTION_BUY, ACTION_SELL)

# Action names for display/logging
ACTION_NAMES = {
    ACTION_HOLD: "HOLD",
    ACTION_BUY: "BUY",
    ACTION_SELL: "SELL",
}

# Array indices for profit_bonus_multipliers [BUY, SELL, HOLD]
# CRITICAL: The order is [BUY, SELL, HOLD], NOT [HOLD, BUY, SELL]!
MULTIPLIER_INDEX_BUY = 0
MULTIPLIER_INDEX_SELL = 1
MULTIPLIER_INDEX_HOLD = 2

# SAC continuous action discretization thresholds
SAC_CONTINUOUS_THRESHOLD = (
    0.3333  # Threshold for converting continuous actions to discrete
)
SAC_CONTINUOUS_THRESHOLD_NEG = -0.3333  # Negative threshold for SELL action

# Financial constants
TRADING_DAYS_PER_YEAR = 252  # Standard number of trading days in a year


def get_action_name(action: int) -> str:
    """
    Get human-readable name for action.

    Args:
        action: Action index (0=HOLD, 1=BUY, -1=SELL, or legacy 2=SELL)

    Returns:
        Action name as a string

    Raises:
        ValueError: If action is not a valid action index
    """
    if action not in ACTION_NAMES:
        # Legacy support: ACTION_SELL was previously 2
        if action == 2:
            return "SELL"
        return f"UNKNOWN_ACTION_{action}"
    return ACTION_NAMES[action]


def get_action_count_index(action: int) -> int:
    """
    Get the index for action counts array [BUY, SELL, HOLD].

    Args:
        action: Action value (ACTION_HOLD=0, ACTION_BUY=1, ACTION_SELL=-1)

    Returns:
        Index in action counts array (0=BUY, 1=SELL, 2=HOLD)
    """
    if action == ACTION_BUY:
        return MULTIPLIER_INDEX_BUY
    elif action == ACTION_SELL:
        return MULTIPLIER_INDEX_SELL
    elif action == ACTION_HOLD:
        return MULTIPLIER_INDEX_HOLD
    else:
        return 0  # Default to BUY index


def normalize_action(action: float | int) -> int:
    """Normalize an action value to one of the discrete ACTION_* constants.

    Accepts either already-discrete (-1, 0, 1) or continuous actions in [-1,1].
    This helper is used by legacy components that expect a normalized action.
    """
    try:
        val = float(action)
    except Exception:
        return ACTION_HOLD

    if val >= SAC_CONTINUOUS_THRESHOLD:
        return ACTION_BUY
    if val <= SAC_CONTINUOUS_THRESHOLD_NEG:
        return ACTION_SELL
    return ACTION_HOLD
