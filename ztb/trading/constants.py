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
SAC_CONTINUOUS_THRESHOLD = 0.33  # Threshold for converting continuous actions to discrete
SAC_CONTINUOUS_THRESHOLD_NEG = -0.33  # Negative threshold for SELL action


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


def normalize_action(action: int) -> int:
    """
    Normalize action value to current standard.
    
    Legacy support: converts old ACTION_SELL=2 to new ACTION_SELL=-1.
    
    Args:
        action: Action value (may be legacy format)
        
    Returns:
        Normalized action value
    """
    if action == 2:  # Legacy ACTION_SELL
        return ACTION_SELL
    return action
