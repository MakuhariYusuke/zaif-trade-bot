"""Trading action constants.

This module defines constants for trading actions used throughout the system.
Using constants instead of magic numbers improves code readability and maintainability.
"""

# Trading action constants
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2

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


def get_action_name(action: int) -> str:
    """Get the name of an action.
    
    Args:
        action: Action index (0=HOLD, 1=BUY, 2=SELL)
        
    Returns:
        Action name as a string
        
    Raises:
        ValueError: If action is not a valid action index
    """
    if action not in ACTION_NAMES:
        raise ValueError(f"Invalid action: {action}. Must be one of {ALL_ACTIONS}")
    return ACTION_NAMES[action]
