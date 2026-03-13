import sys

sys.path.append(".")
import numpy as np

from ztb.trading.environment.components.action_validator import ActionValidator
from ztb.trading.environment.utils.config import EnvironmentConfig

# Test ActionValidator with short position
config = EnvironmentConfig.from_dict(
    {"max_position_size": 1.0, "transaction_cost": 0.0, "exchange": "coincheck"}
)

validator = ActionValidator(config, initial_portfolio_value=200000.0)

# Test case: short position, should allow BUY to close
position = -0.018
current_price = 7100000.0
total_pnl = 142588.33 - 200000.0  # portfolio_value - initial_portfolio_value

# Create sufficient price history like in the unit tests
price_history = np.full(150, current_price)

legal_actions = validator.get_legal_actions(
    current_step=100,
    position=position,
    total_pnl=total_pnl,
    trades_count=0,
    last_trade_step=None,
    consecutive_trade_steps=0,
    close_array=price_history,
    price_array=price_history,
    df=None,
)

print(f"Short position test: position={position}, legal_actions={legal_actions}")
print("BUY allowed:", bool(legal_actions[1]))
print("SELL allowed:", bool(legal_actions[2]))
print("HOLD allowed:", bool(legal_actions[0]))
