import numpy as np
from ztb.trading.environment.components.action_validator import ActionValidator
from ztb.trading.environment.utils.config import EnvironmentConfig

config = EnvironmentConfig.from_dict({
    'max_position_size': 1.0,
    'transaction_cost': 0.001,
    'exchange': 'coincheck'
})

validator = ActionValidator(config, initial_portfolio_value=200000.0)
price_history = np.full(150, 5000000.0)

legal_actions = validator.get_legal_actions(
    current_step=100,
    position=0.0,
    total_pnl=0.0,
    trades_count=0,
    last_trade_step=None,
    consecutive_trade_steps=0,
    close_array=price_history,
    price_array=price_history,
    df=None
)

print('Legal actions:', legal_actions)
print('Portfolio value:', 200000.0)
print('Current price:', 5000000.0)
print('Position size:', 1.0)
print('Transaction cost:', 0.001)
ideal_buy_cost = 1.0 * 5000000.0 * 1.001
print('Ideal buy cost:', ideal_buy_cost)
affordable_size = 200000.0 * 0.9 / (5000000.0 * 1.001)
print('Affordable size:', affordable_size)
min_affordable_value = affordable_size * 5000000.0
print('Min affordable value:', min_affordable_value)
print('BTC_MIN_UNIT:', 0.0001)
print('Min purchase amount:', 10000.0)

# Check conditions
print('Condition 1 (ideal):', 200000.0 >= ideal_buy_cost)
print('Condition 2 (affordable_size >= BTC_MIN_UNIT):', affordable_size >= 0.0001)
print('Condition 3 (min_affordable_value >= min_purchase_amount):', min_affordable_value >= 10000.0)