from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings

config = EnvironmentConfig.from_dict(
    {
        "max_position_size": 1.0,
        "transaction_cost": 0.001,
        "exchange": "coincheck",
        "reward_scaling": 1.0,
        "action_space_type": "continuous",
        "use_continuous_actions": True,
        "feature_set": "minimal",
        "enable_action_masking": True,
        "use_standardized_observations": True,
        "random_start": True,
        "continuous_to_discrete_threshold": 0.08,
        "behavior_optimization": {
            "action_balance_target": 0.333,
            "entropy_regularization": 0.01,
            "action_smoothing": 0.1,
            "consistency_penalty": 0.05,
            "balance_penalty": 0.1,
            "balance_penalty_min_actions": 1,
            "redundant_trade_penalty": 5.0,
        },
        "action_bonuses": {
            "buy_action_bonus": 0.0,
            "sell_action_bonus": 0.0,
            "hold_action_bonus": 0.0,
        },
        "base_action_penalty": 0.015,
    }
)

reward_settings = RewardSettings(
    use_simple_reward=False,
    reward_scale=100.0,
    trading_bonus=0.01,
    profit_bonuses={"base": 1.5, "ultra": 2.0},
    penalty_coefficients={"loss": 2.0, "position": 0.01, "stagnation": 0.001},
    entropy_bonus=0.0,
    custom_reward_params={},
    balance_penalty=0.1,
    balance_penalty_tolerance=0.05,
    profit_weight=1.0,
    risk_weight=0.5,
    consistency_weight=0.2,
    ultra_profit_multiplier=2.0,
    ultra_risk_multiplier=0.5,
    position_soft_cap=0.5,
    position_penalty_scale=0.1,
    position_penalty_exponent=2.0,
    inventory_window=10,
    inventory_penalty_scale=0.01,
    trade_frequency_penalty=0.001,
    trade_frequency_halflife=100.0,
    trade_cooldown_steps=5,
    trade_cooldown_penalty=0.01,
    max_consecutive_trades=3,
    consecutive_trade_penalty=0.05,
    volatility_window=20,
    volatility_penalty_scale=0.01,
    sharpe_bonus_scale=0.01,
    sortino_bonus_scale=0.01,
    calmar_bonus_scale=0.005,
    reward_clip_value=10.0,
    profit_bonus_multipliers=[1.0, 1.5, 2.0],
    enable_forced_diversity=False,
)

rc = RewardCalculator(
    config=config, reward_settings=reward_settings, initial_portfolio_value=100000.0
)
print(
    "behavioral min actions",
    rc.behavioral_penalty_calculator.balance_penalty_min_actions,
)
print("behavioral config type:", type(rc.behavioral_penalty_calculator.config))
print("behavioral config repr:", rc.behavioral_penalty_calculator.config)
for action in [ACTION_BUY, ACTION_SELL, ACTION_HOLD]:
    r = rc.calculate_reward(
        action=action,
        current_price=100.0,
        position=0.0,
        portfolio_value=100000.0,
        atr=1.0,
        transaction_cost=0.001,
        reward_scaling=1.0,
        pnl=0.0,
        old_position=0.0,
        step=1,
        observation=None,
        reward_history=[],
        portfolio_value_history=[100000.0],
    )
    print("action", action, "reward", r)
    print("components", rc.get_last_reward_components())
