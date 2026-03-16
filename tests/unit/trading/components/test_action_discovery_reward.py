import numpy as np

from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


def _make_calculator(curriculum_stage: str) -> RewardCalculator:
    config = EnvironmentConfig.from_dict(
        {
            "curriculum_stage": curriculum_stage,
            "curriculum_learning": {"enabled": False},
        }
    )
    reward_settings = RewardSettings()
    calc = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=100000.0
    )
    return calc


def test_action_discovery_positive_pnl_rewards_more():
    calc = _make_calculator("action_discovery")

    # Simulate taking an action with positive PnL
    reward = calc.calculate_reward(
        action=1,
        continuous_action_value=0.05,
        current_price=100.0,
        position=0.0,
        portfolio_value=100000.0,
        atr=1.0,
        transaction_cost=0.01,
        reward_scaling=1.0,
        pnl=0.1,
        old_position=0.0,
        step=10,
        observation=np.array([1.0, 2.0]),
        reward_history=[],
        portfolio_value_history=[100000.0],
    )
    assert reward > 0.0


def test_action_discovery_negative_pnl_penalty():
    calc = _make_calculator("action_discovery")

    # Negative pnl should be penalized but less than a normal pnl-focused penalty
    reward_neg = calc.calculate_reward(
        action=1,
        continuous_action_value=0.05,
        current_price=100.0,
        position=0.0,
        portfolio_value=100000.0,
        atr=1.0,
        transaction_cost=0.01,
        reward_scaling=1.0,
        pnl=-0.1,
        old_position=0.0,
        step=10,
        observation=np.array([1.0, 2.0]),
        reward_history=[],
        portfolio_value_history=[100000.0],
    )
    assert reward_neg < 0.0
