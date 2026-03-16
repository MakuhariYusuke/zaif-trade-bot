from __future__ import annotations

from ztb.trading.environment.components.calculators.reward_calculator import (
    RewardCalculator,
)
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


def make_reward_calculator(
    *,
    initial_portfolio_value: float = 100000.0,
    behavior_optimization: dict[str, object] | None = None,
    reward_settings: RewardSettings | None = None,
) -> RewardCalculator:
    config = EnvironmentConfig()
    config.behavior_optimization = behavior_optimization or {}
    resolved_reward_settings = reward_settings or RewardSettings()
    config.reward_settings = resolved_reward_settings
    return RewardCalculator(
        config=config,
        reward_settings=resolved_reward_settings,
        initial_portfolio_value=initial_portfolio_value,
    )
