from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.components.behavioral_penalty_calculator import (
    BehavioralPenaltyCalculator,
)


def test_behavioral_penalty_merges_to_reward_settings():
    cfg = EnvironmentConfig.from_dict(
        {
            "environment": {
                "behavioral_penalty": {
                    "skewness_penalty_enabled": True,
                    "skewness_penalty_value": 0.2,
                    "skewness_penalty_tolerance": 0.01,
                }
            }
        }
    )

    # When passed via environment.behavioral_penalty, the env config should pick them up
    assert cfg.reward_settings is not None
    # Use a calculator to ensure settings are applied
    calc = BehavioralPenaltyCalculator(cfg)
    assert calc.skewness_penalty_enabled
    assert calc.skewness_penalty_value == 0.2