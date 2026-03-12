import json

from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.v4xx_config_converter import V4XXConfigConverter


def test_env_config_from_config_file_sets_continuous_action(tmp_path):
    """Converted config should preserve continuous action flags."""
    config_path = tmp_path / "sac_v444_test_config.json"
    config_path.write_text(
        json.dumps(
            {
                "training": {
                    "environment": {
                        "config": {
                            "initial_balance": 200000.0,
                            "commission": 0.001,
                            "action_space_type": "continuous",
                            "use_continuous_actions": True,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    converted = V4XXConfigConverter.load_and_convert_config(str(config_path))
    env_section = converted.get("training", {}).get("environment", {})
    actual_env_config = env_section.get("config", env_section)
    env_config_obj = EnvironmentConfig.from_dict(actual_env_config)

    assert hasattr(env_config_obj, "use_continuous_actions")
    assert env_config_obj.use_continuous_actions is True
