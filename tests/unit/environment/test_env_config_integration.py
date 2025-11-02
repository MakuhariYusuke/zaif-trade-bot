from pathlib import Path

from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.v4xx_config_converter import V4XXConfigConverter


def test_env_config_from_config_file_sets_continuous_action(tmp_path):
    """Load the sample test config and ensure use_continuous_actions is preserved/recognized."""
    # Find repository root by walking parents until we find 'config/sac_v444_test_config.json'
    p = Path(__file__).resolve()
    config_path = None
    for parent in p.parents:
        candidate = parent / "config" / "sac_v444_test_config.json"
        if candidate.exists():
            config_path = candidate
            break
    assert config_path is not None, "Test config not found in repository parents"

    # Load and convert using converter (mimic the real loading flow)
    converted = V4XXConfigConverter.load_and_convert_config(str(config_path))

    # Extract environment configuration portion similar to trainer flow
    env_section = converted.get("training", {}).get("environment", {})
    actual_env_config = env_section.get("config", env_section)

    # Convert to EnvironmentConfig dataclass
    env_config_obj = EnvironmentConfig.from_dict(actual_env_config)

    assert hasattr(
        env_config_obj, "use_continuous_actions"
    ), "EnvironmentConfig missing attribute"
    assert (
        env_config_obj.use_continuous_actions is True
    ), "Expected use_continuous_actions=True from test config"
