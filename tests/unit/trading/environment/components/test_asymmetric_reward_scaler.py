
import pytest
from unittest.mock import MagicMock, PropertyMock
from ztb.trading.environment.components.asymmetric_reward_scaler import AsymmetricRewardScaler
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings

@pytest.fixture
def mock_env_config():
    """Pytest fixture for creating a mock EnvironmentConfig object."""
    env_config = MagicMock(spec=EnvironmentConfig)
    
    # Mock the nested reward_settings and its asymmetric_reward_scaling attribute
    reward_settings = MagicMock(spec=RewardSettings)
    type(env_config).reward_settings = PropertyMock(return_value=reward_settings)
    
    reward_settings.asymmetric_reward_scaling = {
        "long_position_reward_multiplier": 1.3,
        "short_position_reward_multiplier": 0.7,
        "long_position_penalty_multiplier": 0.9,
        "short_position_penalty_multiplier": 0.95,
    }
    return env_config

@pytest.fixture
def scaler(mock_env_config):
    """Pytest fixture for creating an AsymmetricRewardScaler instance."""
    return AsymmetricRewardScaler(mock_env_config)

@pytest.mark.parametrize(
    "reward, position, pnl, expected_scaled_reward",
    [
        # Long position scenarios
        (1.0, 0.5, 100, 1.3),   # Profit in long position
        (-1.0, 0.5, -100, -0.9), # Loss in long position

        # Short position scenarios
        (1.0, -0.5, 100, 0.7),  # Profit in short position
        (-1.0, -0.5, -100, -0.95),# Loss in short position

        # Neutral position scenarios
        (1.0, 0.0, 100, 1.0),    # Profit in neutral position
        (-1.0, 0.0, -100, -1.0),   # Loss in neutral position

        # Boundary value tests
        (1.0, 0.01, 100, 1.0),     # Exactly at long threshold (considered neutral)
        (1.0, 0.011, 100, 1.3),    # Just above long threshold
        (1.0, -0.01, 100, 1.0),    # Exactly at short threshold (considered neutral)
        (1.0, -0.011, 100, 0.7),   # Just below short threshold
        
        # Zero reward
        (0.0, 0.5, 100, 0.0),
        (0.0, -0.5, -100, 0.0),
    ],
)
def test_scale_reward(scaler, reward, position, pnl, expected_scaled_reward):
    """
    Test the scale_reward method with various scenarios.
    """
    scaled_reward = scaler.scale_reward(reward, position, pnl)
    assert scaled_reward == pytest.approx(expected_scaled_reward)

def test_initialization_with_custom_config():
    """
    Test that the scaler initializes correctly with custom values from config.
    """
    custom_env_config = MagicMock(spec=EnvironmentConfig)
    reward_settings = MagicMock(spec=RewardSettings)
    type(custom_env_config).reward_settings = PropertyMock(return_value=reward_settings)
    
    reward_settings.asymmetric_reward_scaling = {
        "long_position_reward_multiplier": 2.0,
        "short_position_reward_multiplier": 0.5,
        "long_position_penalty_multiplier": 0.8,
        "short_position_penalty_multiplier": 1.1,
    }

    scaler = AsymmetricRewardScaler(custom_env_config)

    assert scaler.long_pos_reward_multiplier == 2.0
    assert scaler.short_pos_reward_multiplier == 0.5
    assert scaler.long_pos_penalty_multiplier == 0.8
    assert scaler.short_pos_penalty_multiplier == 1.1
    
    # Thresholds are hardcoded for now, so they should be the default values
    assert scaler.long_pos_threshold == 0.01
    assert scaler.short_pos_threshold == -0.01

    # Test scaling with custom values
    scaled_reward = scaler.scale_reward(reward=1.0, position=0.1, pnl=100)
    assert scaled_reward == pytest.approx(2.0)
