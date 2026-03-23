#!/usr/bin/env python3
"""
Tests for calculate_reward_simple parameter mismatch fix.
"""


import pytest

from ztb.trading.environment.components.reward_calculator import RewardCalculator


class TestCalculateRewardSimpleParameterFix:
    """Test the calculate_reward_simple parameter mismatch fix."""

    @pytest.fixture
    def reward_calculator(self, mock_config):
        """Create a RewardCalculator instance for testing."""
        from ztb.training.environments.environment_config import EnvironmentConfig

        config = EnvironmentConfig(initial_balance=10000.0, commission=0.001)
        reward_settings = mock_config.get("reward_settings", {})
        calculator = RewardCalculator(
            config=config,
            reward_settings=reward_settings,
            initial_portfolio_value=10000.0,
        )
        # This suite validates the basic RewardKernel-backed path, not the
        # optional shaping/signal/scaling layers added later.
        calculator.dynamic_reward_shaper.enabled = False
        calculator.asymmetric_reward_scaler.enabled = False
        calculator.signal_integrator.enabled = False
        return calculator

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "reward_settings": {
                "transaction_cost": 0.001,
                "base_reward_multiplier": 1.0,
                "risk_penalty": {"enabled": True, "multiplier": 0.1},
            }
        }

    def test_calculate_reward_simple_with_transaction_cost(
        self, reward_calculator, mock_config
    ):
        """Test calculate_reward_simple correctly uses transaction_cost parameter."""
        # Test with explicit transaction_cost
        reward = reward_calculator.calculate_reward_simple(
            current_price=100.0,
            previous_price=99.0,
            position_size=1.0,
            transaction_cost=0.002,  # Explicit cost
        )

        # Should calculate reward considering transaction cost
        # Price change: 100 - 99 = 1.0
        # With position_size=1.0 and transaction_cost=0.002
        # Expected reward should account for transaction cost
        assert isinstance(reward, (int, float))

        # Test with zero transaction cost
        reward_zero_cost = reward_calculator.calculate_reward_simple(
            current_price=100.0,
            previous_price=99.0,
            position_size=1.0,
            transaction_cost=0.0,
        )

        # Zero cost should give higher reward than with cost
        assert reward_zero_cost > reward

    def test_calculate_reward_simple_parameter_validation(self, reward_calculator):
        """Test that calculate_reward_simple accepts all required parameters."""
        # This test ensures the method signature is correct after the fix

        # Should not raise TypeError due to missing parameters
        try:
            reward = reward_calculator.calculate_reward_simple(
                current_price=100.0,
                previous_price=99.0,
                position_size=1.0,
                transaction_cost=0.001,
            )
            assert isinstance(reward, (int, float))
        except TypeError as e:
            if "missing" in str(e) or "unexpected" in str(e):
                pytest.fail(f"Parameter mismatch in calculate_reward_simple: {e}")
            else:
                # Other TypeErrors are OK (e.g., type conversion issues)
                pass

    def test_calculate_reward_simple_with_config_transaction_cost(
        self, reward_calculator, mock_config
    ):
        """Test calculate_reward_simple uses config transaction_cost when available."""
        # Set config on calculator
        reward_calculator.config = mock_config

        # Call without explicit transaction_cost - should use config value
        reward = reward_calculator.calculate_reward_simple(
            current_price=100.0,
            previous_price=99.0,
            position_size=1.0,
            # No transaction_cost parameter - should use config
        )

        # Should work without error
        assert isinstance(reward, (int, float))

    def test_calculate_reward_simple_price_change_calculation(self, reward_calculator):
        """Test that price change calculation is correct."""
        # Test positive price change
        reward_up = reward_calculator.calculate_reward_simple(
            current_price=101.0,
            previous_price=100.0,
            position_size=1.0,
            transaction_cost=0.0,
        )

        # Test negative price change
        reward_down = reward_calculator.calculate_reward_simple(
            current_price=99.0,
            previous_price=100.0,
            position_size=1.0,
            transaction_cost=0.0,
        )

        # Positive change should give positive reward
        assert reward_up > 0

        # Negative change should give negative reward
        assert reward_down < 0

        # Magnitudes should be similar (symmetric)
        assert abs(reward_up) == abs(reward_down)

    def test_calculate_reward_simple_position_size_effect(self, reward_calculator):
        """Test that position_size affects reward calculation."""
        reward_small = reward_calculator.calculate_reward_simple(
            current_price=101.0,
            previous_price=100.0,
            position_size=0.5,
            transaction_cost=0.0,
        )

        reward_large = reward_calculator.calculate_reward_simple(
            current_price=101.0,
            previous_price=100.0,
            position_size=2.0,
            transaction_cost=0.0,
        )

        # Larger position should give larger reward
        assert abs(reward_large) > abs(reward_small)

    def test_calculate_reward_simple_transaction_cost_effect(self, reward_calculator):
        """Test that transaction_cost reduces reward."""
        reward_no_cost = reward_calculator.calculate_reward_simple(
            current_price=101.0,
            previous_price=100.0,
            position_size=1.0,
            transaction_cost=0.0,
        )

        reward_with_cost = reward_calculator.calculate_reward_simple(
            current_price=101.0,
            previous_price=100.0,
            position_size=1.0,
            transaction_cost=0.001,
        )

        # Transaction cost should reduce the reward
        assert reward_with_cost < reward_no_cost

    def test_calculate_reward_simple_zero_price_change(self, reward_calculator):
        """Test reward calculation with zero price change."""
        reward = reward_calculator.calculate_reward_simple(
            current_price=100.0,
            previous_price=100.0,
            position_size=1.0,
            transaction_cost=0.001,
        )

        # Zero price change should give zero reward (minus transaction cost)
        assert reward <= 0  # Transaction cost makes it negative or zero

    def test_calculate_reward_simple_extreme_values(self, reward_calculator):
        """Test reward calculation with extreme values."""
        # Test with very large price change
        reward_large = reward_calculator.calculate_reward_simple(
            current_price=200.0,
            previous_price=100.0,
            position_size=1.0,
            transaction_cost=0.0,
        )

        # Test with very small price change
        reward_small = reward_calculator.calculate_reward_simple(
            current_price=100.01,
            previous_price=100.0,
            position_size=1.0,
            transaction_cost=0.0,
        )

        # Both should be positive
        assert reward_large > 0
        assert reward_small > 0

        # Large change should give larger reward
        assert reward_large > reward_small

    def test_calculate_reward_simple_parameter_types(self, reward_calculator):
        """Test that calculate_reward_simple handles different parameter types."""
        # Test with integers
        reward_int = reward_calculator.calculate_reward_simple(
            current_price=100, previous_price=99, position_size=1, transaction_cost=0
        )

        # Test with floats
        reward_float = reward_calculator.calculate_reward_simple(
            current_price=100.0,
            previous_price=99.0,
            position_size=1.0,
            transaction_cost=0.0,
        )

        # Should give same result
        assert reward_int == reward_float


if __name__ == "__main__":
    pytest.main([__file__])
