import sys
from unittest.mock import MagicMock

# Mock torch before importing anything else
sys.modules["torch"] = MagicMock()

import unittest

import numpy as np
import pandas as pd

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


class TestBankruptcyAndDrawdown(unittest.TestCase):
    def setUp(self):
        # Create a real dataframe
        self.df = pd.DataFrame(
            {
                "close": np.random.rand(100) * 100,
                "open": np.random.rand(100) * 100,
                "high": np.random.rand(100) * 100,
                "low": np.random.rand(100) * 100,
                "volume": np.random.rand(100) * 1000,
            }
        )

        # Create config with bankruptcy and drawdown settings
        self.config = EnvironmentConfig(
            initial_portfolio_value=100000.0,
            bankruptcy_threshold=2000.0,
            bankruptcy_penalty=1000.0,
            drawdown_penalty_threshold=0.20,
            drawdown_penalty_factor=0.1,
            reward_scaling=1.0,
            transaction_cost=0.0,
            max_steps=100,
            feature_set="minimal",  # Use minimal feature set to speed up init
        )

        # Mock dependencies
        # We need to patch HeavyTradingEnv methods that use heavy computation or external resources
        # But first we init it with real DF to pass type validation

        # To avoid full initialization overhead, we can mock the mixins or just let it run if it's fast enough.
        # Given the complexity, let's try to instantiate it and then mock the components we need for the test.

        self.env = HeavyTradingEnv(df=self.df, config=self.config)

        # Mock internal components to bypass complex logic during step
        self.env.data_manager = MagicMock()
        self.env.data_manager.get_price_at_step.return_value = 100.0
        self.env.data_manager.get_atr_at_step.return_value = 1.0
        self.env.data_manager.is_episode_boundary.return_value = False
        self.env.data_manager.ensure_data_available = MagicMock()

        self.env.position_manager = MagicMock()
        self.env.position_manager.position = 0.0
        self.env.position_manager.execute_action.return_value = 0.0
        self.env.position_manager.calculate_unrealized_pnl.return_value = 0.0
        self.env.position_manager.close_position.return_value = 0.0
        self.env.position_manager.get_position_info.return_value = {
            "position": 0.0,
            "entry_price": 0.0,
            "realized_pnl": 0.0,
            "total_pnl": 0.0,
            "trades_count": 0,
        }

        self.env.reward_calculator = MagicMock()
        self.env.reward_calculator.calculate_reward.return_value = 0.0
        self.env.reward_calculator.trend_detector = None
        self.env.reward_calculator.curriculum_manager = None
        self.env.reward_calculator.get_last_reward_components.return_value = {}

        self.env.observation_builder = MagicMock()
        self.env.observation_builder.get_observation.return_value = np.zeros(10)
        self.env.observation_builder.get_info.return_value = {}

        self.env.action_executor = MagicMock()
        self.env.action_executor.convert_and_validate_action.return_value = (0, 0.0)

        self.env.validation_manager = MagicMock()
        self.env.validation_manager.validate_action.return_value = 0
        self.env.validation_manager.validate_reward_calculation.side_effect = (
            lambda x: x
        )

        self.env.state_manager = MagicMock()
        self.env.statistics_calculator = MagicMock()
        self.env.memory_manager = MagicMock()
        self.env.memory_manager.should_collect_garbage = False
        self.env.memory_manager.should_log_memory.return_value = False

        # Initialize env state
        self.env.current_step = 0
        self.env.n_steps = 100
        self.env.portfolio_value = 100000.0
        self.env.initial_portfolio_value = 100000.0
        self.env.total_pnl = 0.0
        self.env.realized_pnl = 0.0
        self.env._prev_unrealized_pnl = 0.0
        self.env.position = 0.0
        self.env.entry_price = 0.0
        self.env.trades_count = 0

        # Ensure these are floats, not Mocks
        self.env.position = 0.0
        self.env.entry_price = 0.0

        self.env.pnl_history = []
        self.env.position_abs_history = []
        self.env.portfolio_value_history = []
        self.env.reward_history = []
        self.env.position_history = []
        self.env.regime_classifier = None

    def test_bankruptcy_trigger(self):
        # Simulate portfolio value dropping below threshold
        # Initial PV = 100,000. Bankruptcy Threshold = 2,000.
        # Need realized_pnl to be approx -98,500.

        target_pv = 1500.0
        realized_pnl = target_pv - 100000.0

        # Update mock return value for sync
        self.env.position_manager.get_position_info.return_value = {
            "position": 0.0,
            "entry_price": 0.0,
            "realized_pnl": realized_pnl,
            "total_pnl": realized_pnl,
            "trades_count": 0,
        }

        # Also set env attributes directly just in case sync happens later or earlier
        self.env.realized_pnl = realized_pnl
        self.env.portfolio_value = target_pv

        # Step
        _, reward, done, _, info = self.env.step(0)

        # Check bankruptcy
        self.assertTrue(
            done, f"Episode should be done. PV={info.get('portfolio_value')}"
        )
        self.assertTrue(info.get("bankruptcy"))
        # Reward should include bankruptcy penalty (-1000)
        self.assertLessEqual(reward, -1000.0)

    def test_drawdown_penalty(self):
        # Simulate 30% drawdown (Threshold is 20%)
        # PV = 70,000 (30k loss)
        target_pv = 70000.0
        realized_pnl = target_pv - 100000.0

        # Update mock return value for sync
        self.env.position_manager.get_position_info.return_value = {
            "position": 0.0,
            "entry_price": 0.0,
            "realized_pnl": realized_pnl,
            "total_pnl": realized_pnl,
            "trades_count": 0,
        }

        self.env.realized_pnl = realized_pnl
        self.env.portfolio_value = target_pv

        # Step
        _, reward, done, _, info = self.env.step(0)

        # Check drawdown penalty
        # Excess drawdown = 0.30 - 0.20 = 0.10
        # Penalty = 0.10 * 0.1 (factor) * 1.0 (scaling) = 0.01
        # Note: Base reward is 0.0 from mock

        self.assertIn("drawdown_penalty", info)
        expected_penalty = (0.30 - 0.20) * 0.1
        self.assertAlmostEqual(info["drawdown_penalty"], expected_penalty, places=5)
        self.assertAlmostEqual(reward, -expected_penalty, places=5)
        self.assertFalse(done)  # Should not be done yet

    def test_no_penalty_within_limits(self):
        # Simulate 10% drawdown (Threshold is 20%)
        self.env.portfolio_value = 90000.0
        self.env.realized_pnl = -10000.0

        # Step
        _, reward, done, _, info = self.env.step(0)

        self.assertNotIn("drawdown_penalty", info)
        self.assertNotIn("bankruptcy", info)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)


if __name__ == "__main__":
    unittest.main()
