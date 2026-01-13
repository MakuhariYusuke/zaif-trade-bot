import unittest
from unittest.mock import MagicMock

from ztb.trading.environment.components.rewards.base import RewardContext
from ztb.trading.environment.components.rewards.confidence_penalty import (
    ConfidencePenaltyReward,
)


class TestConfidencePenaltyReward(unittest.TestCase):
    def setUp(self) -> None:
        self.reward_component = ConfidencePenaltyReward()
        self.mock_settings = {
            "confidence_penalty_threshold": 0.1,
            "confidence_penalty_factor": 2.0,
        }

    def create_context(
        self,
        pnl: float,
        continuous_action_value: float,
        atr: float = 100.0,
        atr_normalised: float = 0.0,
        portfolio_return: float = 0.0,
    ) -> RewardContext:
        # Mock config and settings
        config = MagicMock()
        settings = MagicMock()
        settings.confidence_penalty_threshold = 0.1
        settings.confidence_penalty_factor = 2.0
        settings.get = lambda k: getattr(settings, k, None)

        return RewardContext(
            action=0,
            atr_normalised=atr_normalised,
            portfolio_return=portfolio_return,
            position=0.0,
            effective_max_position=1.0,
            current_price=1000.0,
            atr=atr,
            pnl=pnl,
            reward_scaling=1.0,
            observation=None,
            step=1,
            portfolio_value=10000.0,
            transaction_cost=0.0,
            old_position=0.0,
            reward_history=[],
            portfolio_value_history=[],
            config=config,
            reward_settings=settings,
            continuous_action_value=continuous_action_value,
        )

    def test_no_penalty_on_profit(self) -> None:
        # PnL > 0, High confidence
        context = self.create_context(
            pnl=50.0, continuous_action_value=0.9, atr_normalised=0.5
        )
        penalty = self.reward_component.calculate(context)
        self.assertEqual(penalty, 0.0)

    def test_no_penalty_below_threshold(self) -> None:
        # PnL < 0, Low confidence (<= 0.1)
        context = self.create_context(
            pnl=-50.0, continuous_action_value=0.1, atr_normalised=-0.5
        )
        penalty = self.reward_component.calculate(context)
        self.assertEqual(penalty, 0.0)

    def test_hinge_penalty_calculation(self) -> None:
        # PnL < 0, High confidence (0.5 > 0.1)
        # Threshold = 0.1, Factor = 2.0
        # Excess = 0.5 - 0.1 = 0.4
        # Loss Magnitude = abs(-0.5) = 0.5
        # Expected Penalty = -1.0 * 0.5 * 0.4 * 2.0 = -0.4
        
        context = self.create_context(
            pnl=-50.0, continuous_action_value=0.5, atr_normalised=-0.5
        )
        penalty = self.reward_component.calculate(context)
        self.assertAlmostEqual(penalty, -0.4)

    def test_fallback_loss_magnitude(self) -> None:
        # ATR is 0 or 1.0 (unreliable), use portfolio_return
        # Portfolio Return = -0.01 (-1%)
        # Loss Magnitude = abs(-0.01) * 100 = 1.0
        # Action = 0.6, Threshold = 0.1 -> Excess = 0.5
        # Factor = 2.0
        # Expected Penalty = -1.0 * 1.0 * 0.5 * 2.0 = -1.0
        
        context = self.create_context(
            pnl=-100.0,
            continuous_action_value=0.6,
            atr=1.0,  # Unreliable ATR
            atr_normalised=0.0,
            portfolio_return=-0.01,
        )
        penalty = self.reward_component.calculate(context)
        self.assertAlmostEqual(penalty, -1.0)

    def test_default_settings(self) -> None:
        # Test with empty settings (should use defaults: threshold=0.05, factor=1.0)
        self.mock_settings = {}
        
        # Action = 0.15, Threshold = 0.05 -> Excess = 0.1
        # Loss Magnitude = 1.0
        # Factor = 1.0
        # Expected Penalty = -1.0 * 1.0 * 0.1 * 1.0 = -0.1
        
        context = self.create_context(
            pnl=-100.0, continuous_action_value=0.15, atr_normalised=-1.0
        )
        penalty = self.reward_component.calculate(context)
        self.assertAlmostEqual(penalty, -0.1)

if __name__ == "__main__":
    unittest.main()
