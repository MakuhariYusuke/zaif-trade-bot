
import unittest
from unittest.mock import MagicMock
from types import SimpleNamespace
from ztb.trading.environment.components.position_manager import PositionManager
from ztb.analysis.v444_regime_classifier import RegimeType

class TestPositionScaling(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace()
        self.config.max_position_size = 1.0
        self.config.initial_portfolio_value = 100000.0
        self.config.transaction_cost = 0.001
        self.config.allow_reverse = True
        self.config.risk_management = {}
        self.config.enforce_reverse_cooldown = False
        
        self.get_price = MagicMock(return_value=100.0)
        self.pm = PositionManager(self.config, self.get_price)
        
        # Mock risk manager to return base position
        self.pm.risk_manager.calculate_risk_adjusted_position = MagicMock(side_effect=lambda base_position, **kwargs: {
            "adjusted_position": base_position,
            "control_active": False,
            "reasons": []
        })

    def test_scaling_extreme_volatility(self):
        # Mock regime data
        regime_data = MagicMock()
        regime_data.primary_regime = RegimeType.EXTREME_VOLATILITY
        regime_data.confidence = 0.5
        
        # Expected scale: 1.0 + 0.5 = 1.5x
        # Base size 1.0 -> 1.5
        
        self.pm.open_position(1, 0, regime_data=regime_data)
        
        # Check if risk manager was called with scaled position
        call_args = self.pm.risk_manager.calculate_risk_adjusted_position.call_args
        self.assertAlmostEqual(call_args.kwargs['base_position'], 1.5)

    def test_scaling_strong_trend(self):
        # Mock regime data
        regime_data = MagicMock()
        regime_data.primary_regime = RegimeType.STRONG_BULL_TREND
        regime_data.confidence = 0.8
        
        # Expected scale: 1.0 + (0.8 * 0.5) = 1.4x
        
        self.pm.open_position(1, 0, regime_data=regime_data)
        
        # Check call args
        call_args = self.pm.risk_manager.calculate_risk_adjusted_position.call_args
        self.assertAlmostEqual(call_args.kwargs['base_position'], 1.4)

    def test_no_scaling_normal(self):
        # Mock regime data
        regime_data = MagicMock()
        regime_data.primary_regime = RegimeType.LOW_VOLATILITY_RANGING
        regime_data.confidence = 0.9
        
        # Expected scale: 1.0x
        
        self.pm.open_position(1, 0, regime_data=regime_data)
        
        call_args = self.pm.risk_manager.calculate_risk_adjusted_position.call_args
        self.assertAlmostEqual(call_args.kwargs['base_position'], 1.0)

if __name__ == '__main__':
    unittest.main()
