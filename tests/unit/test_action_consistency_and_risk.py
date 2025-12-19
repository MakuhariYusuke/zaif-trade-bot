
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from ztb.trading.constants import ACTION_SELL as CONST_ACTION_SELL
from ztb.trading.live.action_mask_provider import ACTION_SELL as MASK_ACTION_SELL
from ztb.trading.environment.components.rewards.profit_optimized import ProfitOptimizedReward
from ztb.trading.environment.components.rewards.ultra_profit import UltraProfitReward
from ztb.trading.environment.components.rewards.trading_focused import TradingFocusedReward
from ztb.trading.environment.components.rewards.pnl_focused import PnlFocusedReward
from ztb.trading.live.core.trade_executor import ACTION_SELL as EXEC_ACTION_SELL
from ztb.trading.live_trade import ACTION_SELL as LIVE_ACTION_SELL

from ztb.trading.environment.components.position_manager import PositionManager
from ztb.trading.environment.utils.config import EnvironmentConfig

class TestActionConsistency(unittest.TestCase):
    def test_action_sell_value(self):
        self.assertEqual(CONST_ACTION_SELL, -1, "ztb.trading.constants.ACTION_SELL should be -1")
        self.assertEqual(MASK_ACTION_SELL, -1, "ztb.trading.live.action_mask_provider.ACTION_SELL should be -1")
        self.assertEqual(EXEC_ACTION_SELL, -1, "ztb.trading.live.core.trade_executor.ACTION_SELL should be -1")
        self.assertEqual(LIVE_ACTION_SELL, -1, "ztb.trading.live_trade.ACTION_SELL should be -1")
        
        self.assertEqual(ProfitOptimizedReward().ACTION_SELL, -1, "ProfitOptimizedReward.ACTION_SELL should be -1")
        self.assertEqual(UltraProfitReward().ACTION_SELL, -1, "UltraProfitReward.ACTION_SELL should be -1")
        self.assertEqual(TradingFocusedReward().ACTION_SELL, -1, "TradingFocusedReward.ACTION_SELL should be -1")
        self.assertEqual(PnlFocusedReward().ACTION_SELL, -1, "PnlFocusedReward.ACTION_SELL should be -1")

class TestPositionManagerRisk(unittest.TestCase):
    def setUp(self):
        self.config = EnvironmentConfig()
        self.config.max_position_size = 1.0
        self.config.initial_portfolio_value = 100000.0
        self.config.transaction_cost = 0.001
        self.config.exchange_profile = None # Force legacy transaction cost usage
        
        self.get_price = MagicMock(return_value=10000.0)
        self.pm = PositionManager(self.config, self.get_price)
        
        # Mock RiskManager
        self.pm.risk_manager = MagicMock()
        
    def test_open_position_respects_risk_manager(self):
        # Setup RiskManager to return a small size (0.5)
        self.pm.risk_manager.calculate_risk_adjusted_position.return_value = {
            "adjusted_position": 0.5,
            "control_active": True,
            "reasons": ["Test constraint"]
        }
        
        # Setup funds to allow large size (e.g. 10.0)
        # available = 100000, price = 10000 -> affordable ~ 10
        
        # Call open_position
        # We need to mock execution_model if present, but default is None
        
        # We can't easily check internal variable 'actual_position_size' directly without mocking more stuff
        # or checking the return value (cost).
        # Cost = size * price * fee
        # If size is 0.5, cost = 0.5 * 10000 * 0.001 = 5.0
        
        cost = self.pm.open_position(1, 0)
        
        expected_cost = 0.5 * 10000.0 * 0.001
        self.assertAlmostEqual(cost, expected_cost, places=5)
        
    def test_open_position_aborts_if_too_small(self):
        # Setup RiskManager to return very small size (0.0001)
        # Min trade size is 0.001
        self.pm.risk_manager.calculate_risk_adjusted_position.return_value = {
            "adjusted_position": 0.0001,
            "control_active": True,
            "reasons": ["Test constraint"]
        }
        
        cost = self.pm.open_position(1, 0)
        
        # Should return 0.0 (aborted)
        self.assertEqual(cost, 0.0)
        
    def test_open_position_aborts_if_funds_low(self):
        # Setup RiskManager to allow 1.0
        self.pm.risk_manager.calculate_risk_adjusted_position.return_value = {
            "adjusted_position": 1.0,
            "control_active": False,
            "reasons": []
        }
        
        # Setup funds to be very low
        # We can modify realized_pnl to reduce available funds
        # Initial = 100000. We want available < min_trade_size * price
        # min_trade_size * price = 0.001 * 10000 = 10 JPY
        # So we need available < 10
        self.pm.realized_pnl = -99995.0 # Available = 5.0
        
        cost = self.pm.open_position(1, 0)
        
        # Should return 0.0 (aborted)
        self.assertEqual(cost, 0.0)

if __name__ == '__main__':
    unittest.main()
