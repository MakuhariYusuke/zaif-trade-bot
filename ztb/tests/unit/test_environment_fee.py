"""
Unit tests for environment dynamic fee configuration
環境の動的fee設定の単体テスト
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ztb.trading.environment.environment import HeavyTradingEnv


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [102, 103, 104],
        'volume': [1000, 1100, 1200]
    })


class TestEnvironmentDynamicFee:
    """Test dynamic fee configuration in HeavyTradingEnv"""

    def test_default_exchange_coincheck(self, sample_df):
        """Test default exchange is coincheck"""
        config = {}
        env = HeavyTradingEnv(df=sample_df, config=config)
        assert env.config["exchange"] == "coincheck"

    def test_custom_exchange_bitflyer(self, sample_df):
        """Test setting custom exchange to bitflyer"""
        config = {"exchange": "bitflyer"}
        env = HeavyTradingEnv(df=sample_df, config=config)
        assert env.config["exchange"] == "bitflyer"
        assert env.fee_model.current_exchange == "bitflyer"

    def test_custom_exchange_binance(self, sample_df):
        """Test setting custom exchange to binance"""
        config = {"exchange": "binance"}
        env = HeavyTradingEnv(df=sample_df, config=config)
        assert env.config["exchange"] == "binance"
        assert env.fee_model.current_exchange == "binance"

    def test_fee_model_initialization(self, sample_df):
        """Test that ExchangeFeeModel is properly initialized"""
        config = {}
        env = HeavyTradingEnv(df=sample_df, config=config)
        assert hasattr(env, 'fee_model')
        assert env.fee_model.current_exchange == "coincheck"

    def test_transaction_cost_coincheck_zero(self, sample_df):
        """Test transaction cost is zero for coincheck"""
        config = {"exchange": "coincheck"}
        env = HeavyTradingEnv(df=sample_df, config=config)
        assert env.config["transaction_cost"] == 0.0

    def test_transaction_cost_bitflyer_001(self, sample_df):
        """Test transaction cost is 0.1% for bitflyer"""
        config = {"exchange": "bitflyer"}
        env = HeavyTradingEnv(df=sample_df, config=config)
        assert env.config["transaction_cost"] == 0.001

    def test_transaction_cost_binance_001(self, sample_df):
        """Test transaction cost is 0.1% for binance"""
        config = {"exchange": "binance"}
        env = HeavyTradingEnv(df=sample_df, config=config)
        assert env.config["transaction_cost"] == 0.001

    def test_transaction_cost_uses_buy_rate(self, sample_df):
        """Test that transaction_cost uses buy rate by default"""
        config = {"exchange": "bitflyer"}
        env = HeavyTradingEnv(df=sample_df, config=config)

        # Verify that the transaction cost matches the buy rate
        buy_rate = env.fee_model.get_fee_rate("buy")
        assert env.config["transaction_cost"] == buy_rate

    def test_exchange_change_after_initialization(self, sample_df):
        """Test changing exchange after environment initialization"""
        config = {"exchange": "coincheck"}
        env = HeavyTradingEnv(df=sample_df, config=config)

        # Initially coincheck
        assert env.fee_model.current_exchange == "coincheck"
        assert env.config["transaction_cost"] == 0.0

        # Change to bitflyer
        env.fee_model.set_exchange("bitflyer")
        env.config["transaction_cost"] = env.fee_model.get_fee_rate("buy")

        assert env.fee_model.current_exchange == "bitflyer"
        assert env.config["transaction_cost"] == 0.001

    def test_invalid_exchange_fallback(self, sample_df):
        """Test behavior with invalid exchange (should not crash)"""
        config = {"exchange": "invalid_exchange"}
        env = HeavyTradingEnv(df=sample_df, config=config)
        # Should not crash, fee_model handles invalid exchanges gracefully
        # Invalid exchange falls back to default 'binance'
        assert env.fee_model.current_exchange == "binance"
        # Will use binance rate
        assert env.config["transaction_cost"] == env.fee_model.get_fee_rate("buy")

    @patch('ztb.trading.environment.environment.ExchangeFeeModel')
    def test_fee_model_mock(self, mock_fee_model_class: MagicMock, sample_df):
        """Test with mocked ExchangeFeeModel"""
        mock_fee_model = MagicMock()
        mock_fee_model.get_fee_rate.return_value = 0.002
        mock_fee_model.current_exchange = "mock_exchange"
        mock_fee_model_class.return_value = mock_fee_model

        config = {"exchange": "mock_exchange"}
        env = HeavyTradingEnv(df=sample_df, config=config)

        # Verify fee model was created and configured
        mock_fee_model_class.assert_called_once()
        mock_fee_model.set_exchange.assert_called_once_with("mock_exchange")
        mock_fee_model.get_fee_rate.assert_called_once_with("buy")

        # Verify transaction cost was set from fee model
        assert env.config["transaction_cost"] == 0.002

    def test_config_preservation(self, sample_df):
        """Test that other config values are preserved when setting fee"""
        config = {
            "exchange": "bitflyer",
            "reward_scaling": 2.0,
            "max_position_size": 0.5,
            "timeframe": "5m"
        }
        env = HeavyTradingEnv(df=sample_df, config=config)

        # Verify fee-related changes
        assert env.config["exchange"] == "bitflyer"
        assert env.config["transaction_cost"] == 0.001

        # Verify other config values are preserved
        assert env.config["reward_scaling"] == 2.0
        assert env.config["max_position_size"] == 0.5
        assert env.config["timeframe"] == "5m"

    def test_transaction_cost_inheritance(self, sample_df):
        """Test that fee_model is properly configured even when transaction_cost is manually set"""
        config = {
            "exchange": "bitflyer",
            "transaction_cost": 0.005  # Manual override (will be overwritten by fee_model)
        }
        env = HeavyTradingEnv(df=sample_df, config=config)

        # transaction_cost gets overwritten by fee_model during initialization
        assert env.config["transaction_cost"] == 0.001  # bitflyer buy rate

        # But fee model is still set up correctly for dynamic changes
        assert env.fee_model.current_exchange == "bitflyer"
        assert env.fee_model.get_fee_rate("buy") == 0.001