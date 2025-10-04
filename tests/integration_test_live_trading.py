#!/usr/bin/env python3
"""
Integration tests for live trading system.

Tests end-to-end functionality including:
- Price fetching and validation
- Feature computation
- Model prediction
- Risk management
- Health monitoring
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

# Import the live trader
try:
    from live_trade import LiveTrader

    live_trade_available = True
except ImportError:
    live_trade_available = False


@pytest.mark.skipif(not live_trade_available, reason="live_trade module not available")
class TestLiveTradingIntegration:
    """Integration tests for live trading system."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "price_history_length": 50,
            "rsi_neutral_value": 50.0,
            "rsi_period": 14,
            "fallback_price": 5000000.0,
            "price_min": 1000000,
            "price_max": 50000000,
            "price_change_threshold": 0.20,
            "reward_scaling": 1.0,
            "transaction_cost": 0.001,
            "max_position_size": 0.1,
            "sell_bias_multiplier": 2.0,
            "min_trade_amount": 0.001,
            "max_trades_per_hour": 6,
            "price_check_interval": 60,
            "max_daily_loss": 10000.0,
            "max_daily_trades": 50,
            "emergency_stop_loss": 0.05,
        }

    @pytest.fixture
    def mock_model(self):
        """Mock PPO model for testing."""
        model = Mock()
        model.predict.return_value = ([1], None)  # Buy action
        model.observation_space.shape = [10]  # Mock observation space
        return model

    @pytest.fixture
    def mock_adapter(self):
        """Mock Coincheck adapter for testing."""
        adapter = Mock()
        adapter.get_current_price = AsyncMock(return_value=5000000.0)
        return adapter

    def test_initialization_with_mocks(self, mock_config: dict[str, Any], mock_model: Mock, mock_adapter: Mock) -> None:
        """Test LiveTrader initialization with mocked dependencies."""
        with patch("live_trade.PPO.load", return_value=mock_model), patch(
            "live_trade.CoincheckAdapter", return_value=mock_adapter
        ), patch("live_trade.create_production_auto_stop", return_value=Mock()):

            trader = LiveTrader(
                model_path="/fake/path/model.zip", config=mock_config, dry_run=True
            )

            assert trader.config == mock_config
            assert trader.dry_run is True
            assert trader.coincheck_adapter == mock_adapter

    def test_price_fetch_integration(self, mock_config: dict[str, Any], mock_adapter: Mock) -> None:
        """Test price fetching integration."""
        with patch("live_trade.CoincheckAdapter", return_value=mock_adapter):

            # Test successful price fetch
            trader = LiveTrader(
                model_path="/fake/path/model.zip", config=mock_config, dry_run=True
            )

            price = trader._get_current_price()
            assert price == 5000000.0

            # Test price validation
            assert 1000000 <= price <= 50000000

    def test_price_validation_edge_cases(self, mock_config: dict[str, Any]) -> None:
        """Test price validation with edge cases."""
        with patch("live_trade.CoincheckAdapter") as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter_class.return_value = mock_adapter

            trader = LiveTrader(
                model_path="/fake/path/model.zip", config=mock_config, dry_run=True
            )

            # Test invalid price (too low)
            mock_adapter.get_current_price = AsyncMock(return_value=500000.0)
            with pytest.raises(SystemExit):
                trader._get_current_price()

            # Test invalid price (too high)
            mock_adapter.get_current_price = AsyncMock(return_value=60000000.0)
            with pytest.raises(SystemExit):
                trader._get_current_price()

    def test_feature_computation_integration(self, mock_config: dict[str, Any]) -> None:
        """Test feature computation integration."""
        with patch(
            "live_trade.compute_features_batch",
            return_value=pd.DataFrame({"rsi": [55.0]}),
        ), patch("live_trade.CoincheckAdapter"):

            trader = LiveTrader(
                model_path="/fake/path/model.zip", config=mock_config, dry_run=True
            )

            # Initialize price history
            trader.price_history = [5000000.0] * 50

            features = trader._get_market_features()
            assert isinstance(features, dict)
            assert len(features) > 0

    def test_rsi_calculation_integration(self, mock_config: dict[str, Any]) -> None:
        """Test RSI calculation integration."""
        with patch("live_trade.compute_rsi", return_value=pd.Series([65.0])), patch(
            "live_trade.CoincheckAdapter"
        ):

            trader = LiveTrader(
                model_path="/fake/path/model.zip", config=mock_config, dry_run=True
            )

            prices = [5000000.0] * 20
            rsi = trader._calculate_rsi(prices, period=14)
            assert rsi == 65.0

    def test_health_status_integration(self, mock_config: dict[str, Any], mock_adapter: Mock) -> None:
        """Test health status reporting integration."""
        with patch("live_trade.CoincheckAdapter", return_value=mock_adapter):

            trader = LiveTrader(
                model_path="/fake/path/model.zip", config=mock_config, dry_run=True
            )

            health = trader.get_health_status()

            assert health["status"] in ["healthy", "degraded", "error"]
            assert "timestamp" in health
            assert "price_feed" in health
            assert "model_loaded" in health
            assert health["dry_run"] is True

    def test_risk_limits_integration(self, mock_config: dict[str, Any]) -> None:
        """Test risk limits integration."""
        with patch("live_trade.CoincheckAdapter"):

            # Test with risk limits enabled
            trader = LiveTrader(
                model_path="/fake/path/model.zip",
                config=mock_config,
                dry_run=True,
                disable_risk_limits=False,
            )

            assert trader.disable_risk_limits is False
            assert trader.config["max_daily_loss"] == 10000.0

            # Test with risk limits disabled
            trader_disabled = LiveTrader(
                model_path="/fake/path/model.zip",
                config=mock_config.copy(),
                dry_run=True,
                disable_risk_limits=True,
            )

            assert trader_disabled.disable_risk_limits is True
            assert trader_disabled.config["max_daily_loss"] == float("inf")

    @pytest.mark.asyncio
    async def test_concurrent_price_fetches(self, mock_config: dict[str, Any]) -> None:
        """Test concurrent price fetching doesn't cause issues."""
        with patch("live_trade.CoincheckAdapter") as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.get_current_price = AsyncMock(return_value=5000000.0)
            mock_adapter_class.return_value = mock_adapter

            trader = LiveTrader(
                model_path="/fake/path/model.zip", config=mock_config, dry_run=True
            )

            # Simulate concurrent price fetches
            tasks = [asyncio.create_task(trader._get_current_price()) for _ in range(5)]
            results = await asyncio.gather(*tasks)

            assert all(price == 5000000.0 for price in results)

    def test_configuration_override(self) -> None:
        """Test configuration override functionality."""
        custom_config = {
            "price_history_length": 100,
            "fallback_price": 6000000.0,
            "max_daily_trades": 100,
        }

        with patch("live_trade.CoincheckAdapter"):

            trader = LiveTrader(
                model_path="/fake/path/model.zip", config=custom_config, dry_run=True
            )

            assert trader.config["price_history_length"] == 100
            assert trader.config["fallback_price"] == 6000000.0
            assert trader.config["max_daily_trades"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
