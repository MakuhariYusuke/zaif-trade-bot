"""Tests for LiveTrader validation methods."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ztb.trading.live_trader.live_trader import LiveTrader
from ztb.utils.errors import ValidationError


@pytest.fixture
def mock_config() -> dict:
    """Mock configuration for LiveTrader."""
    return {
        "exchange": "coincheck",
        "api_key": "test_key",
        "api_secret": "test_secret",
        "demo_mode": True,
        "max_position_size": 0.001,
        "dry_run": True,
    }


@pytest.fixture
def live_trader(mock_config: dict) -> LiveTrader:
    """LiveTrader instance for testing."""
    # Mock the required dependencies
    trader = LiveTrader.__new__(LiveTrader)
    trader.demo_mode = mock_config["demo_mode"]
    trader.dry_run = mock_config["dry_run"]
    trader._last_valid_price = 5000000.0

    # Mock exchange adapter
    mock_adapter = AsyncMock()
    trader.exchange_adapter = mock_adapter

    # Mock notifier
    trader.notifier = MagicMock()

    return trader


class TestLiveTraderValidation:
    """Test LiveTrader input validation."""

    def test_get_current_price_valid_price(self, live_trader: LiveTrader) -> None:
        """Test _get_current_price with valid price."""
        # Mock the async function to return a valid price

        async def mock_async_get_price():
            return 6000000.0

        # Mock asyncio.run
        with patch("asyncio.run", return_value=6000000.0):
            price = live_trader._get_current_price()
            assert price == 6000000.0

    def test_get_current_price_invalid_price_zero(
        self, live_trader: LiveTrader
    ) -> None:
        """Test _get_current_price with zero price does not raise exception but logs error."""
        # Mock the async function to return zero
        with patch("asyncio.run", return_value=0.0):
            # Should not raise ValidationError, but should fallback to last valid price
            price = live_trader._get_current_price()
            assert price == 5000000.0  # Should fallback to _last_valid_price

    def test_get_current_price_invalid_price_negative(
        self, live_trader: LiveTrader
    ) -> None:
        """Test _get_current_price with negative price does not raise exception but logs error."""
        # Mock the async function to return negative price
        with patch("asyncio.run", return_value=-1000.0):
            # Should not raise ValidationError, but should fallback to last valid price
            price = live_trader._get_current_price()
            assert price == 5000000.0  # Should fallback to _last_valid_price

    def test_get_current_price_none_fallback(self, live_trader: LiveTrader) -> None:
        """Test _get_current_price fallback when price is None."""
        # Mock the async function to return None
        with patch("asyncio.run", return_value=None):
            price = live_trader._get_current_price()
            assert price == 5000000.0  # Should fallback to _last_valid_price

    def test_execute_trade_valid_amount(self, live_trader: LiveTrader) -> None:
        """Test _execute_trade with valid amount."""
        # Should not raise an exception
        result = live_trader._execute_trade("buy", 0.001)
        assert result is True  # Demo mode returns True

    def test_execute_trade_zero_amount(self, live_trader: LiveTrader) -> None:
        """Test _execute_trade with zero amount raises ValidationError."""
        with pytest.raises(ValidationError, match="amount must be positive"):
            live_trader._execute_trade("buy", 0.0)

    def test_execute_trade_negative_amount(self, live_trader: LiveTrader) -> None:
        """Test _execute_trade with negative amount raises ValidationError."""
        with pytest.raises(ValidationError, match="amount must be positive"):
            live_trader._execute_trade("buy", -0.001)

    def test_execute_trade_very_small_amount(self, live_trader: LiveTrader) -> None:
        """Test _execute_trade with very small positive amount."""
        result = live_trader._execute_trade("buy", 0.000001)
        assert result is True  # Demo mode returns True
