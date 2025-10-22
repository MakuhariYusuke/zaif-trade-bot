"""Tests for PositionManager V433 component."""

import pytest

from ztb.trading.position_manager import (
    PortfolioState,
    PositionManagementConfig,
    PositionManager,
    PositionSignal,
)
from ztb.utils.errors import ValidationError


@pytest.fixture
def sample_config() -> PositionManagementConfig:
    """Sample position management config for testing."""
    return PositionManagementConfig()


@pytest.fixture
def position_manager(sample_config: PositionManagementConfig) -> PositionManager:
    """PositionManager instance for testing."""
    # Mock execution engine
    from unittest.mock import MagicMock

    mock_execution_engine = MagicMock()
    return PositionManager(mock_execution_engine, config=sample_config)


@pytest.fixture
def sample_position() -> PositionSignal:
    """Sample position signal for testing."""
    return PositionSignal(
        symbol="btc_jpy",
        action="open_long",
        strength=0.8,
        target_quantity=0.001,
        confidence=0.9,
        reason="Test signal",
    )


class TestPositionManagerInitialization:
    """Test PositionManager initialization."""

    def test_initialization(
        self, position_manager: PositionManager, sample_config: PositionManagementConfig
    ) -> None:
        """Test proper initialization."""
        assert position_manager.config_obj == sample_config
        assert isinstance(position_manager.portfolio_state, PortfolioState)
        assert position_manager.portfolio_state.total_capital == 100000.0


class TestPositionManagerPriceCalculations:
    """Test price calculation methods."""

    def test_calculate_stop_loss_price_long_position(
        self, position_manager: PositionManager
    ) -> None:
        """Test stop loss price calculation for long position."""
        from ztb.trading.position_manager import Position

        # Create a mock position
        position = Position(
            symbol="btc_jpy",
            quantity=0.001,
            average_price=100000.0,
            current_price=100000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=0,
        )

        current_price = 105000.0
        stop_price = position_manager.risk_manager.calculate_stop_loss_price(
            position, current_price
        )

        # For long position: stop_price = current_price * (1 - stop_loss_pct)
        expected_stop = current_price * (1 - position_manager.config.stop_loss_pct)
        assert stop_price == expected_stop

    def test_calculate_stop_loss_price_short_position(
        self, position_manager: PositionManager
    ) -> None:
        """Test stop loss price calculation for short position."""
        from ztb.trading.position_manager import Position

        # Create a mock position
        position = Position(
            symbol="btc_jpy",
            quantity=-0.001,  # Short position
            average_price=100000.0,
            current_price=100000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=0,
        )

        current_price = 95000.0
        stop_price = position_manager.risk_manager.calculate_stop_loss_price(
            position, current_price
        )

        # For short position: stop_price = current_price * (1 + stop_loss_pct)
        expected_stop = current_price * (1 + position_manager.config.stop_loss_pct)
        assert stop_price == expected_stop

    def test_calculate_take_profit_price_long_position(
        self, position_manager: PositionManager
    ) -> None:
        """Test take profit price calculation for long position."""
        from ztb.trading.position_manager import Position

        # Create a mock position
        position = Position(
            symbol="btc_jpy",
            quantity=0.001,
            average_price=100000.0,
            current_price=100000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=0,
        )

        current_price = 105000.0
        take_price = position_manager.risk_manager.calculate_take_profit_price(
            position, current_price
        )

        # For long position: take_price = current_price * (1 + take_profit_pct)
        expected_take = current_price * (1 + position_manager.config.take_profit_pct)
        assert take_price == expected_take

    def test_calculate_take_profit_price_short_position(
        self, position_manager: PositionManager
    ) -> None:
        """Test take profit price calculation for short position."""
        from ztb.trading.position_manager import Position

        # Create a mock position
        position = Position(
            symbol="btc_jpy",
            quantity=-0.001,  # Short position
            average_price=100000.0,
            current_price=100000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=0,
        )

        current_price = 95000.0
        take_price = position_manager.risk_manager.calculate_take_profit_price(
            position, current_price
        )

        # For short position: take_price = current_price * (1 - take_profit_pct)
        expected_take = current_price * (1 - position_manager.config.take_profit_pct)
        assert take_price == expected_take

    def test_calculate_stop_loss_price_invalid_price(
        self, position_manager: PositionManager
    ) -> None:
        """Test stop loss price calculation with invalid price."""
        from ztb.trading.position_manager import Position

        position = Position(
            symbol="btc_jpy",
            quantity=0.001,
            average_price=100000.0,
            current_price=100000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=0,
        )

        with pytest.raises(ValidationError, match="current_price must be positive"):
            position_manager.risk_manager.calculate_stop_loss_price(position, -1000.0)

    def test_calculate_take_profit_price_invalid_price(
        self, position_manager: PositionManager
    ) -> None:
        """Test take profit price calculation with invalid price."""
        from ztb.trading.position_manager import Position

        position = Position(
            symbol="btc_jpy",
            quantity=0.001,
            average_price=100000.0,
            current_price=100000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=0,
        )

        with pytest.raises(ValidationError, match="current_price must be positive"):
            position_manager.risk_manager.calculate_take_profit_price(position, 0.0)


class TestPositionManagerStopLossTakeProfitChecks:
    """Test stop loss and take profit checking methods."""

    def test_should_stop_loss_long_position_hit(
        self, position_manager: PositionManager
    ) -> None:
        """Test stop loss check for long position when hit."""
        from ztb.trading.position_manager import Position

        position = Position(
            symbol="btc_jpy",
            quantity=0.001,
            average_price=100000.0,
            current_price=95000.0,  # Below stop loss level
            unrealized_pnl=-5000.0,
            realized_pnl=0.0,
            timestamp=0,
        )

        # Stop loss should be triggered when current price <= stop price
        stop_price = position_manager.risk_manager.calculate_stop_loss_price(
            position, position.average_price
        )
        assert position.current_price <= stop_price

    def test_should_take_profit_long_position_hit(
        self, position_manager: PositionManager
    ) -> None:
        """Test take profit check for long position when hit."""
        from ztb.trading.position_manager import Position

        position = Position(
            symbol="btc_jpy",
            quantity=0.001,
            average_price=100000.0,
            current_price=108000.0,  # Above take profit level
            unrealized_pnl=8000.0,
            realized_pnl=0.0,
            timestamp=0,
        )

        # Take profit should be triggered when current price >= take price
        take_price = position_manager.risk_manager.calculate_take_profit_price(
            position, position.average_price
        )
        assert position.current_price >= take_price


class TestPositionManagerValidation:
    """Test input validation."""

    def test_calculate_stop_loss_price_zero_price(
        self, position_manager: PositionManager
    ) -> None:
        """Test stop loss calculation with zero price raises error."""
        from ztb.trading.position_manager import Position

        position = Position(
            symbol="btc_jpy",
            quantity=0.001,
            average_price=100000.0,
            current_price=100000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=0,
        )

        with pytest.raises(ValidationError, match="current_price must be positive"):
            position_manager.risk_manager.calculate_stop_loss_price(position, 0.0)

    def test_calculate_take_profit_price_negative_price(
        self, position_manager: PositionManager
    ) -> None:
        """Test take profit calculation with negative price raises error."""
        from ztb.trading.position_manager import Position

        position = Position(
            symbol="btc_jpy",
            quantity=0.001,
            average_price=100000.0,
            current_price=100000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=0,
        )

        with pytest.raises(ValidationError, match="current_price must be positive"):
            position_manager.risk_manager.calculate_take_profit_price(
                position, -50000.0
            )
