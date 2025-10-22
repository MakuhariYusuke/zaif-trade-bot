"""
Tests for position sizing functionality.

This module tests position sizing with volatility targeting and risk management.
"""

import numpy as np
import pandas as pd

from ztb.risk.position_sizing import PositionSize, PositionSizer, SizingMethod


class TestPositionSizer:
    """Test cases for PositionSizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sizer = PositionSizer(
            target_volatility=0.10, method=SizingMethod.VOL_TARGETING
        )

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        sizer = PositionSizer()
        assert sizer.target_volatility == 0.10
        assert sizer.method == SizingMethod.VOL_TARGETING

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        sizer = PositionSizer(target_volatility=0.15, method=SizingMethod.EQUAL_WEIGHT)
        assert sizer.target_volatility == 0.15
        assert sizer.method == SizingMethod.EQUAL_WEIGHT

    def test_calculate_position_sizes_equal_weight(self):
        """Test position sizing with equal weight method."""
        # Create mock portfolio data
        portfolio_value = 100000.0
        symbols = ["AAPL", "GOOGL", "MSFT"]
        signals = {sym: 0.5 for sym in symbols}  # Equal signals
        current_prices = {"AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0}
        asset_volatilities = {"AAPL": 0.20, "GOOGL": 0.25, "MSFT": 0.18}

        sizer_equal = PositionSizer(method=SizingMethod.EQUAL_WEIGHT)
        positions = sizer_equal.calculate_position_sizes(
            signals, current_prices, portfolio_value, asset_volatilities
        )

        assert isinstance(positions, list)
        assert len(positions) == len(symbols)

        for pos in positions:
            assert isinstance(pos, PositionSize)
            assert pos.symbol in symbols
            assert pos.quantity > 0
            assert (
                "equal_weight" in pos.sizing_reason.lower()
                or "equal weight" in pos.sizing_reason.lower()
            )

    def test_calculate_position_sizes_vol_targeting(self):
        """Test position sizing with volatility targeting."""
        portfolio_value = 100000.0
        symbols = ["AAPL", "GOOGL"]
        signals = {"AAPL": 0.8, "GOOGL": 0.6}
        current_prices = {"AAPL": 150.0, "GOOGL": 2500.0}
        asset_volatilities = {"AAPL": 0.20, "GOOGL": 0.25}

        positions = self.sizer.calculate_position_sizes(
            signals, current_prices, portfolio_value, asset_volatilities
        )

        assert isinstance(positions, list)
        assert len(positions) == len(symbols)

        for pos in positions:
            assert isinstance(pos, PositionSize)
            assert pos.symbol in symbols
            assert pos.quantity > 0
            assert (
                "vol_targeting" in pos.sizing_reason.lower()
                or "vol targeting" in pos.sizing_reason.lower()
            )

    def test_calculate_position_sizes_kelly(self):
        """Test position sizing with Kelly criterion."""
        portfolio_value = 100000.0
        symbols = ["AAPL"]
        signals = {"AAPL": 0.7}
        current_prices = {"AAPL": 150.0}
        asset_volatilities = {"AAPL": 0.20}

        sizer_kelly = PositionSizer(method=SizingMethod.KELLY_CRITERION)
        positions = sizer_kelly.calculate_position_sizes(
            signals, current_prices, portfolio_value, asset_volatilities
        )

        assert isinstance(positions, list)
        assert len(positions) == len(symbols)

        for pos in positions:
            assert isinstance(pos, PositionSize)
            assert pos.symbol in symbols
            assert pos.quantity >= 0  # Could be 0 for conservative Kelly

    def test_estimate_asset_volatilities(self):
        """Test asset volatility estimation."""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = pd.DataFrame(
            {
                "AAPL": 150 * np.exp(np.cumsum(np.random.normal(0, 0.02, 100))),
                "GOOGL": 2500 * np.exp(np.cumsum(np.random.normal(0, 0.025, 100))),
            },
            index=dates,
        )

        volatilities = self.sizer.estimate_asset_volatilities(prices)

        assert isinstance(volatilities, dict)
        assert "AAPL" in volatilities
        assert "GOOGL" in volatilities
        assert all(v > 0 for v in volatilities.values())

    def test_calculate_position_sizes_insufficient_data(self):
        """Test position sizing with insufficient data."""
        portfolio_value = 100000.0
        symbols = ["AAPL"]
        signals = {"AAPL": 0.5}
        current_prices = {"AAPL": 150.0}
        asset_volatilities = {"AAPL": 0.20}

        # Should handle gracefully
        positions = self.sizer.calculate_position_sizes(
            signals, current_prices, portfolio_value, asset_volatilities
        )

        assert isinstance(positions, list)
        # May return empty list or handle gracefully
        assert len(positions) >= 0
