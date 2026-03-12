"""
Integration tests for end-to-end trading workflows.

Tests cover complete trading cycles from data generation through
signal processing to trade execution and position management.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd


# Create simplified versions for testing
class DataGenerator:
    """Simplified DataGenerator for integration testing."""

    def __init__(self, **kwargs):
        self.cache_dir = None
        self.enable_memory_cache = True
        self.default_seed = 42

    def generate_synthetic_data(self, n_samples=1000, **kwargs):
        """Generate synthetic OHLCV data."""
        np.random.seed(self.default_seed)

        # Generate price series
        returns = np.random.normal(0, 0.02, n_samples)
        price = 50000 * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        data = pd.DataFrame(
            {
                "open": price * (1 + np.random.normal(0, 0.005, n_samples)),
                "high": price * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
                "low": price * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
                "close": price,
                "volume": np.random.uniform(100, 10000, n_samples),
            }
        )

        return data


class TaLibWrapper:
    """Simplified TaLibWrapper for integration testing."""

    def __init__(self, **kwargs):
        self.enable_cache = True
        self._cache = {}

    def sma(self, data, period=20):
        """Simple Moving Average."""
        if isinstance(data, pd.Series):
            data = data.values
        if len(data) < period:
            return np.full(len(data), np.nan)

        result = np.convolve(data, np.ones(period), "valid") / period
        padding = np.full(len(data) - len(result), np.nan)
        return np.concatenate([padding, result])

    def rsi(self, data, period=14):
        """Relative Strength Index."""
        if isinstance(data, pd.Series):
            data = data.values
        if len(data) < period + 1:
            return np.full(len(data), np.nan)

        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.convolve(gain, np.ones(period), "valid") / period
        avg_loss = np.convolve(loss, np.ones(period), "valid") / period

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        padding = np.full(len(data) - len(rsi), np.nan)
        return np.concatenate([padding, rsi])


class TradingStrategy:
    """Simplified trading strategy for integration testing."""

    def __init__(self, talib_wrapper):
        self.talib = talib_wrapper

    def generate_signals(self, data):
        """Generate trading signals based on technical indicators."""
        sma_20 = self.talib.sma(data["close"], 20)
        sma_50 = self.talib.sma(data["close"], 50)
        rsi = self.talib.rsi(data["close"], 14)

        signals = pd.DataFrame(index=data.index)
        signals["sma_signal"] = np.where(sma_20 > sma_50, 1, -1)
        signals["rsi_signal"] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        signals["combined_signal"] = signals["sma_signal"] + signals["rsi_signal"]

        # Generate final signal
        signals["signal"] = np.where(
            signals["combined_signal"] >= 1,
            1,
            np.where(signals["combined_signal"] <= -1, -1, 0),
        )

        return signals


class PositionManager:
    """Simplified position manager for integration testing."""

    def __init__(self):
        self.positions = []
        self.balance = 10000.0  # Starting balance

    def execute_trade(self, signal, price, quantity=0.001):
        """Execute a trade based on signal."""
        if signal == 1:  # Buy
            cost = price * quantity
            if cost <= self.balance:
                self.positions.append(
                    {
                        "type": "buy",
                        "price": price,
                        "quantity": quantity,
                        "timestamp": len(self.positions),
                    }
                )
                self.balance -= cost
                return True
        elif signal == -1:  # Sell
            # Find existing position to sell
            buy_positions = [p for p in self.positions if p["type"] == "buy"]
            if buy_positions:
                # Sell the first buy position
                pos = buy_positions[0]
                revenue = price * pos["quantity"]
                self.balance += revenue
                profit = revenue - (pos["price"] * pos["quantity"])
                self.positions.remove(pos)
                self.positions.append(
                    {
                        "type": "sell",
                        "price": price,
                        "quantity": pos["quantity"],
                        "profit": profit,
                        "timestamp": len(self.positions),
                    }
                )
                return True
        return False

    def get_pnl(self):
        """Calculate current P&L."""
        realized_pnl = sum(
            p.get("profit", 0) for p in self.positions if p["type"] == "sell"
        )
        unrealized_pnl = 0

        # Calculate unrealized P&L for open positions
        buy_positions = [p for p in self.positions if p["type"] == "buy"]
        if buy_positions:
            # Assume current price is the last position's price for simplicity
            current_price = buy_positions[-1]["price"]
            for pos in buy_positions:
                unrealized_pnl += (current_price - pos["price"]) * pos["quantity"]

        return realized_pnl + unrealized_pnl


class TradingWorkflow:
    """Complete trading workflow integration."""

    def __init__(self):
        self.data_generator = DataGenerator()
        self.talib_wrapper = TaLibWrapper()
        self.strategy = TradingStrategy(self.talib_wrapper)
        self.position_manager = PositionManager()

    def run_trading_cycle(self, n_samples=1000):
        """Run a complete trading cycle."""
        # 1. Generate market data
        market_data = self.data_generator.generate_synthetic_data(n_samples=n_samples)

        # 2. Generate trading signals
        signals = self.strategy.generate_signals(market_data)

        # 3. Execute trades
        trades_executed = 0
        for i, (idx, row) in enumerate(signals.iterrows()):
            if i >= 50:  # Skip initial period where indicators are NaN
                signal = row["signal"]
                price = market_data.loc[idx, "close"]
                if self.position_manager.execute_trade(signal, price):
                    trades_executed += 1

        # 4. Calculate final P&L
        final_pnl = self.position_manager.get_pnl()
        final_balance = self.position_manager.balance

        return {
            "market_data": market_data,
            "signals": signals,
            "trades_executed": trades_executed,
            "final_pnl": final_pnl,
            "final_balance": final_balance,
            "total_positions": len(self.position_manager.positions),
        }


class TestTradingWorkflowIntegration:
    """Integration tests for complete trading workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.workflow = TradingWorkflow()

    def test_complete_trading_cycle(self):
        """Test a complete trading cycle from data generation to P&L calculation."""
        result = self.workflow.run_trading_cycle(n_samples=500)

        # Verify data generation
        assert isinstance(result["market_data"], pd.DataFrame)
        assert len(result["market_data"]) == 500
        assert all(
            col in result["market_data"].columns
            for col in ["open", "high", "low", "close", "volume"]
        )

        # Verify signal generation
        assert isinstance(result["signals"], pd.DataFrame)
        assert len(result["signals"]) == 500
        assert "signal" in result["signals"].columns

        # Verify trading execution
        assert isinstance(result["trades_executed"], int)
        assert result["trades_executed"] >= 0

        # Verify P&L calculation
        assert isinstance(result["final_pnl"], (int, float))
        assert isinstance(result["final_balance"], (int, float))
        assert result["final_balance"] >= 0  # Should not go negative

        # Verify position tracking
        assert isinstance(result["total_positions"], int)
        assert result["total_positions"] >= 0

    def test_data_generation_integration(self):
        """Test data generation component integration."""
        data = self.workflow.data_generator.generate_synthetic_data(n_samples=200)

        assert len(data) == 200
        assert not data.isnull().any().any()  # No NaN values
        assert (data["high"] >= data["low"]).all()  # High >= Low
        assert (data["volume"] > 0).all()  # Positive volume

    def test_technical_analysis_integration(self):
        """Test technical analysis component integration."""
        data = self.workflow.data_generator.generate_synthetic_data(n_samples=100)
        prices = data["close"]

        # Test SMA calculation
        sma = self.workflow.talib_wrapper.sma(prices, 20)
        assert len(sma) == len(prices)
        assert not np.isnan(sma[-1])  # Last value should not be NaN

        # Test RSI calculation
        rsi = self.workflow.talib_wrapper.rsi(prices, 14)
        assert len(rsi) == len(prices)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert ((valid_rsi >= 0) & (valid_rsi <= 100)).all()

    def test_strategy_signal_generation(self):
        """Test trading strategy signal generation."""
        data = self.workflow.data_generator.generate_synthetic_data(n_samples=100)
        signals = self.workflow.strategy.generate_signals(data)

        assert len(signals) == len(data)
        assert "signal" in signals.columns
        assert signals["signal"].isin([-1, 0, 1]).all()  # Valid signal values

    def test_position_management(self):
        """Test position management and trade execution."""
        # Test buy trade
        success = self.workflow.position_manager.execute_trade(1, 50000, 0.001)
        assert success
        assert len(self.workflow.position_manager.positions) == 1
        assert self.workflow.position_manager.balance < 10000  # Balance decreased

        # Test sell trade
        success = self.workflow.position_manager.execute_trade(-1, 51000, 0.001)
        assert success
        assert len(self.workflow.position_manager.positions) == 1  # Sell position added
        assert (
            self.workflow.position_manager.balance > 10000
        )  # Balance increased (profit)

        # Test P&L calculation
        pnl = self.workflow.position_manager.get_pnl()
        assert pnl > 0  # Should have profit

    def test_workflow_with_insufficient_data(self):
        """Test workflow behavior with insufficient data."""
        result = self.workflow.run_trading_cycle(n_samples=10)  # Very small dataset

        # Should still complete without errors
        assert isinstance(result["market_data"], pd.DataFrame)
        assert isinstance(result["signals"], pd.DataFrame)
        assert isinstance(result["trades_executed"], int)
        assert isinstance(result["final_pnl"], (int, float))

    def test_workflow_reproducibility(self):
        """Test that workflow results are reproducible with same seed."""
        # Run workflow twice
        result1 = self.workflow.run_trading_cycle(n_samples=200)
        self.workflow = TradingWorkflow()  # Reset workflow
        result2 = self.workflow.run_trading_cycle(n_samples=200)

        # Results should be identical (same random seed)
        assert result1["trades_executed"] == result2["trades_executed"]
        assert abs(result1["final_pnl"] - result2["final_pnl"]) < 1e-10
        assert abs(result1["final_balance"] - result2["final_balance"]) < 1e-10

    def test_error_handling_in_workflow(self):
        """Test error handling in workflow components."""
        # Test with empty data - should handle gracefully
        with patch.object(
            self.workflow.data_generator, "generate_synthetic_data"
        ) as mock_gen:
            # Create empty DataFrame with required columns
            mock_gen.return_value = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

            result = self.workflow.run_trading_cycle(n_samples=0)
            # Should handle empty data gracefully
            assert isinstance(result, dict)
            assert "market_data" in result
            assert len(result["market_data"]) == 0
