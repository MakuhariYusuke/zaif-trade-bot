"""
Unit tests for trading core interfaces
取引コアインターフェースの単体テスト
"""

import pytest
import pandas as pd
from unittest.mock import AsyncMock
from typing import Dict, Any
from ztb.trading.core.interfaces import TradingStrategy, DataProvider, OrderManager


class TestTradingStrategy:
    def test_trading_strategy_is_abstract(self):
        """Test that TradingStrategy cannot be instantiated directly"""
        with pytest.raises(TypeError):
            TradingStrategy()

    def test_trading_strategy_abstract_methods(self):
        """Test that TradingStrategy has required abstract methods"""
        assert hasattr(TradingStrategy, 'generate_signal')
        assert hasattr(TradingStrategy, 'get_required_columns')


class TestDataProvider:
    def test_data_provider_is_abstract(self):
        """Test that DataProvider cannot be instantiated directly"""
        with pytest.raises(TypeError):
            DataProvider()

    def test_data_provider_abstract_methods(self):
        """Test that DataProvider has required abstract methods"""
        assert hasattr(DataProvider, 'get_historical_data')
        assert hasattr(DataProvider, 'get_current_price')


class TestOrderManager:
    def test_order_manager_is_abstract(self):
        """Test that OrderManager cannot be instantiated directly"""
        with pytest.raises(TypeError):
            OrderManager()

    def test_order_manager_abstract_methods(self):
        """Test that OrderManager has required abstract method"""
        assert hasattr(OrderManager, 'place_order')


class ConcreteTradingStrategy(TradingStrategy):
    """Concrete implementation for testing"""
    def generate_signal(self, data: pd.DataFrame, current_position: int) -> Dict[str, Any]:
        return {"action": "buy", "quantity": 1}

    def get_required_columns(self) -> list[str]:
        return ["close", "volume"]


class ConcreteDataProvider(DataProvider):
    """Concrete implementation for testing"""
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame({"close": [100, 101], "volume": [1000, 1100]})

    async def get_current_price(self, symbol: str) -> float:
        return 100.5


class ConcreteOrderManager(OrderManager):
    """Concrete implementation for testing"""
    async def place_order(self, order: Dict[str, Any]) -> str:
        return "order_123"

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        return {"status": "filled", "order_id": order_id}


class TestConcreteImplementations:
    def test_concrete_trading_strategy(self):
        """Test concrete trading strategy implementation"""
        strategy = ConcreteTradingStrategy()

        # Test generate_signal
        data = pd.DataFrame({"close": [100, 101], "volume": [1000, 1100]})
        signal = strategy.generate_signal(data, 0)
        assert signal["action"] == "buy"
        assert signal["quantity"] == 1

        # Test get_required_columns
        columns = strategy.get_required_columns()
        assert columns == ["close", "volume"]

    @pytest.mark.asyncio
    async def test_concrete_data_provider(self):
        """Test concrete data provider implementation"""
        provider = ConcreteDataProvider()

        # Test get_historical_data
        data = await provider.get_historical_data("BTC_JPY", "2023-01-01", "2023-01-02")
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert "close" in data.columns
        assert "volume" in data.columns

        # Test get_current_price
        price = await provider.get_current_price("BTC_JPY")
        assert price == 100.5

    @pytest.mark.asyncio
    async def test_concrete_order_manager(self):
        """Test concrete order manager implementation"""
        manager = ConcreteOrderManager()

        # Test place_order
        order = {"symbol": "BTC_JPY", "side": "buy", "quantity": 1}
        order_id = await manager.place_order(order)
        assert order_id == "order_123"