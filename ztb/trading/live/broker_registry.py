"""
Broker Registry for Trading Operations

Provides a registry of available brokers and their implementations,
with contract testing support.
"""

import logging
from typing import Any, Dict, Optional, Protocol, Type

from .coincheck_adapter import CoincheckAdapter

logger = logging.getLogger(__name__)


class BrokerProtocol(Protocol):
    """Protocol defining broker interface."""

    def get_balance(self, currency: str) -> float:
        """Get balance for currency."""
        ...

    def place_order(
        self, symbol: str, side: str, quantity: float, price: Optional[float] = None
    ) -> str:
        """Place an order and return order ID."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        ...

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        ...

    def get_open_orders(self) -> list[Dict[str, Any]]:
        """Get all open orders."""
        ...


class SimBroker:
    """Simple simulation broker for testing."""

    def __init__(self) -> None:
        self.balances = {"JPY": 100000.0, "BTC": 0.0}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.order_counter = 0

    def get_balance(self, currency: str) -> float:
        return self.balances.get(currency, 0.0)

    def place_order(
        self, symbol: str, side: str, quantity: float, price: Optional[float] = None
    ) -> str:
        self.order_counter += 1
        order_id = f"sim_{self.order_counter}"
        self.orders[order_id] = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "filled",  # Sim broker fills immediately
        }
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id]["status"] = "cancelled"
            return True
        return False

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        return self.orders.get(order_id, {"status": "not_found"})

    def get_open_orders(self) -> list[Dict[str, Any]]:
        return [order for order in self.orders.values() if order["status"] == "open"]


class CoincheckSkeletonBroker:
    """Skeleton implementation for Coincheck broker (raises NotImplemented for network calls)."""

    def __init__(
        self, api_key: Optional[str] = None, api_secret: Optional[str] = None
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        if not self.api_key or not self.api_secret:
            logger.warning(
                "Coincheck broker initialized without credentials - will raise NotImplementedError"
            )

    def get_balance(self, currency: str) -> float:
        if not self.api_key:
            raise NotImplementedError("Coincheck API not configured")
        # TODO: Implement actual API call
        raise NotImplementedError("Coincheck balance API not implemented")

    def place_order(
        self, symbol: str, side: str, quantity: float, price: Optional[float] = None
    ) -> str:
        if not self.api_key:
            raise NotImplementedError("Coincheck API not configured")
        # TODO: Implement actual API call
        raise NotImplementedError("Coincheck place order API not implemented")

    def cancel_order(self, order_id: str) -> bool:
        if not self.api_key:
            raise NotImplementedError("Coincheck API not configured")
        # TODO: Implement actual API call
        raise NotImplementedError("Coincheck cancel order API not implemented")

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        if not self.api_key:
            raise NotImplementedError("Coincheck API not configured")
        # TODO: Implement actual API call
        raise NotImplementedError("Coincheck order status API not implemented")

    def get_open_orders(self) -> list[Dict[str, Any]]:
        if not self.api_key:
            raise NotImplementedError("Coincheck API not configured")
        # TODO: Implement actual API call
        raise NotImplementedError("Coincheck open orders API not implemented")


class BrokerRegistry:
    """Registry of available brokers."""

    def __init__(self) -> None:
        self._brokers: Dict[str, Type[Any]] = {}
        self._register_default_brokers()

    def _register_default_brokers(self) -> None:
        """Register default broker implementations."""
        self.register_broker("sim", SimBroker)
        self.register_broker("coincheck", CoincheckAdapter)
        self.register_broker("coincheck_skeleton", CoincheckSkeletonBroker)

    def register_broker(self, name: str, broker_class: Type[Any]) -> None:
        """Register a broker implementation."""
        self._brokers[name] = broker_class
        logger.info(f"Registered broker: {name}")

    def get_broker(self, name: str, **kwargs: Any) -> Any:
        """Get a broker instance."""
        if name not in self._brokers:
            raise ValueError(f"Unknown broker: {name}")
        return self._brokers[name](**kwargs)

    def list_brokers(self) -> list[str]:
        """List available brokers."""
        return list(self._brokers.keys())


# Global registry instance
_broker_registry: Optional[BrokerRegistry] = None


def get_broker_registry() -> BrokerRegistry:
    """Get global broker registry instance."""
    global _broker_registry
    if _broker_registry is None:
        _broker_registry = BrokerRegistry()
    return _broker_registry
