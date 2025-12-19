"""Minimal paper trading manager stub used by some integration tests.

Only implements small surface area: Order, OrderSide, OrderType, and a
PaperTradingManager that can accept orders and return simple confirmations.
"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Order:
    id: str
    side: OrderSide
    qty: float
    price: Optional[float] = None
    type: OrderType = OrderType.MARKET


class PaperTradingManager:
    def __init__(self):
        self._orders: List[Order] = []

    def place_order(self, order: Order) -> dict:
        self._orders.append(order)
        return {"status": "ok", "order_id": order.id}

    def get_orders(self) -> List[Order]:
        return list(self._orders)
"""Minimal paper trading manager stub used by some integration tests."""
from enum import Enum
from dataclasses import dataclass
from typing import Any




class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
class Order:
    id: int
    side: OrderSide
    type: OrderType
    amount: float
    price: float | None = None


class PaperTradingManager:
    def __init__(self, *args, **kwargs):
        self.orders = []

    def place_order(self, order: Order) -> Order:
        self.orders.append(order)
        return order

    def cancel_order(self, order_id: int) -> bool:
        for o in list(self.orders):
            if o.id == order_id:
                self.orders.remove(o)
                return True
        return False

__all__ = ["Order", "OrderSide", "OrderType", "PaperTradingManager"]
