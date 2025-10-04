"""
Simulated broker for paper trading.

Provides realistic trading simulation without real exchange connectivity.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .broker_interfaces import Balance, IBroker, Order, Position
from .order_state import OrderStateMachine


class SimBroker(IBroker):
    """Simulated broker for paper trading."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        slippage_bps: float = 5.0,
        commission_bps: float = 0.0,
        price_feed: Optional[pd.DataFrame] = None,
        venue_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize simulated broker."""
        self.initial_balance = initial_balance
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.venue_config = venue_config
        self.symbols = venue_config.get("symbols", []) if venue_config else []

        # Account state
        self.balance = {"JPY": initial_balance, "BTC": 0.0}
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0

        # Price feed for simulation
        self.price_feed = price_feed
        self.current_time_index = 0

        # Trading history
        self.trade_log: List[Dict[str, Any]] = []

        # Order state management
        self.order_state_machine = OrderStateMachine()

    def _is_symbol_configured(self, symbol: str) -> bool:
        """Check if symbol is configured in venue."""
        return any(s.get("symbol") == symbol for s in self.symbols)

    def _generate_idempotency_key(
        self, symbol: str, side: str, quantity: float, price: Optional[float]
    ) -> str:
        """Generate idempotency key for order deduplication."""
        import hashlib

        key_data = f"{symbol}:{side}:{quantity}:{price or 0}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price."""
        slippage_factor = self.slippage_bps / 10000  # Convert bps to decimal
        if side == "buy":
            return price * (1 + slippage_factor)
        else:  # sell
            return price * (1 - slippage_factor)

    def _apply_commission(self, notional: float) -> float:
        """Calculate commission cost."""
        return notional * (self.commission_bps / 10000)

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from feed or generate synthetic."""
        if self.price_feed is not None and self.current_time_index < len(
            self.price_feed
        ):
            row = self.price_feed.iloc[self.current_time_index]
            if "close" in row:
                return float(row["close"])

        # Fallback: generate synthetic price
        base_price = 30000
        # Simple random walk
        np.random.seed(42 + self.current_time_index)
        return base_price * (1 + np.random.normal(0, 0.01))

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
        client_order_id: Optional[str] = None,
        sizing_reason: Optional[str] = None,
        target_vol: Optional[float] = None,
    ) -> Order:
        """Place a simulated order."""
        await asyncio.sleep(0.01)  # Simulate network latency

        if not self._is_symbol_configured(symbol):
            raise ValueError(f"Symbol {symbol} not configured in venue")

        # Generate or use client_order_id
        if client_order_id is None:
            import time
            import uuid

            client_order_id = (
                f"{symbol}_{side}_{quantity}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            )

        # Check for duplicate orders using state machine
        existing_order = self.order_state_machine.get_order_by_idempotency_key(
            self._generate_idempotency_key(symbol, side, quantity, price)
        )
        if existing_order and not existing_order.is_terminal_state():
            raise ValueError(f"Duplicate order detected: {client_order_id}")

        order_id = f"sim_{self.order_counter}"
        self.order_counter += 1

        # For market orders, use current price
        if order_type == "market" or price is None:
            current_price = self._get_current_price(symbol)
            if current_price is None:
                raise ValueError(f"No price available for {symbol}")

            execution_price = self._apply_slippage(current_price, side)
        else:
            execution_price = price

        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
            order_type=order_type,
            status="filled",  # Immediate fill for simulation
            client_order_id=client_order_id,
            sizing_reason=sizing_reason,
            target_vol=target_vol,
        )

        self.orders[order_id] = order

        # Register with state machine
        from .order_state import OrderData

        order_data = OrderData(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
            order_type=order_type,
        )
        self.order_state_machine.create_order(order_data)

        # Execute the trade
        await self._execute_trade(order)

        return order

    async def _execute_trade(self, order: Order) -> None:
        """Execute the trade and update account state."""
        symbol = order.symbol
        side = order.side
        quantity = order.quantity
        price = order.price

        if price is None:
            return

        notional = quantity * price
        commission = self._apply_commission(notional)

        if side == "buy":
            # Check sufficient balance
            required_jpy = notional + commission
            if self.balance.get("JPY", 0) < required_jpy:
                order.status = "rejected"
                return

            # Update balance and position
            self.balance["JPY"] -= required_jpy
            self.balance["BTC"] = self.balance.get("BTC", 0) + quantity

            # Update or create position
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_quantity = pos.quantity + quantity
                total_cost = (pos.quantity * pos.avg_price) + notional
                new_avg_price = total_cost / total_quantity
                pos.quantity = total_quantity
                pos.avg_price = new_avg_price
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                    pnl=0.0,
                )

        else:  # sell
            # Check sufficient position
            current_btc = self.balance.get("BTC", 0)
            if current_btc < quantity:
                order.status = "rejected"
                return

            # Update balance and position
            self.balance["BTC"] -= quantity
            self.balance["JPY"] += notional - commission

            # Update position
            if symbol in self.positions:
                pos = self.positions[symbol]
                if pos.quantity <= quantity:
                    # Close position
                    pnl = (price - pos.avg_price) * pos.quantity
                    self.trade_log.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "symbol": symbol,
                            "side": side,
                            "quantity": pos.quantity,
                            "price": price,
                            "pnl": pnl,
                            "commission": commission,
                        }
                    )
                    del self.positions[symbol]
                else:
                    # Partial close
                    pnl = (price - pos.avg_price) * quantity
                    pos.quantity -= quantity
                    self.trade_log.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "symbol": symbol,
                            "side": side,
                            "quantity": quantity,
                            "price": price,
                            "pnl": pnl,
                            "commission": commission,
                        }
                    )

        # Update position P&L
        current_price = self._get_current_price(symbol) or price
        for pos in self.positions.values():
            pos.current_price = current_price
            pos.pnl = (current_price - pos.avg_price) * pos.quantity

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a simulated order."""
        await asyncio.sleep(0.01)

        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == "pending":
                order.status = "cancelled"
                return True

        return False

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        return self.orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        orders = [o for o in self.orders.values() if o.status == "pending"]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        return list(self.positions.values())

    async def get_balance(self, currency: Optional[str] = None) -> List[Balance]:
        """Get account balance."""
        balances = []
        for curr, amount in self.balance.items():
            if currency is None or curr == currency:
                balances.append(
                    Balance(
                        currency=curr,
                        free=amount,
                        locked=0.0,  # Simplified
                        total=amount,
                    )
                )
        return balances

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price."""
        return self._get_current_price(symbol)

    def get_trade_log(self) -> List[Dict[str, Any]]:
        """Get trading history."""
        return self.trade_log.copy()

    def get_pnl_series(self) -> List[Dict[str, Any]]:
        """Get P&L time series."""
        pnl_data = []
        cumulative_pnl = 0.0

        for trade in self.trade_log:
            cumulative_pnl += trade["pnl"] - trade.get("commission", 0)
            pnl_data.append(
                {
                    "timestamp": trade["timestamp"],
                    "pnl": trade["pnl"],
                    "cumulative_pnl": cumulative_pnl,
                    "balance_jpy": self.balance.get("JPY", 0),
                    "balance_btc": self.balance.get("BTC", 0),
                }
            )

        return pnl_data
