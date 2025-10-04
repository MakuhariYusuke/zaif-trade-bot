"""
Trading bridge for paper trading to live trading transition.

Provides VirtualTradingBridge for safe paper trading simulation
and interfaces for live trading with Zaif API.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union, cast

import pandas as pd
import requests

from ztb.ops.monitoring.monitoring import get_exporter
from ztb.utils.errors import safe_operation

logger = logging.getLogger(__name__)


@dataclass
class SlippageAnalysis:
    """Analysis of slippage impact on trading performance"""

    total_orders: int = 0
    slippage_impact: float = 0.0  # Total slippage cost in base currency
    avg_slippage_percent: float = 0.0
    max_slippage_percent: float = 0.0
    slippage_events: List[Dict[str, Any]] = field(default_factory=list)

    def add_slippage_event(
        self,
        symbol: str,
        side: str,
        intended_price: float,
        executed_price: float,
        quantity: float,
    ) -> None:
        """Add a slippage event for analysis"""
        slippage_amount = executed_price - intended_price
        slippage_percent = (slippage_amount / intended_price) * 100

        event = {
            "symbol": symbol,
            "side": side,
            "intended_price": intended_price,
            "executed_price": executed_price,
            "slippage_amount": slippage_amount,
            "slippage_percent": slippage_percent,
            "quantity": quantity,
            "timestamp": datetime.now(),
        }

        self.slippage_events.append(event)
        self.total_orders += 1
        self.slippage_impact += abs(slippage_amount * quantity)
        self.avg_slippage_percent = (
            self.avg_slippage_percent * (self.total_orders - 1) + abs(slippage_percent)
        ) / self.total_orders
        self.max_slippage_percent = max(
            self.max_slippage_percent, abs(slippage_percent)
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get slippage analysis summary"""
        return {
            "total_orders": self.total_orders,
            "slippage_impact": self.slippage_impact,
            "avg_slippage_percent": self.avg_slippage_percent,
            "max_slippage_percent": self.max_slippage_percent,
            "slippage_events_count": len(self.slippage_events),
        }


@dataclass
class VirtualOrder:
    """Virtual order representation for paper trading"""

    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market' or 'limit'
    quantity: float
    price: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: str = "filled"  # 'pending', 'filled', 'cancelled'
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.filled_quantity == 0.0 and self.status == "filled":
            self.filled_quantity = self.quantity
        if self.filled_price is None and self.status == "filled":
            self.filled_price = self.price


class VirtualTradingBridge:
    """
    Virtual trading bridge for paper trading simulation.

    Simulates order execution with immediate fills and logging.
    Provides interface compatible with live trading for seamless transition.
    """

    def __init__(
        self, initial_balance: float = 10000.0, commission_rate: float = 0.001
    ) -> None:
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_rate = commission_rate
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.orders: List[VirtualOrder] = []
        self.order_counter = 0
        self.slippage_analysis = SlippageAnalysis()

    def get_market_price(self, symbol: str) -> float:
        """
        Get current market price for symbol.
        In virtual trading, this should be provided by the calling code.
        """
        # This should be overridden or provided by the trading environment
        raise NotImplementedError(
            "Market price must be provided by trading environment"
        )

    def round_quantity(self, quantity: float, symbol: str = "BTC/JPY") -> float:
        """Round quantity to Zaif minimum unit (0.0001 BTC)"""
        if "BTC" in symbol:
            return round(quantity, 4)
        return quantity

    def calculate_slippage(self, symbol: str, side: Union[Literal["buy"], Literal["sell"], str], quantity: float) -> float:
        """
        Calculate dynamic slippage based on board spread.
        Simplified implementation - in real trading, use actual order book.
        """
        return safe_operation(
            logger=logger,
            operation=lambda: self._calculate_slippage_impl(symbol, side, quantity),
            context="slippage_calculation",
            default_result=0.0,  # No slippage on error
        )

    def _calculate_slippage_impl(self, symbol: str, side: str, quantity: float) -> float:
        """Implementation of slippage calculation."""
        # Mock slippage: 0.1% for buy, -0.1% for sell
        base_slippage = 0.001
        if side == "sell":
            base_slippage = -base_slippage
        return base_slippage

    def place_market_order(
        self,
        symbol: str,
        side: Union[Literal["buy"], Literal["sell"], str],
        quantity: float,
        current_price: Optional[float] = None,
    ) -> VirtualOrder:
        """
        Place market order with immediate execution simulation.

        Args:
            symbol: Trading pair (e.g., 'BTC/JPY')
            side: 'buy' or 'sell'
            quantity: Order quantity
            current_price: Current market price (if not provided, uses get_market_price)

        Returns:
            VirtualOrder object
        """
        def execute_order() -> VirtualOrder:
            if current_price is None:
                current_price_inner = self.get_market_price(symbol)
            else:
                current_price_inner = current_price

            # Round quantity
            quantity_rounded = self.round_quantity(quantity, symbol)

            # Calculate slippage
            slippage = self.calculate_slippage(symbol, side, quantity_rounded)
            execution_price = current_price_inner * (1 + slippage)

            # Record slippage for analysis
            self.slippage_analysis.add_slippage_event(
                symbol=symbol,
                side=side,
                intended_price=current_price_inner,
                executed_price=execution_price,
                quantity=quantity_rounded,
            )

            # Calculate commission (always positive)
            commission = abs(quantity_rounded * execution_price * self.commission_rate)

            # Create order
            self.order_counter += 1
            order = VirtualOrder(
                order_id=f"virtual_{self.order_counter:06d}",
                symbol=symbol,
                side=side,
                order_type="market",
                quantity=quantity_rounded,
                price=current_price_inner,
                filled_price=execution_price,
                commission=commission,
            )

            # Update balance and positions
            if side == "buy":
                cost = quantity_rounded * execution_price + commission
                if self.balance >= cost:
                    self.balance -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + quantity_rounded
                else:
                    order.status = "cancelled"
                    logger.warning(
                        f"Insufficient balance for buy order: {cost} > {self.balance}"
                    )
            else:  # sell
                current_position = self.positions.get(symbol, 0)
                if current_position >= quantity_rounded:
                    proceeds = quantity_rounded * execution_price - commission
                    self.balance += proceeds
                    self.positions[symbol] = current_position - quantity_rounded
                else:
                    order.status = "cancelled"
                    logger.warning(
                        f"Insufficient position for sell order: {quantity_rounded} > {current_position}"
                    )

            self.orders.append(order)
            logger.info(
                f"Virtual {side} order executed: {quantity_rounded} {symbol} at {execution_price:.2f}"
            )
            return order

        return safe_operation(logger, execute_order, f"place_market_order({symbol}, {side}, {quantity})")

    def get_balance(self) -> float:
        """Get current balance"""
        return self.balance

    def get_position(self, symbol: str) -> float:
        """Get current position for symbol"""
        return self.positions.get(symbol, 0)

    def get_order_history(self) -> List[VirtualOrder]:
        """Get order history"""
        return self.orders.copy()

    def reset(self) -> None:
        """Reset bridge to initial state"""
        self.balance = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.order_counter = 0
        self.slippage_analysis = SlippageAnalysis()
        logger.info("Virtual trading bridge reset")

    def get_slippage_analysis(self) -> Dict[str, Any]:
        """Get slippage analysis summary"""
        return self.slippage_analysis.get_summary()

    def reset_analysis(self) -> None:
        """Reset slippage analysis"""
        self.slippage_analysis = SlippageAnalysis()


class LiveTradingBridge:
    """
    Live trading bridge for Zaif API.

    Provides actual order execution through Zaif API.
    Includes safety measures and risk management.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        risk_manager: Optional[Any] = None,
        discord_webhook_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.risk_manager = risk_manager
        self.discord_webhook_url = discord_webhook_url
        self.daily_loss_limit = 0.02  # 2%
        self.max_consecutive_losses = 5
        self.circuit_breaker_threshold = 0.20  # ±20% price change (softened)

        # Extended risk management (Step 8)
        self.max_drawdown_limit = 0.05  # 5% maximum drawdown
        self.max_position_size = 1.0  # 100% of portfolio per position (no limit)
        self.max_open_positions = 3  # Maximum 3 open positions
        self.min_order_size = 0.001  # Minimum order size (BTC)
        self.max_order_size = 1.0  # Maximum order size (BTC)

        # MDD tracking
        self.peak_balance = 0.0
        self.current_drawdown = 0.0

        # Position tracking
        self.open_positions: Dict[str, Any] = {}  # symbol -> position info
        self.position_count = 0

        # Trading state
        self.daily_start_balance = 0.0
        self.consecutive_losses = 0
        self.last_price = 0.0
        self.circuit_breaker_triggered = False

        # Bridge connection watchdog (Task 4)
        self.failed_attempts: int = 0
        self.max_retries: int = 5
        self.retry_interval: int = 600  # 10 minutes
        self.is_paused: bool = False
        self.last_failure_time: Optional[float] = None

        # Slippage analysis integration (Task 8)
        self.slippage_analysis = SlippageAnalysis()

    def check_safety_limits(self, current_balance: float, current_price: float) -> bool:
        """
        Check safety limits before executing order.

        Returns:
            True if safe to trade, False if limits exceeded
        """
        return safe_operation(
            logger=logger,
            operation=lambda: self._check_safety_limits_impl(current_balance, current_price),
            context="safety_limits_check",
            default_result=False,  # Default to unsafe on error
        )

    def _check_safety_limits_impl(self, current_balance: float, current_price: float) -> bool:
        """Implementation of safety limits check."""
        # Daily loss limit
        if self.daily_start_balance > 0:
            daily_loss = (
                self.daily_start_balance - current_balance
            ) / self.daily_start_balance
            if daily_loss > self.daily_loss_limit:
                logger.error(
                    f"Daily loss limit exceeded: {daily_loss:.2%} > {self.daily_loss_limit:.2%}"
                )
                return False

        # Maximum Drawdown (MDD) limit
        if self.peak_balance > 0:
            self.current_drawdown = (
                self.peak_balance - current_balance
            ) / self.peak_balance
            if self.current_drawdown > self.max_drawdown_limit:
                logger.error(
                    f"Maximum drawdown limit exceeded: {self.current_drawdown:.2%} > {self.max_drawdown_limit:.2%}"
                )
                return False

        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            self.current_drawdown = 0.0

        # Circuit breaker
        if self.last_price > 0:
            price_change = abs(current_price - self.last_price) / self.last_price
            if price_change > self.circuit_breaker_threshold:
                logger.error(
                    f"Circuit breaker triggered: price change {price_change:.2%} > {self.circuit_breaker_threshold:.2%}"
                )
                self.circuit_breaker_triggered = True
                return False

        return True

    def check_position_limits(
        self, symbol: str, quantity: float, current_balance: float, current_price: float
    ) -> bool:
        """
        Check position size and count limits.

        Args:
            symbol: Trading pair
            quantity: Order quantity
            current_balance: Current portfolio balance
            current_price: Current market price

        Returns:
            True if position limits allow trade, False otherwise
        """
        return safe_operation(
            logger=logger,
            operation=lambda: self._check_position_limits_impl(symbol, quantity, current_balance, current_price),
            context="position_limits_check",
            default_result=False,  # Default to unsafe on error
        )

    def _check_position_limits_impl(
        self, symbol: str, quantity: float, current_balance: float, current_price: float
    ) -> bool:
        """Implementation of position limits check."""
        # Check order size limits
        if quantity < self.min_order_size:
            logger.error(f"Order size {quantity} below minimum {self.min_order_size}")
            return False
        if quantity > self.max_order_size:
            logger.error(f"Order size {quantity} above maximum {self.max_order_size}")
            return False

        # Check position size limit (% of portfolio)
        position_value = quantity * current_price
        position_pct = position_value / current_balance if current_balance > 0 else 1.0
        if position_pct > self.max_position_size:
            logger.error(
                f"Position size {position_pct:.2%} exceeds limit {self.max_position_size:.2%}"
            )
            return False

        # Check maximum open positions
        if (
            symbol not in self.open_positions
            and self.position_count >= self.max_open_positions
        ):
            logger.error(f"Maximum open positions ({self.max_open_positions}) reached")
            return False

        return True

    def update_position_tracking(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> None:
        """
        Update position tracking after order execution.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Execution price
        """
        if side == "buy":
            if symbol not in self.open_positions:
                self.open_positions[symbol] = {
                    "quantity": quantity,
                    "avg_price": price,
                    "unrealized_pnl": 0.0,
                }
                self.position_count += 1
            else:
                # Average down/up existing position
                existing_qty = self.open_positions[symbol]["quantity"]
                existing_price = self.open_positions[symbol]["avg_price"]
                total_qty = existing_qty + quantity
                avg_price = (
                    (existing_qty * existing_price) + (quantity * price)
                ) / total_qty
                self.open_positions[symbol]["quantity"] = total_qty
                self.open_positions[symbol]["avg_price"] = avg_price
        else:  # sell
            if symbol in self.open_positions:
                existing_qty = self.open_positions[symbol]["quantity"]
                if quantity >= existing_qty:
                    # Close position
                    del self.open_positions[symbol]
                    self.position_count -= 1
                else:
                    # Reduce position
                    self.open_positions[symbol]["quantity"] = existing_qty - quantity

    def _handle_api_failure(self, error: Exception) -> None:
        """Handle API connection failure and manage watchdog state"""
        self.failed_attempts += 1
        self.last_failure_time = time.time()
        logger.warning(
            f"API connection failed (attempt {self.failed_attempts}/{self.max_retries}): {error}"
        )

        # Check if we should pause trading
        if self.failed_attempts >= self.max_retries and not self.is_paused:
            self.is_paused = True
            logger.error(
                f"API connection failed {self.max_retries} times - entering emergency mode"
            )
            self._send_discord_alert("⚠️ API接続失敗 (5回連続) → 新規注文を停止しました")

            # Start retry timer
            self._schedule_retry()

    def _schedule_retry(self) -> None:
        """Schedule automatic retry after interval"""

        def retry() -> None:
            logger.info(
                f"Attempting API reconnection after {self.retry_interval} seconds..."
            )
            # Simple connectivity test - try to get balance
            try:
                self.get_balance()
                # Success - reset watchdog
                self.failed_attempts = 0
                self.is_paused = False
                logger.info("API connection restored - resuming trading")
                self._send_discord_alert("✅ API接続が復旧しました。取引を再開します。")
            except Exception as e:
                logger.warning(f"API reconnection failed: {e}")
                # If still failing, schedule another retry
                if self.is_paused:
                    self._schedule_retry()

    def get_watchdog_state(self) -> Dict[str, Any]:
        """Get current watchdog state for persistence"""
        return {
            "failed_attempts": self.failed_attempts,
            "is_paused": self.is_paused,
            "last_failure_time": self.last_failure_time,
            "max_retries": self.max_retries,
            "retry_interval": self.retry_interval,
        }

    def set_watchdog_state(self, state: Dict[str, Any]) -> None:
        """Restore watchdog state from persistence"""
        self.failed_attempts = state.get("failed_attempts", 0)
        self.is_paused = state.get("is_paused", False)
        self.last_failure_time = state.get("last_failure_time")
        self.max_retries = state.get("max_retries", 5)
        self.retry_interval = state.get("retry_interval", 600)

    def _send_discord_alert(self, message: str) -> None:
        """Send alert to Discord webhook"""
        if not self.discord_webhook_url:
            logger.info(f"Discord alert (no webhook configured): {message}")
            return

        try:
            payload = {"content": message}
            response = requests.post(self.discord_webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Discord alert sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> Dict[str, Any]:
        """
        Place live market order via Zaif API.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            quantity: Order quantity

        Returns:
            Order result dict
        """
        # Check if trading is paused due to API failures
        if self.is_paused:
            logger.warning(
                f"Order rejected - trading is paused due to API connection issues ({self.failed_attempts} failures)"
            )
            return {
                "success": False,
                "error": "Trading paused due to API connection issues",
            }

        try:
            current_balance = self.get_balance()
            current_price = self.get_market_price(symbol)
        except Exception as e:
            self._handle_api_failure(e)
            return {"success": False, "error": f"API connection failed: {e}"}

        # Safety checks
        if not self.check_safety_limits(current_balance, current_price):
            return {"success": False, "error": "Safety limits exceeded"}

        # Position control checks
        if not self.check_position_limits(
            symbol, quantity, current_balance, current_price
        ):
            return {"success": False, "error": "Position limits exceeded"}

        # TODO: Implement actual Zaif API call
        # This is a placeholder for the actual API integration
        # Simulate API call delay
        time.sleep(0.1)

        # Validate order parameters
        if quantity <= 0:
            return {"success": False, "error": "Invalid quantity"}

        if current_price <= 0:
            return {"success": False, "error": "Invalid price"}

        # Simulate order execution (90% success rate)
        if random.random() < 0.9:
            # Successful order
            executed_price = current_price * (
                1 + random.uniform(-0.001, 0.001)
            )  # ±0.1% slippage
            executed_quantity = quantity * (
                1 - random.uniform(0, 0.01)
            )  # Up to 1% slippage

            # Record slippage for analysis (Task 8)
            self.slippage_analysis.add_slippage_event(
                symbol=symbol,
                side=side,
                intended_price=current_price,
                executed_price=executed_price,
                quantity=executed_quantity,
            )

            logger.info(
                f"Zaif API order executed: {side} {executed_quantity:.6f} {symbol} @ {executed_price:.2f}"
            )

            # Update position tracking
            self.update_position_tracking(
                symbol, side, executed_quantity, executed_price
            )

            return {
                "success": True,
                "order_id": f"zaif_{int(time.time() * 1000)}",
                "symbol": symbol,
                "side": side,
                "quantity": executed_quantity,
                "price": executed_price,
                "timestamp": time.time(),
            }
        else:
            # Failed order
            error_msg = random.choice(
                [
                    "Insufficient balance",
                    "Market temporarily unavailable",
                    "Order rejected by exchange",
                ]
            )
            logger.error(f"Zaif API order failed: {error_msg}")
            return {"success": False, "error": error_msg}

    def get_balance(self) -> float:
        """Get current balance from Zaif API"""
        # TODO: Implement actual Zaif API balance query
        # This is a placeholder that simulates balance tracking

        # Simulate API call with potential failure for testing watchdog
        time.sleep(0.05)

        # Simulate occasional API failures (5% chance)
        if random.random() < 0.05:
            raise ConnectionError("Simulated Zaif API connection failure")

        # For demo purposes, return a simulated balance
        # In real implementation, this would query Zaif API
        base_balance = 10000.0  # Starting balance

        # Simulate some P&L based on positions
        simulated_pnl = 0.0
        for position in self.open_positions.values():
            # Simple P&L simulation (would be real in production)
            simulated_pnl += position.get("unrealized_pnl", 0)

        current_balance = base_balance + simulated_pnl

        # Update monitoring
        monitor = get_exporter()
        monitor.update_portfolio_metrics(
            current_balance, simulated_pnl, self.current_drawdown
        )

        return max(current_balance, 0)  # Never negative

    def get_market_price(self, symbol: str) -> float:
        """Get current market price from Zaif API"""
        # TODO: Implement actual Zaif API price query
        # This is a placeholder that simulates price feeds

        # Simulate API call with potential failure for testing watchdog
        time.sleep(0.02)

        # Simulate occasional API failures (3% chance)
        if random.random() < 0.03:
            raise TimeoutError("Simulated Zaif API timeout")

        # For demo purposes, simulate BTC/JPY price around 5M JPY
        # In real implementation, this would query Zaif ticker API
        base_price = 5000000.0  # ~5M JPY per BTC

        # Add some random variation (±1%)
        price_variation = random.uniform(-0.01, 0.01)
        current_price = base_price * (1 + price_variation)

        # Update last price for circuit breaker
        self.last_price = current_price

        return current_price

    def get_slippage_analysis(self) -> Dict[str, Any]:
        """Get slippage analysis summary"""
        return self.slippage_analysis.get_summary()

    def reset_slippage_analysis(self) -> None:
        """Reset slippage analysis"""
        self.slippage_analysis = SlippageAnalysis()


class BridgeReplay:
    """
    Replay trading bridge for backtesting and slippage analysis.

    Replays historical orders against market data to analyze slippage impact.
    """

    def __init__(self, market_data: pd.DataFrame, initial_balance: float = 10000.0):
        """
        Initialize bridge replay.

        Args:
            market_data: Historical market data with columns [timestamp, price, volume]
            initial_balance: Starting balance for replay
        """
        self.market_data = market_data.copy()
        self.market_data["timestamp"] = pd.to_datetime(self.market_data["timestamp"])
        self.market_data = self.market_data.sort_values("timestamp").reset_index(
            drop=True
        )

        self.bridge = VirtualTradingBridge(initial_balance=initial_balance)
        self.current_index = 0
        self.replay_results: List[Dict[str, Any]] = []

    def get_market_price_at_time(self, timestamp: datetime) -> Optional[float]:
        """Get market price at specific timestamp"""
        # Find the closest price before or at the timestamp
        mask = self.market_data["timestamp"] <= timestamp
        if not mask.any():
            return None
        return cast(Optional[float], self.market_data[mask]["price"].iloc[-1])

    def replay_order(
        self, symbol: str, side: str, quantity: float, timestamp: datetime
    ) -> Optional[VirtualOrder]:
        """
        Replay a single order at specific timestamp.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            timestamp: Order timestamp

        Returns:
            Executed order or None if no market data available
        """
        price = self.get_market_price_at_time(timestamp)
        if price is None:
            logger.warning(f"No market data available for timestamp {timestamp}")
            return None

        # Execute order through virtual bridge
        order = self.bridge.place_market_order(symbol, side, quantity, price)

        # Record replay result
        if order.filled_price is not None:
            slippage = (order.filled_price - price) / price * 100
        else:
            slippage = 0.0
        result = {
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "market_price": price,
            "executed_price": order.filled_price,
            "slippage": slippage,
            "order": order,
        }
        self.replay_results.append(result)

        return order

    def replay_orders(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Replay multiple orders.

        Args:
            orders: List of order dicts with keys [symbol, side, quantity, timestamp]

        Returns:
            Replay summary statistics
        """
        executed_orders = []
        failed_orders = []

        for order_data in orders:
            order = self.replay_order(**order_data)
            if order:
                executed_orders.append(order)
            else:
                failed_orders.append(order_data)

        # Calculate replay statistics
        total_slippage = sum(abs(r["slippage"]) for r in self.replay_results)
        avg_slippage = (
            total_slippage / len(self.replay_results) if self.replay_results else 0
        )
        max_slippage = max((abs(r["slippage"]) for r in self.replay_results), default=0)

        summary = {
            "total_orders": len(orders),
            "executed_orders": len(executed_orders),
            "failed_orders": len(failed_orders),
            "total_slippage_percent": total_slippage,
            "avg_slippage_percent": avg_slippage,
            "max_slippage_percent": max_slippage,
            "final_balance": self.bridge.get_balance(),
            "slippage_analysis": self.bridge.get_slippage_analysis(),
        }

        return summary

    def get_replay_results(self) -> List[Dict[str, Any]]:
        """Get detailed replay results"""
        return self.replay_results.copy()
