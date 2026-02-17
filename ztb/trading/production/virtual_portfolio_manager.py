"""
V433 Phase 5: Paper Trading Layer - Virtual Portfolio Manager

仮想ポートフォリオ管理システム。実資金を使わない取引シミュレーションを行い、
リアルタイムでパフォーマンスを追跡する。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from ztb.trading.production.state_persistence import (
    read_state_payload,
    write_state_payload,
)


# Mock classes for testing


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal] = None
    average_price: Decimal
    current_price: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")


@dataclass
class Trade:
    trade_id: str
    order_id: str
    symbol: str
class RiskManager:
    pass


class PortfolioState(Enum):
    """ポートフォリオ状態"""

    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal = field(default=Decimal("0"))
    timestamp: datetime = field(default_factory=datetime.now)

    def update_pnl(self, current_price: Decimal) -> None:
        """PnL更新"""
        self.current_price = current_price
        if self.side == OrderSide.BUY:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SELL
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity


@dataclass
class VirtualTrade:
    """仮想取引"""

    trade_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    commission: Decimal = field(default=Decimal("0"))
    realized_pnl: Decimal = field(default=Decimal("0"))


@dataclass
class PortfolioMetrics:
    """ポートフォリオ指標"""

    total_value: Decimal
    cash_balance: Decimal
    total_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    win_rate: float
    max_drawdown: Decimal
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int


class VirtualPortfolioManager:
    """
    仮想ポートフォリオマネージャー

    実資金を使わない取引シミュレーションを行い、
    リアルタイムでパフォーマンスを追跡する。
    """

    def __init__(
        self,
        initial_balance: Decimal = Decimal("100000"),
        commission_rate: Decimal = Decimal("0.001"),
        max_position_size: Decimal = Decimal("0.1"),
        max_drawdown_limit: Decimal = Decimal("0.2"),
    ):
        """
        初期化

        Args:
            initial_balance: 初期残高
            commission_rate: 手数料率
            max_position_size: 最大ポジションサイズ（ポートフォリオ比率）
            max_drawdown_limit: 最大ドローダウン制限
        """
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.max_position_size = max_position_size
        self.max_drawdown_limit = max_drawdown_limit

        # ポートフォリオ状態
        self.state = PortfolioState.ACTIVE
        self.cash_balance = initial_balance
        self.positions: Dict[str, VirtualPosition] = {}
        self.trades: List[VirtualTrade] = []
        self.portfolio_history: List[PortfolioMetrics] = []

        # パフォーマンス追跡
        self.peak_value = initial_balance
        self.current_drawdown = Decimal("0")
        self.total_realized_pnl = Decimal("0")
        self.total_unrealized_pnl = Decimal("0")

        # 取引カウンター
        self.trade_counter = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # ロギング
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"Virtual Portfolio Manager initialized with balance: {initial_balance}"
        )

    def place_order(self, order: Order) -> bool:
        """
        仮想注文実行

        Args:
            order: 注文オブジェクト

        Returns:
            bool: 注文成功フラグ
        """
        if self.state != PortfolioState.ACTIVE:
            self.logger.warning(f"Portfolio is not active. Current state: {self.state}")
            return False

        # リスクチェック
        if not self._validate_order(order):
            return False

        # 注文実行
        success = self._execute_order(order)
        if success:
            self._update_portfolio_metrics()
            self.logger.info(
                f"Order executed: {order.symbol} {order.side.value} {order.quantity} @ {order.price}"
            )

        return success

    def _validate_order(self, order: Order) -> bool:
        """
        注文バリデーション

        Args:
            order: 注文オブジェクト

        Returns:
            bool: バリデーション結果
        """
        # 基本バリデーション
        if order.quantity <= 0 or order.price <= 0:
            self.logger.warning("Invalid order parameters")
            return False

        # ドローダウンチェック
        if self.current_drawdown >= self.max_drawdown_limit:
            self.logger.warning(f"Max drawdown limit reached: {self.current_drawdown}")
            return False

        # ポジションサイズチェック
        portfolio_value = self.get_portfolio_value()
        order_value = order.quantity * order.price

        if order_value > portfolio_value * self.max_position_size:
            self.logger.warning(
                f"Order size exceeds max position limit: {order_value} > {portfolio_value * self.max_position_size}"
            )
            return False

        # 残高チェック（買い注文の場合）
        if order.side == OrderSide.BUY:
            commission = order_value * self.commission_rate
            total_cost = order_value + commission
            if total_cost > self.cash_balance:
                self.logger.warning(
                    f"Insufficient balance: {total_cost} > {self.cash_balance}"
                )
                return False

        # ポジションチェック（売り注文の場合）
        elif order.side == OrderSide.SELL:
            if order.symbol not in self.positions:
                self.logger.warning(f"No position found for symbol: {order.symbol}")
                return False

            current_position = self.positions[order.symbol]
            if (
                current_position.side != OrderSide.BUY
                or current_position.quantity < order.quantity
            ):
                self.logger.warning(
                    f"Insufficient position: {current_position.quantity} < {order.quantity}"
                )
                return False

        return True

    def _execute_order(self, order: Order) -> bool:
        """
        注文実行

        Args:
            order: 注文オブジェクト

        Returns:
            bool: 実行成功フラグ
        """
        try:
            self.trade_counter += 1
            trade_id = f"VT_{self.trade_counter:06d}"

            # 手数料計算
            order_value = order.quantity * order.price
            commission = order_value * self.commission_rate

            if order.side == OrderSide.BUY:
                # 買い注文
                total_cost = order_value + commission
                self.cash_balance -= total_cost

                # 新規ポジションまたは追加
                if order.symbol in self.positions:
                    existing = self.positions[order.symbol]
                    if existing.side == OrderSide.BUY:
                        # ポジション追加
                        total_quantity = existing.quantity + order.quantity
                        avg_price = (
                            (existing.quantity * existing.entry_price)
                            + (order.quantity * order.price)
                        ) / total_quantity
                        existing.quantity = total_quantity
                        existing.entry_price = avg_price
                        existing.update_pnl(order.price)
                    else:
                        # 反対ポジション（クローズ）
                        self._close_position(order.symbol, order.quantity, order.price)
                else:
                    # 新規ポジション
                    self.positions[order.symbol] = VirtualPosition(
                        symbol=order.symbol,
                        side=OrderSide.BUY,
                        quantity=order.quantity,
                        entry_price=order.price,
                        current_price=order.price,
                    )

            else:  # SELL
                # 売り注文（ポジションクローズ）
                realized_pnl = self._close_position(
                    order.symbol, order.quantity, order.price
                )
                self.cash_balance += order_value - commission
                self.total_realized_pnl += realized_pnl

                # 勝敗判定
                if realized_pnl > 0:
                    self.winning_trades += 1
                elif realized_pnl < 0:
                    self.losing_trades += 1

            # 取引記録
            trade = VirtualTrade(
                trade_id=trade_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                timestamp=datetime.now(),
                commission=commission,
                realized_pnl=self.total_realized_pnl
                if order.side == OrderSide.SELL
                else Decimal("0"),
            )
            self.trades.append(trade)

            return True

        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            return False

    def _close_position(
        self, symbol: str, quantity: Decimal, price: Decimal
    ) -> Decimal:
        """
        ポジションクローズ

        Args:
            symbol: シンボル
            quantity: クローズ数量
            price: クローズ価格

        Returns:
            Decimal: 実現PnL
        """
        if symbol not in self.positions:
            return Decimal("0")

        position = self.positions[symbol]
        close_quantity = min(quantity, position.quantity)
        close_ratio = close_quantity / position.quantity

        # 実現PnL計算
        if position.side == OrderSide.BUY:
            realized_pnl = (price - position.entry_price) * close_quantity
        else:
            realized_pnl = (position.entry_price - price) * close_quantity

        # ポジション更新
        position.quantity -= close_quantity
        if position.quantity <= 0:
            del self.positions[symbol]
        else:
            # 部分クローズの場合、entry_priceは変更しない
            position.update_pnl(price)

        return realized_pnl

    def update_prices(self, price_updates: Dict[str, Decimal]) -> None:
        """
        価格更新

        Args:
            price_updates: 価格更新データ {symbol: price}
        """
        for symbol, price in price_updates.items():
            if symbol in self.positions:
                self.positions[symbol].update_pnl(price)

        self._update_portfolio_metrics()

    def get_portfolio_value(self) -> Decimal:
        """
        ポートフォリオ総価値取得

        Returns:
            Decimal: ポートフォリオ総価値
        """
        total_value = self.cash_balance
        for position in self.positions.values():
            total_value += position.quantity * position.current_price
        return total_value

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """
        ポートフォリオ指標取得

        Returns:
            PortfolioMetrics: ポートフォリオ指標
        """
        portfolio_value = self.get_portfolio_value()
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        total_trades = len([t for t in self.trades if t.realized_pnl != 0])
        win_rate = self.winning_trades / total_trades if total_trades > 0 else 0.0

        # ドローダウン計算
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.current_drawdown = Decimal("0")
        else:
            self.current_drawdown = (
                self.peak_value - portfolio_value
            ) / self.peak_value

        # シャープレシオ（簡易計算）
        returns = (
            [m.total_pnl for m in self.portfolio_history[-30:]]
            if self.portfolio_history
            else [0]
        )
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            sharpe_ratio = avg_return / variance**0.5 if variance > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return PortfolioMetrics(
            total_value=portfolio_value,
            cash_balance=self.cash_balance,
            total_pnl=self.total_realized_pnl + unrealized_pnl,
            realized_pnl=self.total_realized_pnl,
            unrealized_pnl=unrealized_pnl,
            win_rate=win_rate,
            max_drawdown=self.current_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
        )

    def _update_portfolio_metrics(self) -> None:
        """ポートフォリオ指標更新"""
        metrics = self.get_portfolio_metrics()
        self.portfolio_history.append(metrics)

        # 最新の指標のみ保持（メモリ節約）
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-500:]

    def pause_trading(self) -> None:
        """取引一時停止"""
        self.state = PortfolioState.PAUSED
        self.logger.info("Trading paused")

    def resume_trading(self) -> None:
        """取引再開"""
        self.state = PortfolioState.ACTIVE
        self.logger.info("Trading resumed")

    def stop_trading(self) -> None:
        """取引停止"""
        self.state = PortfolioState.STOPPED
        self.logger.info("Trading stopped")

    def get_positions(self) -> Dict[str, VirtualPosition]:
        """
        ポジション取得

        Returns:
            Dict[str, VirtualPosition]: ポジション辞書
        """
        return self.positions.copy()

    def get_trades(self, limit: Optional[int] = None) -> List[VirtualTrade]:
        """
        取引履歴取得

        Args:
            limit: 取得件数制限

        Returns:
            List[VirtualTrade]: 取引履歴
        """
        trades = self.trades
        if limit:
            trades = trades[-limit:]
        return trades.copy()

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            "initial_balance": str(self.initial_balance),
            "cash_balance": str(self.cash_balance),
            "positions": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side.value,
                    "quantity": str(pos.quantity),
                    "entry_price": str(pos.entry_price),
                    "current_price": str(pos.current_price),
                    "unrealized_pnl": str(pos.unrealized_pnl),
                    "timestamp": pos.timestamp.isoformat(),
                }
                for pos in self.positions.values()
            ],
            "trades": [
                {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "quantity": str(trade.quantity),
                    "price": str(trade.price),
                    "timestamp": trade.timestamp.isoformat(),
                    "commission": str(trade.commission),
                    "realized_pnl": str(trade.realized_pnl),
                }
                for trade in self.trades
            ],
            "peak_value": str(self.peak_value),
            "current_drawdown": str(self.current_drawdown),
            "total_realized_pnl": str(self.total_realized_pnl),
            "trade_counter": self.trade_counter,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "state": self.state.value,
        }

        write_state_payload(filepath, state)

        self.logger.info(f"Portfolio state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """
        状態読み込み

        Args:
            filepath: 読み込みファイルパス

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            state = read_state_payload(filepath)

            self.initial_balance = Decimal(state["initial_balance"])
            self.cash_balance = Decimal(state["cash_balance"])
            self.peak_value = Decimal(
                state.get("peak_value", str(self.initial_balance))
            )
            self.current_drawdown = Decimal(state.get("current_drawdown", "0"))
            self.total_realized_pnl = Decimal(state.get("total_realized_pnl", "0"))
            self.trade_counter = state.get("trade_counter", 0)
            self.winning_trades = state.get("winning_trades", 0)
            self.losing_trades = state.get("losing_trades", 0)
            self.state = PortfolioState(state.get("state", "active"))

            # ポジション復元
            self.positions = {}
            for pos_data in state.get("positions", []):
                position = VirtualPosition(
                    symbol=pos_data["symbol"],
                    side=OrderSide(pos_data["side"]),
                    quantity=Decimal(pos_data["quantity"]),
                    entry_price=Decimal(pos_data["entry_price"]),
                    current_price=Decimal(pos_data["current_price"]),
                    unrealized_pnl=Decimal(pos_data["unrealized_pnl"]),
                    timestamp=datetime.fromisoformat(pos_data["timestamp"]),
                )
                self.positions[pos_data["symbol"]] = position

            # 取引履歴復元
            self.trades = []
            for trade_data in state.get("trades", []):
                trade = VirtualTrade(
                    trade_id=trade_data["trade_id"],
                    symbol=trade_data["symbol"],
                    side=OrderSide(trade_data["side"]),
                    quantity=Decimal(trade_data["quantity"]),
                    price=Decimal(trade_data["price"]),
                    timestamp=datetime.fromisoformat(trade_data["timestamp"]),
                    commission=Decimal(trade_data["commission"]),
                    realized_pnl=Decimal(trade_data["realized_pnl"]),
                )
                self.trades.append(trade)

            self.logger.info(f"Portfolio state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load portfolio state: {e}")
            return False
