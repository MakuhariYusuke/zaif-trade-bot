"""
Position Manager - Handles position management for the trading environment.

This module separates position-related logic from the main environment class,
including execution of trades, trade accounting, and position state tracking.
"""

from typing import Any, Callable, Optional

from ztb.risk.risk_manager import RiskManager
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.execution.model import ExecutionModel
from ztb.trading.risk.compat import ensure_risk_manager_protocol
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PositionManager:
    """
    Manages trading positions including opening, closing, and PnL calculation.

    This class handles:
    - Position opening and closing
    - Entry and exit cost calculation
    - Realized and unrealized PnL tracking
    - Trade counting and consecutive trade tracking
    """

    def __init__(
        self,
        config: Any,  # EnvironmentConfig
        get_price_callback: Callable[[], float],  # Function to get current price
        execution_model: Optional[ExecutionModel] = None,
    ):
        """
        Initialize PositionManager.

        Args:
            config: Environment configuration
            get_price_callback: Callback function to get current price
            execution_model: Optional execution model for realistic simulation
        """
        self.config = config
        self._get_price = get_price_callback
        self.execution_model = execution_model

        # Position state
        self.position: float = 0.0
        self.entry_price: float = 0.0

        # PnL tracking
        self.realized_pnl: float = 0.0
        self.total_pnl: float = 0.0  # For backward compatibility

        # Trade tracking
        self.trades_count: int = 0
        self._last_trade_step: int = -1
        self._consecutive_trade_steps: int = 0

        # Risk management (v435 enhancement)
        risk_config = getattr(config, "risk_management", {})
        # Ensure the instantiated risk manager conforms to the RiskManagerProtocol
        self.risk_manager = ensure_risk_manager_protocol(RiskManager(risk_config))

    def execute_action(
        self,
        action: int,
        current_step: int,
        min_holding_period: int = 0,
        df: Optional[Any] = None,
        atr: float = 0.0,
    ) -> float:
        """
        Execute trading action.

        allow_reverse=True (default): Traditional behavior
            - SELL from Long: Close→immediately Short
            - BUY from Short: Close→immediately Long

        allow_reverse=False: No reversal mode
            - SELL from Long: Close only (back to Flat)
            - BUY from Short: Close only (back to Flat)
            - SELL/BUY from Flat: Opens normally

        Args:
            action: Action to execute (0=HOLD, 1=BUY, 2=SELL)
            current_step: Current step number
            min_holding_period: Minimum steps to hold position before reversal (prevents allow_reverse during this period)

        Returns:
            trade_pnl: PnL from this specific trade INCLUDING entry fees (negative for new positions)
        """
        if action == ACTION_HOLD:
            self._consecutive_trade_steps = 0
            return 0.0

        # Check if we're within min_holding_period
        min_holding_period = min_holding_period or 0  # Handle None case
        within_min_holding = (
            self._last_trade_step >= 0
            and current_step - self._last_trade_step < min_holding_period
        )
        enforce_reverse_cooldown = getattr(
            self.config, "enforce_reverse_cooldown", False
        )

        trade_pnl = 0.0  # Track PnL from this action (closes + entries)

        if action == ACTION_BUY:
            if self.position < 0:  # Short position held
                trade_pnl = self.close_position(current_step=current_step, atr=atr)
                self._last_trade_step = current_step  # Update last trade step for close
                self._consecutive_trade_steps += 1

                # Only open Long immediately if allow_reverse=True AND not within min_holding_period
                if self.config.allow_reverse and (
                    not enforce_reverse_cooldown or not within_min_holding
                ):
                    entry_cost = self.open_position(1, current_step, df, atr)
                    trade_pnl -= entry_cost  # Entry fee is negative PnL

            elif self.position == 0:  # Flat
                entry_cost = self.open_position(1, current_step, df, atr)
                trade_pnl -= entry_cost  # Entry fee is negative PnL
                self._consecutive_trade_steps += 1
            # position > 0 (already Long): do nothing

        elif action == ACTION_SELL:
            if self.position > 0:  # Long position held
                trade_pnl = self.close_position(current_step=current_step, atr=atr)
                self._last_trade_step = current_step  # Update last trade step for close
                self._consecutive_trade_steps += 1

                # Only open Short immediately if allow_reverse=True AND not within min_holding_period
                if self.config.allow_reverse and (
                    not enforce_reverse_cooldown or not within_min_holding
                ):
                    entry_cost = self.open_position(-1, current_step, df, atr)
                    trade_pnl -= entry_cost  # Entry fee is negative PnL

            elif self.position == 0:  # Flat
                entry_cost = self.open_position(-1, current_step, df, atr)
                trade_pnl -= entry_cost  # Entry fee is negative PnL
                self._consecutive_trade_steps += 1
            # position < 0 (already Short): do nothing

        return trade_pnl

    def open_position(
        self,
        direction: int,
        current_step: int,
        df: Optional[Any] = None,
        atr: float = 0.0,
    ) -> float:
        """
        Open position (entry cost immediately reflected).

        🔧 CRITICAL FIX: 少額取引対応
        - 実口座では1 mBTC (0.001 BTC ≈ 18,000円) 程度の少額取引が必要
        - max_position_size は理想的なサイズだが、資金不足の場合は利用可能資金で調整
        - これにより、資金がmax_position_size分に満たなくても取引可能になる

        Args:
            direction: Position direction (+1 for Long, -1 for Short)
            current_step: Current step number

        Returns:
            Entry cost (fee paid to open position)
        """
        current_price = self._get_price()
        max_position_size = getattr(self.config, "max_position_size", 1.0)

        # 🔧 少額取引対応: 利用可能資金に基づいてポジションサイズを調整
        initial_portfolio = getattr(self.config, "initial_portfolio_value", 200000.0)
        available_funds = initial_portfolio + self.realized_pnl
        transaction_cost = float(self.config.transaction_cost)

        # 基本ポジションサイズ
        base_position_size = getattr(self.config, "max_position_size", 1.0)

        # リスク管理適用 (v435 enhancement)
        current_price = self._get_price()

        # Determine execution price
        if self.execution_model:
            # Use model to simulate execution
            # Note: We don't know the exact size yet, so we simulate with base size 1.0 to get price
            action_type = "buy" if direction > 0 else "sell"
            sim_result = self.execution_model.simulate_execution(
                action_type=action_type,
                requested_price=current_price,
                requested_size=1.0,  # Placeholder for price estimation
                current_atr=atr,
            )
            execution_price = sim_result.executed_price
            slippage_rate = sim_result.slippage_rate
        else:
            # Apply slippage to execution price (Legacy)
            slippage_rate = 0.0
            if (
                hasattr(self.config, "exchange_profile")
                and self.config.exchange_profile
            ):
                slippage_rate = self.config.exchange_profile.slippage_rate
            elif hasattr(self.config, "slippage"):
                slippage_rate = float(self.config.slippage)

            # Buy: price increases (worse), Sell: price decreases (worse)
            execution_price = (
                current_price * (1 + slippage_rate)
                if direction > 0
                else current_price * (1 - slippage_rate)
            )

        portfolio_value = initial_portfolio + self.realized_pnl

        risk_adjusted = self.risk_manager.calculate_risk_adjusted_position(
            base_position=base_position_size,
            current_price=execution_price,  # Use execution price for sizing
            portfolio_value=portfolio_value,
            atr=atr,
            df=df,
            step=current_step,
        )

        actual_position_size = risk_adjusted["adjusted_position"]

        # ログ出力
        if risk_adjusted["control_active"]:
            logger.debug(f"Risk control applied: {risk_adjusted['reasons']}")

        # 理想的なコスト（リスク調整済みサイズ））

        # 実際に購入可能なサイズ（利用可能資金の90%まで）
        affordable_funds = available_funds * 0.9
        affordable_size = affordable_funds / (execution_price * (1 + transaction_cost))

        # 実際のポジションサイズ: 小さい方を採用（資金制約と設定値の両方を満たす）
        actual_position_size = min(max_position_size, affordable_size)

        # 最小取引単位チェック (0.001 BTC = 1 mBTC ≈ 5,000円)
        min_trade_size = 0.001
        if actual_position_size < min_trade_size:
            logger.warning(
                "Insufficient funds for minimum trade size: available=%.2f, required=%.2f (min_size=%.4f BTC)",
                available_funds,
                min_trade_size * execution_price * (1 + transaction_cost),
                min_trade_size,
            )
            actual_position_size = min_trade_size  # 最小単位で取引試行

        # Calculate entry cost based on actual size
        if self.execution_model:
            action_type = "buy" if direction > 0 else "sell"
            # Re-simulate with actual size to get accurate fee and potentially size-dependent slippage
            sim_result = self.execution_model.simulate_execution(
                action_type=action_type,
                requested_price=current_price,
                requested_size=actual_position_size,
                current_atr=atr,
            )
            execution_price = sim_result.executed_price
            entry_cost = sim_result.fee
        else:
            trade_value = abs(float(actual_position_size)) * execution_price

            # Use ExchangeProfile fee model if available
            if (
                hasattr(self.config, "exchange_profile")
                and self.config.exchange_profile
            ):
                trade_type = "buy" if direction > 0 else "sell"
                entry_cost = self.config.exchange_profile.fee_model.calculate_fee(
                    trade_value, trade_type
                )
            else:
                # Fallback to legacy transaction_cost
                entry_cost = trade_value * float(self.config.transaction_cost)

        # Check if we have enough funds
        if available_funds < entry_cost:
            logger.warning(
                "Insufficient funds for entry cost: available=%.2f, required=%.2f",
                available_funds,
                entry_cost,
            )
            return 0.0

        # Deduct entry cost from realized PnL
        self.realized_pnl -= entry_cost
        self.total_pnl = self.realized_pnl

        # Open position with actual size
        self.position = direction * actual_position_size
        self.entry_price = execution_price  # Use execution price as entry price
        self.trades_count += 1
        self._last_trade_step = current_step

        logger.debug(
            "Opened %s position: size=%.4f (max=%.4f, affordable=%.4f), price=%.2f (slip=%.4f), cost=%.2f, funds=%.2f",
            "Long" if direction > 0 else "Short",
            actual_position_size,
            max_position_size,
            affordable_size,
            execution_price,
            slippage_rate,
            entry_cost,
            available_funds,
        ) if self.trades_count % 5 == 0 else None

        return entry_cost

    def close_position(
        self, current_step: Optional[int] = None, atr: float = 0.0
    ) -> float:
        """
        Close current position.

        Args:
            current_step: Current step number (optional, for trade tracking)
            atr: Current ATR for slippage calculation

        Returns:
            Realized PnL from closing the position
        """
        if self.position == 0:
            return 0.0

        # Calculate realized PnL before closing
        current_price = self._get_price()

        if self.execution_model:
            # Closing Long (SELL) or Closing Short (BUY)
            action_type = "sell" if self.position > 0 else "buy"
            sim_result = self.execution_model.simulate_execution(
                action_type=action_type,
                requested_price=current_price,
                requested_size=abs(self.position),
                current_atr=atr,
            )
            execution_price = sim_result.executed_price
            slippage_rate = sim_result.slippage_rate
            exit_cost = sim_result.fee
        else:
            # Apply slippage to execution price (Legacy)
            slippage_rate = 0.0
            if (
                hasattr(self.config, "exchange_profile")
                and self.config.exchange_profile
            ):
                slippage_rate = self.config.exchange_profile.slippage_rate
            elif hasattr(self.config, "slippage"):
                slippage_rate = float(self.config.slippage)

            # Closing Long (SELL): price decreases (worse)
            # Closing Short (BUY): price increases (worse)
            execution_price = (
                current_price * (1 - slippage_rate)
                if self.position > 0
                else current_price * (1 + slippage_rate)
            )

            trade_value = abs(self.position) * execution_price

            # Use ExchangeProfile fee model if available
            if (
                hasattr(self.config, "exchange_profile")
                and self.config.exchange_profile
            ):
                # Closing Long (SELL) or Closing Short (BUY)
                trade_type = "sell" if self.position > 0 else "buy"
                exit_cost = self.config.exchange_profile.fee_model.calculate_fee(
                    trade_value, trade_type
                )
            else:
                # Fallback to legacy transaction_cost
                exit_cost = trade_value * float(self.config.transaction_cost)

        price_change = execution_price - self.entry_price
        realized_trade_pnl = float(self.position) * price_change

        realized_trade_pnl -= exit_cost

        # Accumulate realized PnL
        self.realized_pnl += realized_trade_pnl
        self.total_pnl = self.realized_pnl

        # Update trade tracking if step provided
        if current_step is not None:
            self._last_trade_step = current_step
            self._consecutive_trade_steps += 1

        logger.debug(
            "Closed %s position: price=%.2f (slip=%.4f), entry=%.2f, pnl=%.2f, cost=%.2f",
            "Long" if self.position > 0 else "Short",
            execution_price,
            slippage_rate,
            self.entry_price,
            realized_trade_pnl + exit_cost,  # PnL before cost
            exit_cost,
        ) if self.trades_count % 5 == 0 else None

        self.trades_count += 1
        self.position = 0.0
        self.entry_price = 0.0

        return float(realized_trade_pnl)

    def calculate_unrealized_pnl(self) -> float:
        """
        Calculate unrealized PnL for current open position.

        Realized PnL is accumulated in close_position().
        This method only returns unrealized PnL for reward calculation.

        Returns:
            Unrealized PnL (price change only, no exit cost yet)
        """
        if self.position == 0:
            return 0.0

        current_price = self._get_price()
        price_change = current_price - self.entry_price

        # Unrealized PnL (price change only, no exit cost yet)
        unrealized_pnl = float(self.position) * price_change

        return float(unrealized_pnl)

    def get_position_info(self) -> dict[str, Any]:
        """
        Get current position information.

        Returns:
            Dictionary with position details
        """
        return {
            "position": self.position,
            "entry_price": self.entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.calculate_unrealized_pnl(),
            "total_pnl": self.total_pnl,
            "trades_count": self.trades_count,
            "consecutive_trade_steps": self._consecutive_trade_steps,
        }

    def reset(self) -> None:
        """Reset position state."""
        self.position = 0.0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.total_pnl = 0.0
        self.trades_count = 0
        self._last_trade_step = -1
        self._consecutive_trade_steps = 0


__all__ = ["PositionManager"]
