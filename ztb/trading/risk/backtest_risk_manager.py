from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ztb.trading.risk.interfaces import RiskManagerProtocol
from ztb.utils.cache_utils import TTLCache


class BacktestRiskManager(RiskManagerProtocol):
    """A compact RiskManager implementation used by backtests.

    This class provides a simple set of functionalities used by the backtest
    adapters while implementing the RiskManagerProtocol for compatibility
    with other risk management consumers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_drawdown_limit = float(self.config.get("max_drawdown_limit", 0.15))
        self.max_position_size = float(self.config.get("max_position_size", 0.1))
        self.stop_loss_atr_multiplier = float(
            self.config.get("stop_loss_atr_multiplier", 2.0)
        )
        self.take_profit_atr_multiplier = float(
            self.config.get("take_profit_atr_multiplier", 4.0)
        )
        self.max_consecutive_losses = int(self.config.get("max_consecutive_losses", 3))
        self.circuit_breaker_threshold = float(
            self.config.get("circuit_breaker_threshold", 0.05)
        )

        # Tracking
        self.current_drawdown = 0.0
        self.consecutive_losses = 0
        self.portfolio_value = float(self.config.get("initial_portfolio_value", 1.0))
        self.circuit_breaker_active = False
        self.open_positions: Dict[str, Dict[str, Any]] = {}

        # Caching for ATR and similar values for speed
        self.atr_cache = TTLCache(ttl_seconds=180)
        self.test_mode = bool(self.config.get("test_mode", False))

    def should_open_position(
        self,
        signal_strength: float,
        market_volatility: float,
        current_portfolio_value: float,
    ) -> bool:
        cb_active = getattr(self, "circuit_breaker_active", False)
        if cb_active:
            return False
        current_dd = getattr(self, "current_drawdown", 0.0)
        if current_dd >= self.max_drawdown_limit:
            return False
        if getattr(self, "consecutive_losses", 0) >= self.max_consecutive_losses:
            return False
        if self.test_mode:
            min_signal_strength = 0.0
        else:
            min_signal_strength = 0.6 + (market_volatility * 0.2)
        if signal_strength < min_signal_strength:
            return False
        max_position_value = current_portfolio_value * self.max_position_size
        if max_position_value < current_portfolio_value * 0.01:
            return False
        return True

    def should_close_position(
        self,
        position_data: Dict[str, Any],
        current_price: float,
        current_portfolio_value: float,
    ) -> Tuple[bool, str]:
        position_type = position_data.get("type", "long")
        stop_loss = position_data.get("stop_loss")
        take_profit = position_data.get("take_profit")
        if (
            position_type == "long"
            and stop_loss is not None
            and current_price <= stop_loss
        ):
            return True, "stop_loss"
        if (
            position_type == "short"
            and stop_loss is not None
            and current_price >= stop_loss
        ):
            return True, "stop_loss"
        if (
            position_type == "long"
            and take_profit is not None
            and current_price >= take_profit
        ):
            return True, "take_profit"
        if (
            position_type == "short"
            and take_profit is not None
            and current_price <= take_profit
        ):
            return True, "take_profit"
        if getattr(self, "circuit_breaker_active", False):
            return True, "circuit_breaker"
        if getattr(self, "current_drawdown", 0.0) >= self.max_drawdown_limit:
            return True, "max_drawdown"
        return False, ""

    def get_risk_adjusted_position_size(
        self, signal_strength: float, market_volatility: float
    ) -> float:
        base_size = self.max_position_size
        strength_multiplier = 0.5 + (signal_strength * 0.5)
        volatility_multiplier = 1.0 / (1.0 + market_volatility * 2.0)
        loss_multiplier = max(0.3, 1.0 - (self.consecutive_losses * 0.2))
        position_size = (
            base_size * strength_multiplier * volatility_multiplier * loss_multiplier
        )
        return min(position_size, self.max_position_size)

    def calculate_atr_stop_levels(
        self, data: pd.DataFrame, entry_price: float, position_type: str
    ) -> Tuple[float, float]:
        cache_key = f"atr_{len(data)}_{hash(str(data.index[-1]) if len(data) > 0 else 'empty')}_{position_type}"
        # check cache
        cached_result = self.atr_cache.get(cache_key)
        if cached_result is not None:
            base_atr = cached_result
        else:
            if len(data) < 14 or "atr" not in data.columns:
                base_atr = entry_price * 0.02
            else:
                base_atr = float(data["atr"].iloc[-1])
            self.atr_cache.set(cache_key, base_atr)

        if position_type == "long":
            stop_loss = entry_price - (base_atr * self.stop_loss_atr_multiplier)
            take_profit = entry_price + (base_atr * self.take_profit_atr_multiplier)
        else:
            stop_loss = entry_price + (base_atr * self.stop_loss_atr_multiplier)
            take_profit = entry_price - (base_atr * self.take_profit_atr_multiplier)
        return stop_loss, take_profit

    def update_risk_metrics(
        self, trade_result: Optional[Dict[str, Any]] = None
    ) -> None:
        if trade_result:
            pnl = trade_result.get("pnl", 0)
            if pnl < 0:
                self.consecutive_losses += 1
                self.portfolio_value += pnl
            else:
                self.consecutive_losses = 0
                self.portfolio_value += pnl
        self.current_drawdown = max(0, 1.0 - self.portfolio_value)
        if self.portfolio_value <= (1.0 - self.circuit_breaker_threshold):
            self.circuit_breaker_active = True
        elif self.portfolio_value >= 0.98:
            self.circuit_breaker_active = False

    def reset(self) -> None:
        self.current_drawdown = 0.0
        self.consecutive_losses = 0
        self.portfolio_value = 1.0
        self.circuit_breaker_active = False
        self.open_positions = {}
