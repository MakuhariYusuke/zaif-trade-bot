from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ztb.trading.risk.interfaces import RiskManagerProtocol


class GenericRiskManagerAdapter(RiskManagerProtocol):
    """Adapter that wraps legacy RiskManager implementations and exposes the
    new RiskManagerProtocol-compatible API. Methods delegate to the underlying
    object where available, otherwise provide conservative default fallbacks.
    """

    def __init__(self, obj: Any):
        self.obj = obj
        # Map commonly used attributes
        self.test_mode = getattr(obj, "test_mode", False)
        self.portfolio_value = float(
            getattr(
                obj, "portfolio_value", getattr(obj, "initial_portfolio_value", 1.0)
            )
        )

    # Helper: check if underlying has attribute
    def _has(self, name: str) -> bool:
        return hasattr(self.obj, name) and callable(getattr(self.obj, name))

    def should_open_position(
        self,
        signal_strength: float,
        market_volatility: float,
        current_portfolio_value: float,
    ) -> bool:
        if self._has("should_open_position"):
            return self.obj.should_open_position(
                signal_strength, market_volatility, current_portfolio_value
            )
        if self._has("can_open_position"):
            return self.obj.can_open_position(signal_strength, market_volatility)
        # Conservative default: allow signals in test_mode, otherwise require medium strength
        if self.test_mode:
            return True
        return signal_strength >= 0.6 and market_volatility < 0.5

    def should_close_position(
        self,
        position_data: Dict[str, Any],
        current_price: float,
        current_portfolio_value: float,
    ) -> Tuple[bool, str]:
        if self._has("should_close_position"):
            return self.obj.should_close_position(
                position_data, current_price, current_portfolio_value
            )
        if self._has("should_exit"):
            return self.obj.should_exit(position_data, current_price)
        # Basic check based on stop / take_profit in position data
        stop = position_data.get("stop_loss")
        tp = position_data.get("take_profit")
        if stop is not None and current_price <= stop:
            return True, "stop_loss"
        if tp is not None and current_price >= tp:
            return True, "take_profit"
        return False, ""

    def get_risk_adjusted_position_size(
        self, signal_strength: float, market_volatility: float
    ) -> float:
        if self._has("get_risk_adjusted_position_size"):
            return self.obj.get_risk_adjusted_position_size(
                signal_strength, market_volatility
            )
        if self._has("get_position_size"):
            return self.obj.get_position_size(signal_strength)
        # Fallback: scale with signal strength
        base = getattr(self.obj, "max_position_size", 0.05)
        return min(base, base * (0.5 + signal_strength / 2.0))

    def calculate_atr_stop_levels(
        self, data: pd.DataFrame, entry_price: float, position_type: str
    ) -> Tuple[float, float]:
        if self._has("calculate_atr_stop_levels"):
            return self.obj.calculate_atr_stop_levels(data, entry_price, position_type)
        if self._has("calculate_stop_levels"):
            return self.obj.calculate_stop_levels(entry_price, position_type)
        # Fallback: use 2% stop and 4% take profit
        return entry_price * 0.98, entry_price * 1.04

    def update_risk_metrics(
        self, trade_result: Optional[Dict[str, Any]] = None
    ) -> None:
        if self._has("update_risk_metrics"):
            return self.obj.update_risk_metrics(trade_result)
        if self._has("on_trade"):
            return self.obj.on_trade(trade_result)
        # No-op fallback
        return None

    def reset(self) -> None:
        if self._has("reset"):
            return self.obj.reset()
        # Best-effort: set some properties
        try:
            setattr(
                self,
                "portfolio_value",
                float(getattr(self.obj, "initial_portfolio_value", 1.0)),
            )
        except Exception:
            pass
        return None


def ensure_risk_manager_protocol(obj: Any) -> RiskManagerProtocol:
    """Return obj if it's already protocol-compatible, otherwise wrap it."""
    if isinstance(obj, RiskManagerProtocol):
        return obj
    return GenericRiskManagerAdapter(obj)
