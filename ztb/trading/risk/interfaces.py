from typing import Any, Protocol, runtime_checkable

import pandas as pd

@runtime_checkable
class RiskManagerProtocol(Protocol):
    """Unified RiskManager interface used across systems.

    This Protocol defines the minimum set of attributes/methods that other
    components expect from a RiskManager implementation. Implementations may
    provide additional methods but should implement these to maintain
    compatibility.
    """

    test_mode: bool
    portfolio_value: float

    def should_open_position(
        self,
        signal_strength: float,
        market_volatility: float,
        current_portfolio_value: float,
    ) -> bool:
        ...

    def should_close_position(
        self,
        position_data: dict[str, Any],
        current_price: float,
        current_portfolio_value: float,
    ) -> tuple[bool, str]:
        ...

    def get_risk_adjusted_position_size(
        self, signal_strength: float, market_volatility: float
    ) -> float:
        ...

    def calculate_atr_stop_levels(
        self, data: pd.DataFrame, entry_price: float, position_type: str
    ) -> tuple[float, float]:
        ...

    def update_risk_metrics(
        self, trade_result: dict[str, Any] | None = None
    ) -> None:
        ...

    def reset(self) -> None:
        ...

    # NOTE: When migrating legacy risk manager implementations, prefer to implement
    # this protocol directly on the RiskManager class. If that isn't possible, use
    # `ensure_risk_manager_protocol(obj)` provided in `ztb.trading.risk.compat` to
    # wrap legacy implementations at instantiation sites to preserve backward
    # compatibility during progressive migration.
