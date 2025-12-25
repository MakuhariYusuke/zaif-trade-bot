from typing import TypedDict, Optional, Any

class MarketState(TypedDict, total=False):
    """
    Represents the current state of the market for a single step.
    Used for execution simulation and signal evaluation.
    """
    high: float
    low: float
    close: float
    atr: float
    volume: float
    timestamp: Optional[Any]
