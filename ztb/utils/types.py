#!/usr/bin/env python3
"""
Type definitions and lightweight protocols used across the ZTB codebase.

This module is intentionally small and dependency-light: it exists primarily to
avoid circular imports and to provide shared typing structures.
"""

from __future__ import annotations

from os import PathLike as OsPathLike
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from typing_extensions import TypedDict

# --- Common aliases ---
ActionType = int
PathLike = Union[str, OsPathLike[str]]
JSONSerializable = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


# --- Results / metrics ---
class TrainingResult(TypedDict, total=False):
    training_time: float


class BacktestResult(TypedDict, total=False):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int


class ValidationResult(TypedDict, total=False):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Optional[Dict[str, float]]


class IndicatorInfo(TypedDict, total=False):
    description: str
    talib_available: bool
    parameters: List[str]
    inputs: Optional[List[str]]
    output_range: Optional[Tuple[Optional[float], Optional[float]]]
    interpretation: Optional[str]


class StatsResult(TypedDict, total=False):
    mean: float
    std: float
    ci95: List[float]


class FeatureMetrics(TypedDict, total=False):
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    sample_count: int


# --- Protocols ---
@runtime_checkable
class LoggerProtocol(Protocol):
    def info(self, message: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, message: str, *args: Any, **kwargs: Any) -> None: ...
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None: ...


@runtime_checkable
class FeeModelProtocol(Protocol):
    def calculate_fee(self, trade_value: float, trade_type: str = "buy") -> float: ...
    def get_fee_rate(self, trade_type: str = "buy") -> float: ...


@runtime_checkable
class ThresholdManagerProtocol(Protocol):
    def get_adaptive_threshold(self, metric_name: str, percentile: float = 20.0) -> float: ...
    def get_adaptive_gates(self) -> Dict[str, float]: ...

