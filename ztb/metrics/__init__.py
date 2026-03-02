"""Metrics package with lazy export loading."""

from __future__ import annotations

from importlib import import_module

_LAZY_MODULE_ATTRS: dict[str, tuple[str, str]] = {
    "sharpe_ratio": ("ztb.metrics.metrics", "sharpe_ratio"),
    "max_drawdown": ("ztb.metrics.metrics", "max_drawdown"),
    "sortino_ratio": ("ztb.metrics.metrics", "sortino_ratio"),
    "profit_factor": ("ztb.metrics.metrics", "profit_factor"),
    "coefficient_of_variation": ("ztb.metrics.metrics", "coefficient_of_variation"),
    "skewness": ("ztb.metrics.metrics", "skewness"),
    "kurtosis": ("ztb.metrics.metrics", "kurtosis"),
    "test_normality": ("ztb.metrics.metrics", "test_normality"),
    "autocorrelation": ("ztb.metrics.metrics", "autocorrelation"),
    "BacktestMetrics": ("ztb.metrics.metrics", "BacktestMetrics"),
    "calculate_all_metrics": ("ztb.metrics.metrics", "calculate_all_metrics"),
    "classify_market_regime": ("ztb.metrics.metrics", "classify_market_regime"),
    "multi_market_backtest_analysis": ("ztb.metrics.metrics", "multi_market_backtest_analysis"),
    "seasonality_analysis": ("ztb.metrics.metrics", "seasonality_analysis"),
    "MetricsCalculator": ("ztb.metrics.metrics", "MetricsCalculator"),
}

__all__ = [
    "sharpe_ratio",
    "max_drawdown",
    "sortino_ratio",
    "profit_factor",
    "coefficient_of_variation",
    "skewness",
    "kurtosis",
    "test_normality",
    "autocorrelation",
    "BacktestMetrics",
    "calculate_all_metrics",
    "classify_market_regime",
    "multi_market_backtest_analysis",
    "seasonality_analysis",
    "MetricsCalculator",
]

def __getattr__(name: str) -> object:
    target = _LAZY_MODULE_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
