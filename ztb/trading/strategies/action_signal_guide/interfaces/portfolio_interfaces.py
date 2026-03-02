"""
Portfolio Optimization Interfaces for Action Signal Guide.

Defines interfaces for portfolio-level strategy optimization and risk
management with shared payload aliases.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.interfaces.common_types import (
    IActionSignalGuideInterface,
    MetricsMap,
    PayloadMap,
    PayloadRecords,
    SeriesMap,
)

class AllocationStrategy(Enum):
    """Portfolio allocation strategies."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"

class RiskMeasure(Enum):
    """Risk measurement methods."""

    VARIANCE = "variance"
    SEMI_VARIANCE = "semi_variance"
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"
    MAX_DRAWDOWN = "max_drawdown"
    EXPECTED_SHORTFALL = "expected_shortfall"

@dataclass
class StrategyPerformance:
    """Performance metrics for a trading strategy."""

    strategy_name: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    timestamp: float
    correlations: dict[str, float] | None = None

@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""

    allocations: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    timestamp: float

@dataclass
class CorrelationMatrix:
    """Strategy correlation matrix."""

    matrix: pd.DataFrame
    rolling_window: int
    last_updated: float

class IStrategyAllocator(IActionSignalGuideInterface):
    """Interface for strategy allocation optimization."""

    @abstractmethod
    def optimize_allocation(
        self,
        strategy_performance: dict[str, StrategyPerformance],
        risk_tolerance: float,
        constraints: PayloadMap,
    ) -> PortfolioAllocation:
        """Optimize portfolio allocation across strategies."""

    @abstractmethod
    def rebalance_portfolio(
        self,
        current_allocation: PortfolioAllocation,
        new_performance_data: dict[str, StrategyPerformance],
        transaction_costs: float,
    ) -> PortfolioAllocation:
        """Rebalance portfolio based on new performance data."""

    @abstractmethod
    def get_allocation_history(self) -> list[PortfolioAllocation]:
        """Get historical allocation decisions."""

class IRiskParityAllocator(IActionSignalGuideInterface):
    """Interface for risk parity allocation strategy."""

    @abstractmethod
    def compute_risk_contributions(
        self, returns: pd.DataFrame, weights: np.ndarray
    ) -> np.ndarray:
        """Compute risk contributions for each asset."""

    @abstractmethod
    def optimize_risk_parity(
        self, returns: pd.DataFrame, target_risk: float | None = None
    ) -> PortfolioAllocation:
        """Optimize portfolio using risk parity approach."""

    @abstractmethod
    def get_risk_parity_metrics(self) -> MetricsMap:
        """Get risk parity specific metrics."""

class ICorrelationManager(IActionSignalGuideInterface):
    """Interface for managing strategy correlations."""

    @abstractmethod
    def update_correlations(self, strategy_returns: SeriesMap) -> CorrelationMatrix:
        """Update correlation matrix with new return data."""

    @abstractmethod
    def detect_correlation_clusters(
        self, correlation_matrix: CorrelationMatrix, threshold: float = 0.7
    ) -> list[list[str]]:
        """Detect correlation clusters among strategies."""

    @abstractmethod
    def get_correlation_stability(self) -> MetricsMap:
        """Get correlation stability metrics."""

class IDiversificationEngine(IActionSignalGuideInterface):
    """Interface for portfolio diversification analysis."""

    @abstractmethod
    def compute_diversification_ratio(
        self, weights: np.ndarray, covariance_matrix: pd.DataFrame
    ) -> float:
        """Compute portfolio diversification ratio."""

    @abstractmethod
    def optimize_diversification(
        self, returns: pd.DataFrame, min_strategies: int = 3
    ) -> PortfolioAllocation:
        """Optimize portfolio for maximum diversification."""

    @abstractmethod
    def get_diversification_metrics(self) -> MetricsMap:
        """Get diversification analysis metrics."""

class IPortfolioRiskManager(IActionSignalGuideInterface):
    """Interface for portfolio-level risk management."""

    @abstractmethod
    def compute_portfolio_risk(
        self,
        weights: np.ndarray,
        covariance_matrix: pd.DataFrame,
        risk_measure: RiskMeasure,
    ) -> float:
        """Compute portfolio risk using specified risk measure."""

    @abstractmethod
    def apply_risk_constraints(
        self, allocation: PortfolioAllocation, risk_limits: dict[str, float]
    ) -> PortfolioAllocation:
        """Apply risk constraints to portfolio allocation."""

    @abstractmethod
    def stress_test_portfolio(
        self, allocation: PortfolioAllocation, scenarios: PayloadRecords
    ) -> PayloadMap:
        """Perform stress testing on portfolio allocation."""

