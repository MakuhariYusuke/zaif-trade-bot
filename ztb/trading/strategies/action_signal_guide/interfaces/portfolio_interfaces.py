"""
Portfolio Optimization Interfaces for Action Signal Guide.

This module defines interfaces for portfolio-level strategy optimization and risk management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


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
    correlations: Optional[Dict[str, float]] = None


# Using RiskMetrics from monitoring types for compatibility


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""

    allocations: Dict[str, float]
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


class IStrategyAllocator(ABC):
    """Interface for strategy allocation optimization."""

    @abstractmethod
    def optimize_allocation(
        self,
        strategy_performance: Dict[str, StrategyPerformance],
        risk_tolerance: float,
        constraints: Dict[str, Any],
    ) -> PortfolioAllocation:
        """
        Optimize portfolio allocation across strategies.

        Args:
            strategy_performance: Performance data for each strategy
            risk_tolerance: Risk tolerance level (0-1)
            constraints: Allocation constraints

        Returns:
            Optimized portfolio allocation
        """
        pass

    @abstractmethod
    def rebalance_portfolio(
        self,
        current_allocation: PortfolioAllocation,
        new_performance_data: Dict[str, StrategyPerformance],
        transaction_costs: float,
    ) -> PortfolioAllocation:
        """
        Rebalance portfolio based on new performance data.

        Args:
            current_allocation: Current portfolio allocation
            new_performance_data: Updated strategy performance
            transaction_costs: Transaction cost rate

        Returns:
            Rebalanced portfolio allocation
        """
        pass

    @abstractmethod
    def get_allocation_history(self) -> List[PortfolioAllocation]:
        """Get historical allocation decisions."""
        pass


class IRiskParityAllocator(ABC):
    """Interface for risk parity allocation strategy."""

    @abstractmethod
    def compute_risk_contributions(
        self, returns: pd.DataFrame, weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute risk contributions for each asset.

        Args:
            returns: Historical returns matrix
            weights: Portfolio weights

        Returns:
            Risk contributions array
        """
        pass

    @abstractmethod
    def optimize_risk_parity(
        self, returns: pd.DataFrame, target_risk: float = None
    ) -> PortfolioAllocation:
        """
        Optimize portfolio using risk parity approach.

        Args:
            returns: Historical returns matrix
            target_risk: Target risk level (optional)

        Returns:
            Risk parity optimized allocation
        """
        pass

    @abstractmethod
    def get_risk_parity_metrics(self) -> Dict[str, Any]:
        """Get risk parity specific metrics."""
        pass


class ICorrelationManager(ABC):
    """Interface for managing strategy correlations."""

    @abstractmethod
    def update_correlations(
        self, strategy_returns: Dict[str, pd.Series]
    ) -> CorrelationMatrix:
        """
        Update correlation matrix with new return data.

        Args:
            strategy_returns: Dictionary of strategy return series

        Returns:
            Updated correlation matrix
        """
        pass

    @abstractmethod
    def detect_correlation_clusters(
        self, correlation_matrix: CorrelationMatrix, threshold: float = 0.7
    ) -> List[List[str]]:
        """
        Detect correlation clusters among strategies.

        Args:
            correlation_matrix: Current correlation matrix
            threshold: Correlation threshold for clustering

        Returns:
            List of strategy clusters
        """
        pass

    @abstractmethod
    def get_correlation_stability(self) -> Dict[str, Any]:
        """Get correlation stability metrics."""
        pass


class IDiversificationEngine(ABC):
    """Interface for portfolio diversification analysis."""

    @abstractmethod
    def compute_diversification_ratio(
        self, weights: np.ndarray, covariance_matrix: pd.DataFrame
    ) -> float:
        """
        Compute portfolio diversification ratio.

        Args:
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix

        Returns:
            Diversification ratio
        """
        pass

    @abstractmethod
    def optimize_diversification(
        self, returns: pd.DataFrame, min_strategies: int = 3
    ) -> PortfolioAllocation:
        """
        Optimize portfolio for maximum diversification.

        Args:
            returns: Historical returns matrix
            min_strategies: Minimum number of strategies

        Returns:
            Diversification optimized allocation
        """
        pass

    @abstractmethod
    def get_diversification_metrics(self) -> Dict[str, Any]:
        """Get diversification analysis metrics."""
        pass


class IPortfolioRiskManager(ABC):
    """Interface for portfolio-level risk management."""

    @abstractmethod
    def compute_portfolio_risk(
        self,
        weights: np.ndarray,
        covariance_matrix: pd.DataFrame,
        risk_measure: RiskMeasure,
    ) -> float:
        """
        Compute portfolio risk using specified risk measure.

        Args:
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix
            risk_measure: Risk measurement method

        Returns:
            Portfolio risk value
        """
        pass

    @abstractmethod
    def apply_risk_constraints(
        self, allocation: PortfolioAllocation, risk_limits: Dict[str, float]
    ) -> PortfolioAllocation:
        """
        Apply risk constraints to portfolio allocation.

        Args:
            allocation: Proposed portfolio allocation
            risk_limits: Risk limit constraints

        Returns:
            Risk-constrained allocation
        """
        pass

    @abstractmethod
    def stress_test_portfolio(
        self, allocation: PortfolioAllocation, scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio allocation.

        Args:
            allocation: Portfolio allocation to test
            scenarios: Stress test scenarios

        Returns:
            Stress test results
        """
        pass
