"""
Strategy Allocator Implementation for Action Signal Guide.

This module implements portfolio-level strategy allocation and risk management
for optimal performance across multiple trading strategies.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

from ..config.asg_portfolio_config import (
    PortfolioOptimizationConfig,
    RiskParityConfig,
    StrategyAllocatorConfig,
)
from ..components.history_helpers import append_with_compaction
from ..interfaces.portfolio_interfaces import (
    AllocationStrategy,
    IStrategyAllocator,
    PortfolioAllocation,
    RiskMetrics,
    StrategyPerformance,
)
from ..interfaces.common_types import PayloadMap
from ztb.metrics.metrics import max_drawdown

logger = logging.getLogger(__name__)

@dataclass
class AllocationResult:
    """Result of portfolio allocation."""

    allocations: dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_risk: float = 0.0
    sharpe_ratio: float = 0.0
    diversification_ratio: float = 0.0
    optimization_time: float = 0.0
    constraints_satisfied: bool = True
    optimization_status: str = "success"

class BaseStrategyAllocator(IStrategyAllocator):
    """Base implementation of strategy allocator."""

    def __init__(self, config: StrategyAllocatorConfig):
        self.config = config
        self.last_allocation: dict[str, float] | None = None
        self.performance_history: list[StrategyPerformance] = []
        self.allocation_history: list[PortfolioAllocation] = []

    def allocate_strategies(
        self,
        strategy_performance: dict[str, StrategyPerformance],
        market_conditions: PayloadMap,
    ) -> PortfolioAllocation:
        """Allocate capital across trading strategies."""
        start_time = time.time()

        try:
            # Validate inputs
            if not strategy_performance:
                return self._create_empty_allocation("No strategy performance data")

            # Prepare data
            returns, risks, correlations = self._prepare_strategy_data(
                strategy_performance
            )

            # Apply allocation strategy
            if self.config.allocation_strategy == AllocationStrategy.EQUAL_WEIGHT:
                allocations = self._equal_weight_allocation(strategy_performance)
            elif self.config.allocation_strategy == AllocationStrategy.RISK_PARITY:
                allocations = self._risk_parity_allocation(returns, risks, correlations)
            elif self.config.allocation_strategy == AllocationStrategy.MAXIMUM_SHARPE:
                allocations = self._maximum_sharpe_allocation(
                    returns, risks, correlations
                )
            elif self.config.allocation_strategy == AllocationStrategy.MINIMUM_VARIANCE:
                allocations = self._minimum_variance_allocation(
                    returns, risks, correlations
                )
            else:
                allocations = self._equal_weight_allocation(strategy_performance)

            # Apply constraints
            allocations = self._apply_constraints(allocations, strategy_performance)

            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                allocations, returns, risks, correlations
            )

            # Create result
            result = AllocationResult(
                allocations=allocations,
                expected_return=portfolio_metrics.get("expected_return", 0.0),
                expected_risk=portfolio_metrics.get("expected_risk", 0.0),
                sharpe_ratio=portfolio_metrics.get("sharpe_ratio", 0.0),
                diversification_ratio=portfolio_metrics.get(
                    "diversification_ratio", 0.0
                ),
                optimization_time=time.time() - start_time,
                constraints_satisfied=self._check_constraints(allocations),
                optimization_status="success",
            )

            self.last_allocation = allocations

            allocation = PortfolioAllocation(
                allocations=allocations,
                expected_return=result.expected_return,
                expected_volatility=result.expected_risk,
                sharpe_ratio=result.sharpe_ratio,
                diversification_ratio=result.diversification_ratio,
                timestamp=time.time(),
            )
            append_with_compaction(
                self.allocation_history,
                allocation,
                high_water=1000,
                retain=500,
            )
            return allocation

        except Exception as e:
            logger.error(f"Strategy allocation failed: {e}")
            return self._create_empty_allocation(f"Allocation failed: {str(e)}")

    def rebalance_portfolio(
        self,
        current_performance: dict[str, StrategyPerformance],
        market_conditions: PayloadMap,
    ) -> PortfolioAllocation:
        """Rebalance portfolio based on current conditions."""
        # Check if rebalancing is needed
        if not self._should_rebalance(current_performance):
            # Return current allocation if no rebalance needed
            if self.last_allocation:
                return PortfolioAllocation(
                    allocations=self.last_allocation,
                    expected_return=0.0,  # Would need to recalculate
                    expected_volatility=0.0,
                    sharpe_ratio=0.0,
                    diversification_ratio=0.0,
                    timestamp=time.time(),
                )

        # Perform full reallocation
        return self.allocate_strategies(current_performance, market_conditions)

    def calculate_risk_metrics(
        self,
        allocations: dict[str, float],
        strategy_performance: dict[str, StrategyPerformance],
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics for the portfolio."""
        try:
            returns, risks, correlations = self._prepare_strategy_data(
                strategy_performance
            )

            # Portfolio return and risk
            port_return = sum(
                allocations.get(s, 0) * returns.get(s, 0) for s in allocations
            )
            port_risk = self._calculate_portfolio_risk(allocations, risks, correlations)

            # Risk contributions
            risk_contributions = self._calculate_risk_contributions(
                allocations, risks, correlations
            )

            # Diversification metrics
            diversification_ratio = self._calculate_diversification_ratio(
                allocations, correlations
            )

            # Sharpe ratio
            sharpe_ratio = port_return / port_risk if port_risk > 0 else 0

            # Value at Risk (simplified)
            var_95 = -2.326 * port_risk  # Assuming normal distribution

            # Maximum drawdown (simplified)
            max_drawdown = self._estimate_max_drawdown(strategy_performance)

            return RiskMetrics(
                portfolio_volatility=port_risk,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sharpe_ratio * 0.8,  # Approximation
                maximum_drawdown=max_drawdown,
                value_at_risk=var_95,
                expected_shortfall=var_95 * 1.2,  # Approximation
                diversification_ratio=diversification_ratio,
                risk_contributions=risk_contributions,
                concentration_metrics=self._calculate_concentration_metrics(
                    allocations
                ),
            )

        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return RiskMetrics(
                portfolio_volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                maximum_drawdown=0.0,
                value_at_risk=0.0,
                expected_shortfall=0.0,
                diversification_ratio=0.0,
                risk_contributions={},
                concentration_metrics={},
            )

    def get_allocation_history(self) -> list[PortfolioAllocation]:
        """Get historical allocation data."""
        return list(self.allocation_history)

    def _prepare_strategy_data(
        self, strategy_performance: dict[str, StrategyPerformance]
    ) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
        """Prepare strategy data for optimization."""
        returns = {}
        risks = {}
        correlations = pd.DataFrame()

        for strategy_name, perf in strategy_performance.items():
            returns[strategy_name] = perf.expected_return
            risks[strategy_name] = perf.volatility

        # Create correlation matrix
        strategy_names = list(strategy_performance.keys())
        n = len(strategy_names)
        corr_matrix = np.eye(n)  # Default to identity matrix

        # Use provided correlations if available
        for i, s1 in enumerate(strategy_names):
            for j, s2 in enumerate(strategy_names):
                if (
                    hasattr(strategy_performance[s1], "correlations")
                    and s2 in strategy_performance[s1].correlations
                ):
                    corr_matrix[i, j] = strategy_performance[s1].correlations[s2]

        correlations = pd.DataFrame(
            corr_matrix, index=strategy_names, columns=strategy_names
        )

        return returns, risks, correlations

    def _equal_weight_allocation(
        self, strategy_performance: dict[str, StrategyPerformance]
    ) -> dict[str, float]:
        """Equal weight allocation across all strategies."""
        return self._build_equal_weight_allocations(list(strategy_performance.keys()))

    def _risk_parity_allocation(
        self,
        returns: dict[str, float],
        risks: dict[str, float],
        correlations: pd.DataFrame,
    ) -> dict[str, float]:
        """Risk parity allocation."""
        try:
            strategy_names = list(returns.keys())
            n = len(strategy_names)

            if n == 0:
                return {}

            # Simplified risk parity - equal risk contribution
            cov_matrix = self._create_covariance_matrix(risks, correlations)

            # Risk parity optimization
            def risk_parity_objective(weights):
                port_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                if port_risk <= 0:
                    return float("inf")
                risk_contribs = weights * (cov_matrix @ weights) / port_risk
                return np.var(risk_contribs)  # Minimize variance of risk contributions

            optimized_weights = self._optimize_weights(risk_parity_objective, n)
            if optimized_weights is None:
                logger.warning("Risk parity optimization failed, using equal weight")
                return self._equal_weight_allocation_from_dict(returns)
            return dict(zip(strategy_names, optimized_weights))

        except Exception as e:
            logger.error(f"Risk parity allocation failed: {e}")
            return self._equal_weight_allocation_from_dict(returns)

    def _maximum_sharpe_allocation(
        self,
        returns: dict[str, float],
        risks: dict[str, float],
        correlations: pd.DataFrame,
    ) -> dict[str, float]:
        """Maximum Sharpe ratio allocation."""
        try:
            strategy_names = list(returns.keys())
            n = len(strategy_names)

            if n == 0:
                return {}

            returns_array = np.array([returns[s] for s in strategy_names])
            cov_matrix = self._create_covariance_matrix(risks, correlations)

            # Maximize Sharpe ratio (return / risk)
            def sharpe_objective(weights):
                port_return = weights @ returns_array
                port_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                if port_risk <= 0:
                    return float("inf")
                return -port_return / port_risk  # Negative for minimization

            optimized_weights = self._optimize_weights(sharpe_objective, n)
            if optimized_weights is None:
                logger.warning("Maximum Sharpe optimization failed, using equal weight")
                return self._equal_weight_allocation_from_dict(returns)
            return dict(zip(strategy_names, optimized_weights))

        except Exception as e:
            logger.error(f"Maximum Sharpe allocation failed: {e}")
            return self._equal_weight_allocation_from_dict(returns)

    def _minimum_variance_allocation(
        self,
        returns: dict[str, float],
        risks: dict[str, float],
        correlations: pd.DataFrame,
    ) -> dict[str, float]:
        """Minimum variance allocation."""
        try:
            strategy_names = list(returns.keys())
            n = len(strategy_names)

            if n == 0:
                return {}

            cov_matrix = self._create_covariance_matrix(risks, correlations)

            # Minimize portfolio variance
            def variance_objective(weights):
                return weights.T @ cov_matrix @ weights

            optimized_weights = self._optimize_weights(variance_objective, n)
            if optimized_weights is None:
                logger.warning(
                    "Minimum variance optimization failed, using equal weight"
                )
                return self._equal_weight_allocation_from_dict(returns)
            return dict(zip(strategy_names, optimized_weights))

        except Exception as e:
            logger.error(f"Minimum variance allocation failed: {e}")
            return self._equal_weight_allocation_from_dict(returns)

    @staticmethod
    def _optimize_weights(
        objective: Callable[[np.ndarray], float],
        n_assets: int,
    ) -> np.ndarray | None:
        """Solve bounded sum-to-1 optimization for portfolio weights."""
        if n_assets <= 0:
            return None
        constraints = [{"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)}]
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        x0 = np.full(n_assets, 1.0 / n_assets)
        result = minimize(objective, x0, bounds=bounds, constraints=constraints)
        if not result.success:
            return None
        return np.asarray(result.x, dtype=float)

    def _equal_weight_allocation_from_dict(
        self, data: dict[str, float]
    ) -> dict[str, float]:
        """Equal weight allocation from dictionary."""
        return self._build_equal_weight_allocations(list(data.keys()))

    @staticmethod
    def _build_equal_weight_allocations(strategy_names: list[str]) -> dict[str, float]:
        """Create equal-weight allocation map for strategy names."""
        n = len(strategy_names)
        if n == 0:
            return {}
        weight = 1.0 / n
        return dict.fromkeys(strategy_names, weight)

    def _create_covariance_matrix(
        self, risks: dict[str, float], correlations: pd.DataFrame
    ) -> np.ndarray:
        """Create covariance matrix from volatilities and correlations."""
        strategy_names = list(risks.keys())
        n = len(strategy_names)

        cov_matrix = np.zeros((n, n))
        for i, s1 in enumerate(strategy_names):
            for j, s2 in enumerate(strategy_names):
                if i == j:
                    cov_matrix[i, j] = risks[s1] ** 2
                else:
                    cov_matrix[i, j] = risks[s1] * risks[s2] * correlations.loc[s1, s2]

        return cov_matrix

    def _apply_constraints(
        self,
        allocations: dict[str, float],
        strategy_performance: dict[str, StrategyPerformance],
    ) -> dict[str, float]:
        """Apply allocation constraints."""
        constrained_allocations = allocations.copy()

        # Minimum and maximum allocation constraints
        for strategy, weight in constrained_allocations.items():
            if strategy in self.config.min_allocation:
                constrained_allocations[strategy] = max(
                    weight, self.config.min_allocation[strategy]
                )
            if strategy in self.config.max_allocation:
                constrained_allocations[strategy] = min(
                    weight, self.config.max_allocation[strategy]
                )

        # Normalize to sum to 1
        total_weight = sum(constrained_allocations.values())
        if total_weight > 0:
            constrained_allocations = {
                k: v / total_weight for k, v in constrained_allocations.items()
            }

        return constrained_allocations

    def _check_constraints(self, allocations: dict[str, float]) -> bool:
        """Check if allocations satisfy constraints."""
        total_weight = sum(allocations.values())
        if abs(total_weight - 1.0) > 1e-6:
            return False

        for strategy, weight in allocations.items():
            if weight < 0 or weight > 1:
                return False

        return True

    def _calculate_portfolio_metrics(
        self,
        allocations: dict[str, float],
        returns: dict[str, float],
        risks: dict[str, float],
        correlations: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate portfolio-level metrics."""
        strategy_names = list(allocations.keys())
        weights = np.array([allocations[s] for s in strategy_names])
        returns_array = np.array([returns.get(s, 0) for s in strategy_names])

        # Expected return
        expected_return = weights @ returns_array

        # Expected risk
        cov_matrix = self._create_covariance_matrix(risks, correlations)
        expected_risk = np.sqrt(weights.T @ cov_matrix @ weights)

        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0

        # Diversification ratio
        weighted_vol = weights @ np.array([risks.get(s, 0) for s in strategy_names])
        diversification_ratio = weighted_vol / expected_risk if expected_risk > 0 else 0

        return {
            "expected_return": expected_return,
            "expected_risk": expected_risk,
            "sharpe_ratio": sharpe_ratio,
            "diversification_ratio": diversification_ratio,
        }

    def _calculate_portfolio_risk(
        self,
        allocations: dict[str, float],
        risks: dict[str, float],
        correlations: pd.DataFrame,
    ) -> float:
        """Calculate portfolio volatility."""
        cov_matrix = self._create_covariance_matrix(risks, correlations)
        weights = np.array([allocations.get(s, 0) for s in risks.keys()])
        return np.sqrt(weights.T @ cov_matrix @ weights)

    def _calculate_risk_contributions(
        self,
        allocations: dict[str, float],
        risks: dict[str, float],
        correlations: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate risk contribution of each strategy."""
        cov_matrix = self._create_covariance_matrix(risks, correlations)
        weights = np.array([allocations.get(s, 0) for s in risks.keys()])
        port_risk = np.sqrt(weights.T @ cov_matrix @ weights)

        if port_risk == 0:
            return dict.fromkeys(allocations.keys(), 0.0)

        marginal_contribs = cov_matrix @ weights
        risk_contribs = weights * marginal_contribs / port_risk

        return dict(zip(allocations.keys(), risk_contribs))

    def _calculate_diversification_ratio(
        self, allocations: dict[str, float], correlations: pd.DataFrame
    ) -> float:
        """Calculate diversification ratio."""
        weights = np.array(list(allocations.values()))
        corr_matrix = correlations.loc[allocations.keys(), allocations.keys()].values

        # Weighted average correlation
        avg_corr = weights @ corr_matrix @ weights

        # Diversification ratio = sqrt(weights @ corr_matrix @ weights) / sum(weights)
        # Simplified version
        return 1.0 / np.sqrt(avg_corr) if avg_corr > 0 else 1.0

    def _calculate_concentration_metrics(
        self, allocations: dict[str, float]
    ) -> dict[str, float]:
        """Calculate concentration metrics."""
        weights = np.array(list(allocations.values()))

        # Herfindahl-Hirschman Index
        hhi = np.sum(weights**2)

        # Gini coefficient approximation
        sorted_weights = np.sort(weights)
        n = len(sorted_weights)
        gini = (
            2
            * np.sum(np.arange(1, n + 1) * sorted_weights)
            / (n * np.sum(sorted_weights))
        ) - (n + 1) / n

        # Largest allocation
        max_allocation = np.max(weights)

        return {
            "hhi": hhi,
            "gini_coefficient": gini,
            "largest_allocation": max_allocation,
            "allocation_count": len(allocations),
        }

    def _estimate_max_drawdown(
        self, strategy_performance: dict[str, StrategyPerformance]
    ) -> float:
        """Estimate maximum drawdown from strategy performance."""
        # Try to use actual portfolio values if available
        all_values = []
        for perf in strategy_performance.values():
            if hasattr(perf, 'portfolio_values') and perf.portfolio_values:
                all_values.extend(perf.portfolio_values)

        if all_values:
            return abs(max_drawdown(all_values))

        # Fallback: Simplified estimation based on volatilities
        max_vol = max(
            (perf.volatility for perf in strategy_performance.values()), default=0
        )
        return max_vol * 2.5  # Rough approximation

    def _should_rebalance(
        self, current_performance: dict[str, StrategyPerformance]
    ) -> bool:
        """Determine if portfolio should be rebalanced."""
        if not self.last_allocation:
            return True

        # Check if significant performance changes occurred
        for strategy, perf in current_performance.items():
            if strategy not in self.last_allocation:
                return True

        # Check time-based rebalancing
        # This would need timestamp tracking in practice
        return True  # Simplified - always rebalance for now

    def _create_empty_allocation(self, reason: str) -> PortfolioAllocation:
        """Create empty allocation result."""
        logger.warning(f"Returning empty allocation: {reason}")
        return PortfolioAllocation(
            allocations={},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            diversification_ratio=0.0,
            timestamp=time.time(),
        )

class AdvancedStrategyAllocator(BaseStrategyAllocator):
    """Advanced strategy allocator with additional optimization features."""

    def __init__(self, config: StrategyAllocatorConfig):
        super().__init__(config)
        self.risk_parity_config = RiskParityConfig()  # Would be passed in config

    def optimize_allocation(
        self,
        strategy_performance: dict[str, StrategyPerformance],
        risk_tolerance: float,
        constraints: PayloadMap,
    ) -> PortfolioAllocation:
        """Advanced portfolio optimization with risk parity and constraints."""
        try:
            if not strategy_performance:
                return self._create_empty_allocation("No strategy performance data")

            # Extract returns and create covariance matrix
            returns_data = self._extract_returns_data(strategy_performance)
            if returns_data.empty:
                return self._create_empty_allocation("No returns data available")

            # Apply risk parity optimization
            weights = self._compute_risk_parity_weights(returns_data, risk_tolerance)

            # Apply constraints
            constrained_weights = self._apply_allocation_constraints(
                weights, constraints
            )

            # Calculate portfolio metrics
            (
                portfolio_return,
                portfolio_volatility,
                sharpe_ratio,
            ) = self._calculate_portfolio_metrics(constrained_weights, returns_data)

            diversification_ratio = self._calculate_diversification_ratio(
                constrained_weights, returns_data
            )

            return PortfolioAllocation(
                allocations=constrained_weights,
                expected_return=portfolio_return,
                expected_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                diversification_ratio=diversification_ratio,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(f"Advanced allocation optimization failed: {e}")
            return self._create_empty_allocation(f"Optimization failed: {str(e)}")

    def _extract_returns_data(
        self, strategy_performance: dict[str, StrategyPerformance]
    ) -> pd.DataFrame:
        """Extract returns data from strategy performance."""
        # This is a simplified implementation
        # In practice, you'd have historical returns data
        strategy_names = list(strategy_performance.keys())
        n_periods = 252  # One year of daily data

        # Generate synthetic returns based on performance metrics
        returns_data = {}
        for strategy in strategy_names:
            perf = strategy_performance[strategy]
            # Use expected return and volatility to generate synthetic returns
            mean_return = perf.expected_return / 252  # Daily return
            volatility = perf.volatility / np.sqrt(252)  # Daily volatility

            # Generate random returns (in practice, use real historical data)
            np.random.seed(42)
            returns = np.random.normal(mean_return, volatility, n_periods)
            returns_data[strategy] = returns

        return pd.DataFrame(returns_data)

    def _compute_risk_parity_weights(
        self, returns_data: pd.DataFrame, risk_tolerance: float
    ) -> dict[str, float]:
        """Compute risk parity weights."""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_data.cov()

            # Risk parity optimization (simplified)
            n_assets = len(returns_data.columns)
            initial_weights = np.ones(n_assets) / n_assets

            # Use scipy optimizer for risk parity
            from scipy.optimize import minimize

            def risk_parity_objective(weights):
                # Risk contributions should be equal
                portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
                risk_contributions = weights * (cov_matrix @ weights) / portfolio_vol
                target_contribution = portfolio_vol / n_assets
                return np.sum((risk_contributions - target_contribution) ** 2)

            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            bounds = [(0.01, 0.3) for _ in range(n_assets)]  # Weight bounds

            result = minimize(
                risk_parity_objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                weights = result.x
            else:
                # Fallback to equal weights
                weights = initial_weights

            return dict(zip(returns_data.columns, weights))

        except Exception as e:
            logger.warning(f"Risk parity optimization failed: {e}")
            # Fallback to equal weights
            n_assets = len(returns_data.columns)
            equal_weight = 1.0 / n_assets
            return dict.fromkeys(returns_data.columns, equal_weight)

    def _apply_allocation_constraints(
        self, weights: dict[str, float], constraints: PayloadMap
    ) -> dict[str, float]:
        """Apply allocation constraints to weights."""
        constrained_weights = weights.copy()

        # Apply minimum and maximum weight constraints
        min_weight = self._coerce_float(
            constraints.get("min_weight"), self.config.min_allocation_weight
        )
        max_weight = self._coerce_float(
            constraints.get("max_weight"), self.config.max_allocation_weight
        )

        for strategy, weight in constrained_weights.items():
            constrained_weights[strategy] = np.clip(weight, min_weight, max_weight)

        # Renormalize to ensure sum equals 1
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {
                k: v / total_weight for k, v in constrained_weights.items()
            }

        return constrained_weights

    def _calculate_portfolio_metrics(
        self, weights: dict[str, float], returns_data: pd.DataFrame
    ) -> tuple:
        """Calculate portfolio-level metrics."""
        try:
            weight_array = np.array(list(weights.values()))
            strategy_names = list(weights.keys())

            # Portfolio return (annualized)
            individual_returns = returns_data[strategy_names].mean() * 252
            portfolio_return = weight_array @ individual_returns.values

            # Portfolio volatility (annualized)
            cov_matrix = returns_data[strategy_names].cov() * 252
            portfolio_volatility = np.sqrt(weight_array @ cov_matrix @ weight_array)

            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

            return portfolio_return, portfolio_volatility, sharpe_ratio

        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return 0.0, 0.0, 0.0

    def _calculate_diversification_ratio(
        self, weights: dict[str, float], returns_data: pd.DataFrame
    ) -> float:
        """Calculate portfolio diversification ratio."""
        try:
            weight_array = np.array(list(weights.values()))
            strategy_names = list(weights.keys())

            # Individual volatilities
            individual_vols = returns_data[strategy_names].std() * np.sqrt(252)

            # Portfolio volatility
            cov_matrix = returns_data[strategy_names].cov() * 252
            portfolio_vol = np.sqrt(weight_array @ cov_matrix @ weight_array)

            # Diversification ratio
            weighted_avg_vol = weight_array @ individual_vols.values

            if weighted_avg_vol > 0:
                return portfolio_vol / weighted_avg_vol
            else:
                return 1.0

        except Exception as e:
            logger.warning(f"Diversification ratio calculation failed: {e}")
            return 1.0

    def allocate_strategies(
        self,
        strategy_performance: dict[str, StrategyPerformance],
        market_conditions: PayloadMap,
    ) -> PortfolioAllocation:
        """Advanced allocation using optimization methods."""
        # Use risk tolerance from config or market conditions
        risk_tolerance = self._coerce_float(
            market_conditions.get("risk_tolerance"), self.config.risk_tolerance
        )

        # Create constraints from config
        constraints = {
            "min_weight": self.config.min_allocation_weight,
            "max_weight": self.config.max_allocation_weight,
            "max_strategies": self.config.max_strategies,
        }

        # Use the advanced optimization method
        return self.optimize_allocation(
            strategy_performance, risk_tolerance, constraints
        )

    def _adjust_for_market_conditions(
        self,
        strategy_performance: dict[str, StrategyPerformance],
        market_conditions: PayloadMap,
    ) -> dict[str, StrategyPerformance]:
        """Adjust strategy performance based on market conditions."""
        adjusted = {}

        market_regime = str(market_conditions.get("regime", "neutral"))
        volatility = self._coerce_float(market_conditions.get("volatility"), 0.2)

        for strategy_name, perf in strategy_performance.items():
            adjusted_perf = StrategyPerformance(
                strategy_name=perf.strategy_name,
                expected_return=self._adjust_return_for_regime(
                    perf.expected_return, market_regime
                ),
                volatility=self._adjust_volatility_for_conditions(
                    perf.volatility, volatility
                ),
                sharpe_ratio=perf.sharpe_ratio,
                max_drawdown=perf.max_drawdown,
                win_rate=perf.win_rate,
                profit_factor=perf.profit_factor,
                timestamp=perf.timestamp,
                correlations=perf.correlations.copy() if perf.correlations else None,
            )
            adjusted[strategy_name] = adjusted_perf

        return adjusted

    def _adjust_return_for_regime(self, base_return: float, regime: str) -> float:
        """Adjust expected return based on market regime."""
        regime_multipliers = {"bull": 1.2, "bear": 0.8, "sideways": 0.9, "neutral": 1.0}
        multiplier = regime_multipliers.get(regime, 1.0)
        return base_return * multiplier

    def _adjust_volatility_for_conditions(
        self, base_volatility: float, market_volatility: float
    ) -> float:
        """Adjust volatility based on market conditions."""
        # Increase volatility in high market volatility periods
        adjustment_factor = 1 + (market_volatility - 0.2) * 0.5
        return base_volatility * adjustment_factor

    @staticmethod
    def _coerce_float(value: object, default: float) -> float:
        """Coerce unknown payload values into float safely."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

def create_strategy_allocator(
    config: PortfolioOptimizationConfig,
) -> IStrategyAllocator:
    """Factory function to create strategy allocator."""
    if config.strategy_allocator.advanced_features:
        return AdvancedStrategyAllocator(config.strategy_allocator)
    else:
        return BaseStrategyAllocator(config.strategy_allocator)
