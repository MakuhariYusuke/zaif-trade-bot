"""
Portfolio Optimization Configuration for Action Signal Guide.

This module provides configuration management for portfolio-level optimization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..interfaces.portfolio_interfaces import AllocationStrategy, RiskMeasure


@dataclass
class StrategyAllocatorConfig:
    """Configuration for strategy allocation."""

    enabled: bool = True
    allocation_strategy: AllocationStrategy = AllocationStrategy.RISK_PARITY
    rebalance_frequency: int = 100  # Rebalance every N signals
    min_allocation_weight: float = 0.01
    max_allocation_weight: float = 0.3
    transaction_cost_rate: float = 0.001
    risk_tolerance: float = 0.5  # 0-1 scale
    target_return: Optional[float] = None
    max_strategies: int = 10
    advanced_features: bool = True  # Enable advanced portfolio features
    min_allocation: Dict[str, float] = field(
        default_factory=dict
    )  # Strategy-specific minimum allocations
    max_allocation: Dict[str, float] = field(
        default_factory=dict
    )  # Strategy-specific maximum allocations

    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        if self.min_allocation_weight < 0 or self.min_allocation_weight > 1:
            return False
        if (
            self.max_allocation_weight < self.min_allocation_weight
            or self.max_allocation_weight > 1
        ):
            return False
        if self.transaction_cost_rate < 0:
            return False
        if self.risk_tolerance < 0 or self.risk_tolerance > 1:
            return False
        if self.rebalance_frequency <= 0:
            return False
        return True


@dataclass
class RiskParityConfig:
    """Configuration for risk parity allocation."""

    enabled: bool = True
    risk_measure: RiskMeasure = RiskMeasure.VARIANCE
    target_risk_contribution: Optional[float] = None  # Equal risk contribution
    max_iterations: int = 100
    tolerance: float = 1e-6
    regularization: float = 1e-8


@dataclass
class CorrelationManagerConfig:
    """Configuration for correlation management."""

    enabled: bool = True
    correlation_window: int = 252  # Trading days
    correlation_method: str = "pearson"  # pearson, spearman, kendall
    clustering_method: str = "hierarchical"  # hierarchical, kmeans
    max_clusters: int = 5
    correlation_threshold: float = 0.7
    update_frequency: int = 50


@dataclass
class DiversificationEngineConfig:
    """Configuration for diversification analysis."""

    enabled: bool = True
    diversification_measure: str = "ratio"  # ratio, index, effective_bets
    min_diversification_ratio: float = 1.5
    max_correlation_for_diversification: float = 0.3
    diversification_target: float = 2.0


@dataclass
class PortfolioRiskManagerConfig:
    """Configuration for portfolio risk management."""

    enabled: bool = True
    primary_risk_measure: RiskMeasure = RiskMeasure.VALUE_AT_RISK
    risk_horizon: int = 1  # Days
    confidence_level: float = 0.95
    max_portfolio_risk: float = 0.02  # 2% max risk
    stress_test_scenarios: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "market_crash", "shock": -0.1},
            {"name": "high_volatility", "volatility_multiplier": 2.0},
            {"name": "correlation_breakdown", "correlation_increase": 0.3},
        ]
    )


@dataclass
class PortfolioConstraints:
    """Portfolio allocation constraints."""

    max_weight_per_strategy: float = 0.25
    min_weight_per_strategy: float = 0.01
    max_total_weight: float = 1.0
    min_total_weight: float = 0.8
    max_strategies: int = 10
    sector_constraints: Dict[str, float] = field(default_factory=dict)
    region_constraints: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioOptimizationConfig:
    """Main configuration for portfolio optimization."""

    enabled: bool = True
    strategy_allocator: StrategyAllocatorConfig = field(
        default_factory=StrategyAllocatorConfig
    )
    risk_parity: RiskParityConfig = field(default_factory=RiskParityConfig)
    correlation_manager: CorrelationManagerConfig = field(
        default_factory=CorrelationManagerConfig
    )
    diversification_engine: DiversificationEngineConfig = field(
        default_factory=DiversificationEngineConfig
    )
    risk_manager: PortfolioRiskManagerConfig = field(
        default_factory=PortfolioRiskManagerConfig
    )
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)

    # Global settings
    optimization_frequency: int = 50  # Optimize every N signals
    backtest_window: int = 252  # Trading days for backtesting
    enable_real_time_optimization: bool = True
    max_optimization_time: float = 5.0  # seconds
    log_level: str = "INFO"

    def __post_init__(self):
        """Initialize default configurations."""
        # Set up default sector constraints if empty
        if not self.constraints.sector_constraints:
            self.constraints.sector_constraints = {
                "trend_following": 0.4,
                "mean_reversion": 0.3,
                "oscillator_based": 0.3,
            }

    def get_allocation_constraints(self) -> Dict[str, Any]:
        """Get allocation constraints as dictionary."""
        return {
            "max_weight": self.constraints.max_weight_per_strategy,
            "min_weight": self.constraints.min_weight_per_strategy,
            "max_total": self.constraints.max_total_weight,
            "min_total": self.constraints.min_total_weight,
            "max_strategies": self.constraints.max_strategies,
            "sector_limits": self.constraints.sector_constraints,
            "region_limits": self.constraints.region_constraints,
        }

    def get_risk_limits(self) -> Dict[str, Any]:
        """Get risk limits as dictionary."""
        return {
            "max_portfolio_risk": self.risk_manager.max_portfolio_risk,
            "risk_measure": self.risk_manager.primary_risk_measure.value,
            "confidence_level": self.risk_manager.confidence_level,
            "risk_horizon": self.risk_manager.risk_horizon,
        }

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check allocation constraints
        if (
            self.constraints.max_weight_per_strategy
            < self.constraints.min_weight_per_strategy
        ):
            issues.append("max_weight_per_strategy must be >= min_weight_per_strategy")

        if self.constraints.max_total_weight < self.constraints.min_total_weight:
            issues.append("max_total_weight must be >= min_total_weight")

        # Check risk tolerance
        if not 0 <= self.strategy_allocator.risk_tolerance <= 1:
            issues.append("risk_tolerance must be between 0 and 1")

        # Check diversification target
        if self.diversification_engine.diversification_target < 1:
            issues.append("diversification_target must be >= 1")

        # Check correlation threshold
        if not -1 <= self.correlation_manager.correlation_threshold <= 1:
            issues.append("correlation_threshold must be between -1 and 1")

        # Check optimization frequency
        if self.optimization_frequency < 10:
            issues.append("optimization_frequency should be at least 10")

        return issues

    def get_optimization_schedule(self) -> Dict[str, Any]:
        """Get optimization schedule configuration."""
        return {
            "rebalance_frequency": self.strategy_allocator.rebalance_frequency,
            "optimization_frequency": self.optimization_frequency,
            "correlation_update_frequency": self.correlation_manager.update_frequency,
            "backtest_window": self.backtest_window,
        }
