#!/usr/bin/env python3
"""
Unified Backtest Framework

A comprehensive backtesting framework that supports multiple trading strategies
including SAC models, Action Signal Guide, and hybrid approaches. Designed to
leverage SAC learning outcomes for enhanced analysis and strategy evaluation.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union

import numpy as np
import pandas as pd

from ..adapters import StrategyAdapter
from ..metrics import BacktestMetrics, MetricsCalculator
from ..report import ReportGenerator
from ...utils.logging_utils import get_logger
from .signal_performance import BacktestSignalPerformanceAnalyzer

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    max_position_size: float = 1.0  # Max position as fraction of capital
    enable_shorting: bool = True
    data_start_date: Optional[str] = None
    data_end_date: Optional[str] = None
    warmup_periods: int = 100


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    config: Dict[str, Union[str, int, float, bool]]
    performance_metrics: BacktestMetrics
    trade_history: List[Dict[str, Union[str, int, float]]]
    portfolio_values: List[float]
    execution_time: float
    metadata: Dict[str, Union[str, int, float, bool]]


class TradingStrategy(Protocol):
    """
    Protocol for trading strategies in the unified backtest framework.

    All trading strategies should implement this protocol.
    """

    @property
    def name(self) -> str:
        """Strategy name."""
        ...

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_position: int
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Generate trading signal.

        Args:
            data: Market data with OHLCV and features
            current_position: Current position (-1, 0, 1 for short, flat, long)

        Returns:
            Signal dict with 'action' and optional parameters
        """
        ...

    def update_hyperparameters(self, hyperparameters: Dict[str, float]) -> None:
        """
        Update strategy hyperparameters.

        Args:
            hyperparameters: Dictionary of hyperparameter names and values
        """
        ...


class UnifiedBacktester:
    """
    Unified backtesting framework supporting multiple strategies.

    Features:
    - Multiple strategy support (SAC, Action Signal Guide, Hybrid)
    - SAC learning outcome integration
    - Comprehensive performance analysis
    - Parallel execution capabilities
    - Automated reporting
    """

    def __init__(self, config_manager: Optional['ConfigManager'] = None):
        """
        Initialize the unified backtester.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.results_manager = ResultsManager()
        self.data_manager = DataManager()
        self.logger = logger

        # Strategy registry
        self.strategies: Dict[str, Union[TradingStrategy, StrategyAdapter]] = {}
        self.active_strategy: Optional[Union[TradingStrategy, StrategyAdapter]] = None

        # Signal performance analyzer
        self.signal_performance_analyzer = BacktestSignalPerformanceAnalyzer()

    def register_strategy(
        self,
        name: str,
        strategy: Union[TradingStrategy, StrategyAdapter]
    ) -> None:
        """
        Register a trading strategy.

        Args:
            name: Strategy name
            strategy: Strategy instance
        """
        self.strategies[name] = strategy
        self.logger.info(f"Registered strategy: {name}")

    def set_active_strategy(self, name: str) -> None:
        """
        Set the active strategy for backtesting.

        Args:
            name: Strategy name

        Raises:
            ValueError: If strategy is not registered
        """
        if name not in self.strategies:
            available = list(self.strategies.keys())
            raise ValueError(f"Strategy '{name}' not found. Available: {available}")

        self.active_strategy = self.strategies[name]
        self.logger.info(f"Set active strategy: {name}")

    def run_backtest(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
        save_results: bool = True,
        **kwargs
    ) -> BacktestResult:
        """
        Run a backtest with the specified strategy.

        Args:
            strategy_name: Name of the strategy to use
            data: OHLCV data for backtesting
            config: Backtest configuration
            save_results: Whether to save results automatically
            **kwargs: Additional strategy-specific parameters

        Returns:
            BacktestResult: Complete backtest results
        """
        start_time = time.time()

        # Set active strategy
        self.set_active_strategy(strategy_name)
        strategy = self.active_strategy

        # Default configuration
        if config is None:
            config = BacktestConfig()

        self.logger.info(f"Starting backtest with strategy: {strategy_name}")
        self.logger.info(f"Data shape: {data.shape}")
        self.logger.info(f"Initial capital: ${config.initial_capital:,.2f}")

        try:
            # Run backtest using existing BacktestEngine
            result = self._execute_backtest(strategy, data, config)

            # Calculate execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time

            # Add metadata
            result.metadata.update({
                "data_points": len(data),
                "date_range": f"{data.index[0]} to {data.index[-1]}",
                "strategy_params": kwargs,
                "backtest_config": config.__dict__,
            })

            self.logger.info(f"Backtest completed in {execution_time:.2f} seconds")
            # Save results if requested
            if save_results:
                self.results_manager.save_result(result)

            return result

        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise

    def _execute_backtest(
        self,
        strategy: Union[TradingStrategy, StrategyAdapter],
        data: pd.DataFrame,
        config: BacktestConfig
    ) -> BacktestResult:
        """
        Execute the actual backtest logic using existing BacktestEngine.

        Args:
            strategy: Trading strategy instance
            data: Market data
            config: Backtest configuration

        Returns:
            BacktestResult: Backtest results
        """
        # Import existing BacktestEngine
        from ..runner import BacktestEngine

        # Create backtest engine with configuration
        engine = BacktestEngine(
            initial_capital=config.initial_capital,
            commission_bps=config.commission * 10000,  # Convert to bps
            slippage_bps=config.slippage * 10000,  # Convert to bps
            max_position_size=config.max_position_size,
            signal_performance_analyzer=self.signal_performance_analyzer,
        )

        # Convert strategy to adapter if needed
        if hasattr(strategy, 'generate_signal'):
            adapter = strategy
        else:
            # Create adapter wrapper for protocol-based strategies
            adapter = StrategyAdapterWrapper(strategy)

        # Run backtest using existing engine
        try:
            equity_series, orders_df, adaptation_summary, signal_performance_summary = engine.run_backtest(adapter, data)

            # Convert results to unified format
            return self._convert_to_unified_result(
                equity_series, orders_df, adaptation_summary, signal_performance_summary, strategy.name, config
            )

        except Exception as e:
            self.logger.error(f"Backtest execution failed: {e}")
            raise

    def _convert_to_unified_result(
        self,
        equity_series: pd.Series,
        orders_df: pd.DataFrame,
        adaptation_summary: Optional[Dict[str, Any]],
        signal_performance_summary: Optional[Dict[str, Any]],
        strategy_name: str,
        config: BacktestConfig
    ) -> BacktestResult:
        """
        Convert BacktestEngine results to unified format.

        Args:
            equity_series: Equity curve from backtest
            orders_df: Trade orders DataFrame
            adaptation_summary: Adaptation system results
            signal_performance_summary: Signal performance analysis results
            strategy_name: Name of the strategy
            config: Backtest configuration

        Returns:
            BacktestResult in unified format
        """
        # Calculate metrics from equity series and orders
        from ..metrics import MetricsCalculator
        metrics_calculator = MetricsCalculator()
        metrics = metrics_calculator.calculate_metrics(equity_series, orders_df)

        # Convert orders DataFrame to list of dicts
        trade_history = orders_df.to_dict('records') if not orders_df.empty else []

        # Convert equity series to list of values
        portfolio_values = equity_series.tolist()

        # Prepare metadata
        metadata = {}
        if adaptation_summary:
            metadata["adaptation"] = adaptation_summary
        if signal_performance_summary:
            metadata["signal_performance"] = signal_performance_summary

        return BacktestResult(
            strategy_name=strategy_name,
            config={"backtest_config": config.__dict__},
            performance_metrics=metrics,
            trade_history=trade_history,
            portfolio_values=portfolio_values,
            execution_time=0.0,  # Will be set by caller
            metadata=metadata
        )

    def compare_strategies(
        self,
        strategy_names: List[str],
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
        **kwargs
    ) -> Dict[str, BacktestResult]:
        """
        Compare multiple strategies on the same data.

        Args:
            strategy_names: List of strategy names to compare
            data: Market data
            config: Backtest configuration
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping strategy names to results
        """
        results = {}

        for name in strategy_names:
            try:
                self.logger.info(f"Running comparison for strategy: {name}")
                result = self.run_backtest(name, data, config, save_results=False, **kwargs)
                results[name] = result
            except Exception as e:
                self.logger.error(f"Failed to run strategy {name}: {e}")
                continue

        # Generate comparison report
        if results:
            self.results_manager.generate_comparison_report(results)

        return results

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies."""
        return list(self.strategies.keys())

    def run_advanced_analysis(
        self,
        result: BacktestResult,
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Union[float, str, dict, list]]:
        """
        Run advanced analysis using integrated archived analysis functions.

        Args:
            result: Backtest result to analyze
            analysis_types: List of analysis types to run (optional)

        Returns:
            Advanced analysis results
        """
        if analysis_types is None:
            analysis_types = ["risk_detailed", "temporal", "regime"]

        analyzer = BacktestAnalyzer()
        advanced_results = {}

        for analysis_type in analysis_types:
            try:
                if analysis_type == "risk_detailed":
                    # Use enhanced risk metrics from archived scripts
                    advanced_results["risk_detailed"] = analyzer.analyze_risk_metrics_detailed(result)

                elif analysis_type == "temporal":
                    # Enhanced temporal analysis
                    advanced_results["temporal"] = self._analyze_temporal_patterns_enhanced(result)

                elif analysis_type == "regime":
                    # Market regime analysis (requires price data)
                    if hasattr(result, 'price_data'):
                        advanced_results["regime"] = analyzer.analyze_market_regimes(result.price_data)
                    else:
                        advanced_results["regime"] = {"error": "Price data not available for regime analysis"}

                elif analysis_type == "feature_importance":
                    # Feature importance analysis (requires feature data)
                    if hasattr(result, 'feature_data'):
                        advanced_results["feature_importance"] = analyzer.analyze_feature_importance(result.feature_data)
                    else:
                        advanced_results["feature_importance"] = {"error": "Feature data not available"}

                elif analysis_type == "walkforward":
                    # Walkforward analysis (requires full dataset and strategy function)
                    advanced_results["walkforward"] = {"error": "Walkforward analysis requires additional setup"}

            except Exception as e:
                self.logger.warning(f"Failed to run {analysis_type} analysis: {e}")
                advanced_results[analysis_type] = {"error": str(e)}

        return advanced_results

    def _analyze_temporal_patterns_enhanced(self, result: BacktestResult) -> Dict[str, Union[float, str, dict, list]]:
        """Enhanced temporal pattern analysis."""
        portfolio_values = pd.Series(result.portfolio_values)

        # Monthly analysis
        monthly_values = portfolio_values.resample('M').last() if hasattr(portfolio_values.index, 'freq') else portfolio_values
        monthly_returns = monthly_values.pct_change().dropna()

        if len(monthly_returns) >= 12:
            # Seasonal analysis
            monthly_avg_returns = monthly_returns.groupby(monthly_returns.index.month).mean()
            monthly_volatility = monthly_returns.groupby(monthly_returns.index.month).std()

            best_month = monthly_avg_returns.idxmax()
            worst_month = monthly_avg_returns.idxmin()

            seasonal_analysis = {
                "best_month": int(best_month),
                "worst_month": int(worst_month),
                "best_month_return": float(monthly_avg_returns.loc[best_month]),
                "worst_month_return": float(monthly_avg_returns.loc[worst_month]),
                "monthly_average_returns": monthly_avg_returns.to_dict(),
                "monthly_volatility": monthly_volatility.to_dict(),
            }
        else:
            seasonal_analysis = {"error": "Insufficient data for seasonal analysis"}

        # Drawdown duration analysis
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        drawdown_periods = (drawdown < 0).astype(int).groupby((drawdown >= 0).cumsum()).sum()

        return {
            "seasonal_analysis": seasonal_analysis,
            "drawdown_analysis": {
                "average_drawdown_duration": float(drawdown_periods.mean()),
                "max_drawdown_duration": int(drawdown_periods.max()),
                "total_drawdown_periods": int(len(drawdown_periods[drawdown_periods > 0])),
            },
            "recovery_analysis": self._analyze_recovery_patterns(portfolio_values),
        }

    def _analyze_recovery_patterns(self, portfolio_values: pd.Series) -> Dict[str, Union[float, int]]:
        """Analyze recovery patterns after drawdowns."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak

        # Find drawdown periods
        in_drawdown = drawdown < -0.05  # 5% drawdown threshold
        drawdown_groups = in_drawdown.astype(int).groupby((~in_drawdown).cumsum()).sum()

        recovery_times = []
        for i in range(len(portfolio_values) - 1):
            if drawdown.iloc[i] < -0.05:  # In drawdown
                # Find recovery point
                recovery_idx = None
                for j in range(i + 1, len(portfolio_values)):
                    if portfolio_values.iloc[j] >= peak.iloc[i]:
                        recovery_idx = j
                        break

                if recovery_idx is not None:
                    recovery_times.append(recovery_idx - i)

        return {
            "average_recovery_time": float(np.mean(recovery_times)) if recovery_times else 0.0,
            "max_recovery_time": int(np.max(recovery_times)) if recovery_times else 0,
            "successful_recoveries": len(recovery_times),
            "total_drawdowns": int(len(drawdown_groups[drawdown_groups > 0])),
        }