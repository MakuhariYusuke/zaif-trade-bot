#!/usr/bin/env python3
"""
Backtest Analysis Engine

Provides comprehensive analysis capabilities for backtest results,
including performance analysis, correlation analysis, risk analysis,
and SAC model comparison.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .unified_backtester import BacktestResult
from ...metrics import BacktestMetrics, MetricsCalculator
from ....utils.logging_utils import get_logger

logger = get_logger(__name__)


class BacktestAnalyzer:
    """
    Comprehensive backtest analysis engine.

    Features:
    - Performance analysis and comparison
    - Risk analysis and attribution
    - SAC learning outcome analysis
    - Signal quality assessment
    - Statistical significance testing
    """

    def __init__(self):
        """Initialize the backtest analyzer."""
        self.metrics_calculator = MetricsCalculator()

    def analyze_single_result(
        self,
        result: BacktestResult,
        benchmark_result: Optional[BacktestResult] = None
    ) -> Dict[str, Union[float, str, dict, list]]:
        """
        Analyze a single backtest result.

        Args:
            result: Backtest result to analyze
            benchmark_result: Optional benchmark result for comparison

        Returns:
            Analysis results dictionary
        """
        analysis = {
            "strategy_name": result.strategy_name,
            "performance_summary": self._create_performance_summary(result),
            "risk_analysis": self._analyze_risk(result),
            "trade_analysis": self._analyze_trades(result),
            "temporal_analysis": self._analyze_temporal_patterns(result),
        }

        if benchmark_result:
            analysis["benchmark_comparison"] = self._compare_with_benchmark(result, benchmark_result)

        return analysis

    def compare_strategies(
        self,
        results: Dict[str, BacktestResult],
        statistical_tests: bool = True
    ) -> Dict[str, Union[float, str, dict, list]]:
        """
        Compare multiple strategies.

        Args:
            results: Dictionary of strategy results
            statistical_tests: Whether to perform statistical significance tests

        Returns:
            Comparison analysis results
        """
        comparison = {
            "strategy_count": len(results),
            "performance_comparison": self._compare_performance(results),
            "risk_comparison": self._compare_risk(results),
            "trade_comparison": self._compare_trades(results),
        }

        if statistical_tests and len(results) >= 2:
            comparison["statistical_tests"] = self._perform_statistical_tests(results)

        return comparison

    def analyze_sac_learning_outcomes(
        self,
        sac_results: Dict[str, BacktestResult],
        learning_metadata: Optional[Dict[str, Union[str, int, float]]] = None
    ) -> Dict[str, Union[float, str, dict, list]]:
        """
        Analyze SAC learning outcomes across different models/versions.

        Args:
            sac_results: Dictionary of SAC model results
            learning_metadata: Additional learning metadata

        Returns:
            SAC learning outcome analysis
        """
        analysis = {
            "model_count": len(sac_results),
            "learning_progression": self._analyze_learning_progression(sac_results),
            "regime_adaptation": self._analyze_regime_adaptation(sac_results),
            "hyperparameter_sensitivity": self._analyze_hyperparameter_sensitivity(sac_results),
        }

        if learning_metadata:
            analysis["training_insights"] = self._extract_training_insights(learning_metadata)

        return analysis

    def analyze_signal_quality(
        self,
        signal_strategy_result: BacktestResult,
        market_data: pd.DataFrame
    ) -> Dict[str, Union[float, str, dict, list]]:
        """
        Analyze signal quality for signal-based strategies.

        Args:
            signal_strategy_result: Result from signal-based strategy
            market_data: Market data used in backtest

        Returns:
            Signal quality analysis
        """
        return {
            "signal_distribution": self._analyze_signal_distribution(signal_strategy_result),
            "signal_timing": self._analyze_signal_timing(signal_strategy_result, market_data),
            "signal_effectiveness": self._analyze_signal_effectiveness(signal_strategy_result),
            "pattern_recognition": self._analyze_pattern_recognition(signal_strategy_result),
        }

    def analyze_sac_signal_correlation(
        self,
        sac_result: BacktestResult,
        signal_result: BacktestResult
    ) -> Dict[str, Union[float, str, dict, list]]:
        """
        Analyze correlation between SAC decisions and Action Signal Guide signals.

        Args:
            sac_result: SAC strategy result
            signal_result: Signal strategy result

        Returns:
            Correlation analysis results
        """
        return {
            "action_correlation": self._calculate_action_correlation(sac_result, signal_result),
            "performance_correlation": self._calculate_performance_correlation(sac_result, signal_result),
            "regime_correlation": self._calculate_regime_correlation(sac_result, signal_result),
            "signal_contribution": self._assess_signal_contribution(sac_result, signal_result),
        }

    def _create_performance_summary(self, result: BacktestResult) -> Dict[str, float]:
        """Create performance summary."""
        metrics = result.performance_metrics
        return {
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
        }

    def _analyze_risk(self, result: BacktestResult) -> Dict[str, Union[float, str]]:
        """Analyze risk metrics."""
        portfolio_values = pd.Series(result.portfolio_values)
        returns = portfolio_values.pct_change().dropna()

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)

        # Expected Shortfall (CVaR)
        cvar_95 = returns[returns <= var_95].mean()

        # Tail risk measures
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        return {
            "value_at_risk_95": float(var_95),
            "conditional_var_95": float(cvar_95),
            "returns_skewness": float(skewness),
            "returns_kurtosis": float(kurtosis),
            "volatility": result.performance_metrics.volatility,
        }

    def _analyze_trades(self, result: BacktestResult) -> Dict[str, Union[int, float]]:
        """Analyze trade patterns."""
        trades = result.trade_history

        if not trades:
            return {"total_trades": 0, "message": "No trades executed"}

        # Trade duration analysis
        durations = []
        for trade in trades:
            if "entry_time" in trade and "exit_time" in trade:
                duration = (pd.to_datetime(trade["exit_time"]) - pd.to_datetime(trade["entry_time"])).total_seconds() / 3600  # hours
                durations.append(duration)

        return {
            "total_trades": len(trades),
            "avg_trade_duration_hours": float(np.mean(durations)) if durations else 0.0,
            "max_trade_duration_hours": float(np.max(durations)) if durations else 0.0,
            "min_trade_duration_hours": float(np.min(durations)) if durations else 0.0,
        }

    def _analyze_temporal_patterns(self, result: BacktestResult) -> Dict[str, Union[float, str]]:
        """Analyze temporal patterns in performance."""
        portfolio_values = pd.Series(result.portfolio_values)

        # Monthly returns
        monthly_returns = portfolio_values.resample('M').last().pct_change().dropna()

        # Best and worst months
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()

        # Seasonal patterns (if enough data)
        if len(monthly_returns) >= 12:
            monthly_avg = monthly_returns.groupby(monthly_returns.index.month).mean()
            best_month_of_year = monthly_avg.idxmax()
            worst_month_of_year = monthly_avg.idxmin()
        else:
            best_month_of_year = None
            worst_month_of_year = None

        return {
            "best_month_return": float(best_month),
            "worst_month_return": float(worst_month),
            "best_month_of_year": best_month_of_year,
            "worst_month_of_year": worst_month_of_year,
        }

    def _compare_with_benchmark(
        self,
        result: BacktestResult,
        benchmark: BacktestResult
    ) -> Dict[str, float]:
        """Compare strategy with benchmark."""
        return {
            "return_difference": result.performance_metrics.total_return - benchmark.performance_metrics.total_return,
            "sharpe_difference": result.performance_metrics.sharpe_ratio - benchmark.performance_metrics.sharpe_ratio,
            "drawdown_difference": result.performance_metrics.max_drawdown - benchmark.performance_metrics.max_drawdown,
        }

    def _compare_performance(self, results: Dict[str, BacktestResult]) -> Dict[str, list]:
        """Compare performance across strategies."""
        strategy_names = list(results.keys())
        total_returns = [results[name].performance_metrics.total_return for name in strategy_names]
        sharpe_ratios = [results[name].performance_metrics.sharpe_ratio for name in strategy_names]

        return {
            "strategy_names": strategy_names,
            "total_returns": total_returns,
            "sharpe_ratios": sharpe_ratios,
            "best_performer": strategy_names[np.argmax(total_returns)],
        }

    def _compare_risk(self, results: Dict[str, BacktestResult]) -> Dict[str, list]:
        """Compare risk metrics across strategies."""
        strategy_names = list(results.keys())
        max_drawdowns = [results[name].performance_metrics.max_drawdown for name in strategy_names]
        volatilities = [results[name].performance_metrics.volatility for name in strategy_names]

        return {
            "strategy_names": strategy_names,
            "max_drawdowns": max_drawdowns,
            "volatilities": volatilities,
            "lowest_risk_strategy": strategy_names[np.argmin(max_drawdowns)],
        }

    def _compare_trades(self, results: Dict[str, BacktestResult]) -> Dict[str, list]:
        """Compare trading patterns across strategies."""
        strategy_names = list(results.keys())
        total_trades = [results[name].performance_metrics.total_trades for name in strategy_names]
        win_rates = [results[name].performance_metrics.win_rate for name in strategy_names]

        return {
            "strategy_names": strategy_names,
            "total_trades": total_trades,
            "win_rates": win_rates,
        }

    def _perform_statistical_tests(self, results: Dict[str, BacktestResult]) -> Dict[str, Union[float, str]]:
        """Perform statistical significance tests."""
        if len(results) < 2:
            return {"error": "Need at least 2 strategies for statistical tests"}

        # Extract returns for each strategy (simplified - would need actual return series)
        # This is a placeholder for more sophisticated statistical testing
        return {
            "tests_performed": ["t-test_placeholder", "anova_placeholder"],
            "significance_level": 0.05,
            "note": "Statistical testing requires detailed return series",
        }

    def _analyze_learning_progression(self, sac_results: Dict[str, BacktestResult]) -> Dict[str, Union[float, list]]:
        """Analyze learning progression across SAC versions."""
        # Sort by version/model name (assuming naming convention)
        sorted_results = sorted(sac_results.items(), key=lambda x: x[0])

        progression = {
            "versions": [name for name, _ in sorted_results],
            "returns_progression": [result.performance_metrics.total_return for _, result in sorted_results],
            "sharpe_progression": [result.performance_metrics.sharpe_ratio for _, result in sorted_results],
        }

        return progression

    def _analyze_regime_adaptation(self, sac_results: Dict[str, BacktestResult]) -> Dict[str, Union[float, str]]:
        """Analyze regime adaptation effectiveness."""
        # Placeholder for regime analysis
        return {
            "regime_adaptation_score": 0.0,
            "note": "Regime analysis requires regime classification data",
        }

    def _analyze_hyperparameter_sensitivity(self, sac_results: Dict[str, BacktestResult]) -> Dict[str, Union[float, str]]:
        """Analyze hyperparameter sensitivity."""
        # Placeholder for hyperparameter analysis
        return {
            "sensitivity_analysis": "pending",
            "note": "Requires hyperparameter metadata",
        }

    def _extract_training_insights(self, metadata: Dict[str, Union[str, int, float]]) -> Dict[str, Union[str, int, float]]:
        """Extract insights from training metadata."""
        return metadata

    def _analyze_signal_distribution(self, result: BacktestResult) -> Dict[str, Union[int, float]]:
        """Analyze signal distribution."""
        # Placeholder - would need signal data from strategy
        return {"total_signals": 0, "note": "Signal data not available in result"}

    def _analyze_signal_timing(self, result: BacktestResult, market_data: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """Analyze signal timing."""
        return {"timing_analysis": "pending", "note": "Requires signal timing data"}

    def _analyze_signal_effectiveness(self, result: BacktestResult) -> Dict[str, Union[float, str]]:
        """Analyze signal effectiveness."""
        return {"effectiveness_score": 0.0, "note": "Requires signal outcome data"}

    def _analyze_pattern_recognition(self, result: BacktestResult) -> Dict[str, Union[float, str]]:
        """Analyze pattern recognition accuracy."""
        return {"pattern_accuracy": 0.0, "note": "Requires pattern data"}

    def _calculate_action_correlation(self, sac_result: BacktestResult, signal_result: BacktestResult) -> Dict[str, Union[float, str]]:
        """Calculate action correlation between SAC and signals."""
        return {"correlation_coefficient": 0.0, "note": "Requires aligned action data"}

    def _calculate_performance_correlation(self, sac_result: BacktestResult, signal_result: BacktestResult) -> Dict[str, Union[float, str]]:
        """Calculate performance correlation."""
        return {"performance_correlation": 0.0, "note": "Requires detailed performance data"}

    def _calculate_regime_correlation(self, sac_result: BacktestResult, signal_result: BacktestResult) -> Dict[str, Union[float, str]]:
        """Calculate regime correlation."""
        return {"regime_correlation": 0.0, "note": "Requires regime data"}

    def _assess_signal_contribution(self, sac_result: BacktestResult, signal_result: BacktestResult) -> Dict[str, Union[float, str]]:
        """Assess signal contribution to SAC performance."""
        return {"signal_contribution_score": 0.0, "note": "Requires integrated analysis"}

    def analyze_risk_metrics_detailed(
        self,
        result: BacktestResult,
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Union[float, str, dict, list]]:
        """
        Detailed risk metrics analysis using archived analysis functions.

        Args:
            result: Backtest result to analyze
            confidence_level: Confidence level for VaR calculations
            risk_free_rate: Risk-free rate for Sharpe ratio

        Returns:
            Detailed risk analysis results
        """
        portfolio_values = pd.Series(result.portfolio_values)
        returns = portfolio_values.pct_change().dropna()

        if len(returns) < 2:
            return {"error": "Insufficient data for risk analysis"}

        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized

        # Sharpe ratio
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        # Value at Risk (Historical simulation)
        var_historical = np.percentile(returns, (1 - confidence_level) * 100)

        # Expected Shortfall (CVaR)
        tail_returns = returns[returns <= var_historical]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_historical

        # Maximum drawdown
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = abs(result.performance_metrics.total_return / max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        sortino_ratio = (returns.mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0

        # Omega ratio
        threshold = 0.0  # Minimum acceptable return
        omega_ratio = (returns[returns > threshold].sum() /
                      abs(returns[returns < threshold].sum())) if returns[returns < threshold].sum() != 0 else float('inf')

        return {
            "volatility_annualized": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "omega_ratio": float(omega_ratio),
            "value_at_risk": float(var_historical),
            "expected_shortfall": float(expected_shortfall),
            "max_drawdown": float(max_drawdown),
            "returns_skewness": float(stats.skew(returns)),
            "returns_kurtosis": float(stats.kurtosis(returns)),
            "worst_return": float(returns.min()),
            "best_return": float(returns.max()),
        }

    def analyze_feature_importance(
        self,
        feature_data: pd.DataFrame,
        target_column: str = "action",
        method: str = "permutation"
    ) -> Dict[str, Union[float, str, dict, list]]:
        """
        Analyze feature importance using archived feature analysis functions.

        Args:
            feature_data: DataFrame with features and target
            target_column: Target column name
            method: Importance calculation method ("permutation", "tree", "correlation")

        Returns:
            Feature importance analysis results
        """
        if target_column not in feature_data.columns:
            return {"error": f"Target column '{target_column}' not found"}

        # Identify feature columns
        exclude_cols = ["timestamp", "ts", "pair", "side", "pnl", "win", "source"]
        feature_cols = [
            col for col in feature_data.columns
            if col not in exclude_cols and col != target_column
            and feature_data[col].dtype in ["float64", "int64"]
        ]

        if not feature_cols:
            return {"error": "No suitable feature columns found"}

        X = feature_data[feature_cols].fillna(0)
        y = feature_data[target_column]

        importance_results = {}

        if method == "permutation":
            # Use permutation importance
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.inspection import permutation_importance
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

            importance_results = {
                "method": "permutation_importance",
                "features": feature_cols,
                "importance_scores": perm_importance.importances_mean.tolist(),
                "importance_std": perm_importance.importances_std.tolist(),
            }

        elif method == "correlation":
            # Correlation-based importance
            correlations = {}
            for col in feature_cols:
                corr = abs(X[col].corr(y))
                correlations[col] = float(corr) if not np.isnan(corr) else 0.0

            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

            importance_results = {
                "method": "correlation",
                "features": [f[0] for f in sorted_features],
                "importance_scores": [f[1] for f in sorted_features],
            }

        return importance_results

    def analyze_market_regimes(
        self,
        price_data: pd.DataFrame,
        returns_data: Optional[pd.Series] = None,
        method: str = "volatility_clustering"
    ) -> Dict[str, Union[float, str, dict, list]]:
        """
        Analyze market regimes using archived regime evaluation functions.

        Args:
            price_data: Price data for regime detection
            returns_data: Returns data (optional)
            method: Regime detection method

        Returns:
            Market regime analysis results
        """
        if returns_data is None:
            returns_data = price_data['close'].pct_change().dropna()

        # Simple regime detection based on volatility clustering
        if method == "volatility_clustering":
            # Calculate rolling volatility
            volatility = returns_data.rolling(window=20).std()

            # Define regimes based on volatility percentiles
            vol_median = volatility.median()
            vol_high = volatility.quantile(0.75)

            regimes = []
            for vol in volatility:
                if vol > vol_high:
                    regimes.append("high_volatility")
                elif vol > vol_median:
                    regimes.append("normal_volatility")
                else:
                    regimes.append("low_volatility")

        elif method == "trend_following":
            # Simple trend-based regime detection
            ma_short = price_data['close'].rolling(window=10).mean()
            ma_long = price_data['close'].rolling(window=30).mean()

            regimes = []
            for i in range(len(price_data)):
                if ma_short.iloc[i] > ma_long.iloc[i]:
                    regimes.append("uptrend")
                else:
                    regimes.append("downtrend")

        else:
            return {"error": f"Unknown regime detection method: {method}"}

        # Analyze regime distribution
        regime_counts = pd.Series(regimes).value_counts().to_dict()
        total_periods = len(regimes)
        regime_distribution = {k: v/total_periods for k, v in regime_counts.items()}

        return {
            "method": method,
            "total_periods": total_periods,
            "regime_counts": regime_counts,
            "regime_distribution": regime_distribution,
            "regime_transitions": self._analyze_regime_transitions(regimes),
        }

    def _analyze_regime_transitions(self, regimes: list) -> Dict[str, Union[int, float]]:
        """Analyze regime transition patterns."""
        transitions = {}
        for i in range(1, len(regimes)):
            transition = f"{regimes[i-1]}_to_{regimes[i]}"
            transitions[transition] = transitions.get(transition, 0) + 1

        return transitions

    def perform_walkforward_analysis(
        self,
        data: pd.DataFrame,
        strategy_func: callable,
        window_size: int = 252,  # 1 year
        step_size: int = 21,    # 1 month
        min_train_size: int = 126  # 6 months
    ) -> Dict[str, Union[float, str, dict, list]]:
        """
        Perform walkforward analysis using archived walkforward functions.

        Args:
            data: Market data for analysis
            strategy_func: Function that takes training data and returns strategy
            window_size: Size of each analysis window
            step_size: Step size for moving windows
            min_train_size: Minimum training data size

        Returns:
            Walkforward analysis results
        """
        if len(data) < window_size + min_train_size:
            return {"error": "Insufficient data for walkforward analysis"}

        windows = []
        metrics_history = []

        start_idx = min_train_size
        window_id = 0

        while start_idx + window_size <= len(data):
            # Define window
            train_end = start_idx
            val_end = min(train_end + window_size // 3, len(data))
            test_end = min(train_end + window_size, len(data))

            # Training data
            train_data = data.iloc[:train_end]

            # Validation data (if available)
            val_data = data.iloc[train_end:val_end] if val_end > train_end else None

            # Test data
            test_data = data.iloc[val_end:test_end] if test_end > val_end else data.iloc[train_end:test_end]

            try:
                # Train strategy
                strategy = strategy_func(train_data)

                # Evaluate on test data (simplified - would need actual backtest)
                # This is a placeholder for the actual walkforward implementation
                test_returns = test_data['close'].pct_change().dropna()

                window_metrics = {
                    "window_id": window_id,
                    "train_periods": len(train_data),
                    "test_periods": len(test_data),
                    "test_return": float(test_returns.mean() * 252),  # Annualized
                    "test_volatility": float(test_returns.std() * np.sqrt(252)),
                    "test_sharpe": float(test_returns.mean() / test_returns.std() * np.sqrt(252)) if test_returns.std() > 0 else 0.0,
                }

                metrics_history.append(window_metrics)
                windows.append({
                    "window_id": window_id,
                    "train_end": train_end,
                    "test_start": val_end,
                    "test_end": test_end,
                })

            except Exception as e:
                self.logger.warning(f"Failed to analyze window {window_id}: {e}")
                continue

            start_idx += step_size
            window_id += 1

        if not metrics_history:
            return {"error": "No valid windows analyzed"}

        # Aggregate results
        sharpe_ratios = [m["test_sharpe"] for m in metrics_history]
        returns = [m["test_return"] for m in metrics_history]

        return {
            "total_windows": len(metrics_history),
            "windows": windows,
            "metrics_history": metrics_history,
            "aggregate_metrics": {
                "mean_sharpe": float(np.mean(sharpe_ratios)),
                "std_sharpe": float(np.std(sharpe_ratios)),
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "sharpe_consistency": float(np.mean(sharpe_ratios) / np.std(sharpe_ratios)) if np.std(sharpe_ratios) > 0 else 0.0,
            },
        }