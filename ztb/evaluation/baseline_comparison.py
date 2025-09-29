"""
Baseline comparison output for evaluation metrics.

Compares trained model performance against baseline strategies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BaselineResult:
    """Result from a baseline strategy."""

    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    metrics: Dict[str, Any]


@dataclass
class ComparisonReport:
    """Comparison report between model and baselines."""

    model_result: BaselineResult
    baseline_results: List[BaselineResult]
    superiority_metrics: Dict[str, float]
    statistical_tests: Dict[str, float]


class BaselineStrategy:
    """Base class for baseline strategies."""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, price_data: pd.DataFrame, **kwargs) -> BaselineResult:
        """Evaluate strategy on price data."""
        raise NotImplementedError


class BuyAndHoldStrategy(BaselineStrategy):
    """Buy and hold baseline strategy."""

    def evaluate(self, price_data: pd.DataFrame, **kwargs) -> BaselineResult:
        """Evaluate buy and hold strategy."""
        # Simple buy and hold return calculation
        start_price = price_data["close"].iloc[0]
        end_price = price_data["close"].iloc[-1]
        total_return = (end_price - start_price) / start_price

        # Calculate basic metrics
        returns = price_data["close"].pct_change().dropna()
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        )

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return BaselineResult(
            strategy_name=self.name,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=1.0 if total_return > 0 else 0.0,  # Binary for buy-hold
            total_trades=1,  # One position
            metrics={
                "start_price": start_price,
                "end_price": end_price,
                "holding_period_days": len(price_data),
            },
        )


class SMAStrategy(BaselineStrategy):
    """Simple Moving Average crossover strategy."""

    def evaluate(
        self,
        price_data: pd.DataFrame,
        fast_period: int = 10,
        slow_period: int = 30,
        **kwargs,
    ) -> BaselineResult:
        """Evaluate SMA crossover strategy."""
        # Calculate SMAs
        price_data = price_data.copy()
        price_data["fast_sma"] = price_data["close"].rolling(fast_period).mean()
        price_data["slow_sma"] = price_data["close"].rolling(slow_period).mean()

        # Generate signals
        price_data["signal"] = 0
        price_data.loc[price_data["fast_sma"] > price_data["slow_sma"], "signal"] = 1
        price_data.loc[price_data["fast_sma"] < price_data["slow_sma"], "signal"] = -1

        # Calculate position changes
        price_data["position_change"] = price_data["signal"].diff()
        trades = price_data[price_data["position_change"] != 0]

        # Calculate returns
        price_data["returns"] = price_data["close"].pct_change()
        price_data["strategy_returns"] = (
            price_data["signal"].shift(1) * price_data["returns"]
        )

        total_return = (1 + price_data["strategy_returns"].dropna()).prod() - 1
        returns = price_data["strategy_returns"].dropna()

        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252)
            if len(returns) > 0 and returns.std() > 0
            else 0
        )

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        # Win rate
        winning_trades = len(returns[returns > 0])
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return BaselineResult(
            strategy_name=self.name,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            metrics={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "total_signals": len(trades),
            },
        )


class BaselineComparisonEngine:
    """Engine for comparing model performance against baselines."""

    def __init__(self):
        self.strategies = {
            "buy_hold": BuyAndHoldStrategy("Buy and Hold"),
            "sma_crossover": SMAStrategy("SMA Crossover"),
        }

    def add_strategy(self, strategy: BaselineStrategy):
        """Add a custom baseline strategy."""
        self.strategies[strategy.name] = strategy

    def compare(
        self,
        model_result: BaselineResult,
        price_data: pd.DataFrame,
        strategies: Optional[List[str]] = None,
    ) -> ComparisonReport:
        """Compare model against baseline strategies."""
        if strategies is None:
            strategies = list(self.strategies.keys())

        baseline_results = []
        for strategy_name in strategies:
            if strategy_name in self.strategies:
                result = self.strategies[strategy_name].evaluate(price_data)
                baseline_results.append(result)

        # Calculate superiority metrics
        superiority = self._calculate_superiority(model_result, baseline_results)

        # Statistical tests (simplified)
        statistical_tests = self._run_statistical_tests(model_result, baseline_results)

        return ComparisonReport(
            model_result=model_result,
            baseline_results=baseline_results,
            superiority_metrics=superiority,
            statistical_tests=statistical_tests,
        )

    def _calculate_superiority(
        self, model: BaselineResult, baselines: List[BaselineResult]
    ) -> Dict[str, float]:
        """Calculate superiority metrics over baselines."""
        metrics = {}

        for baseline in baselines:
            prefix = baseline.strategy_name.lower().replace(" ", "_")

            metrics[f"{prefix}_return_diff"] = (
                model.total_return - baseline.total_return
            )
            metrics[f"{prefix}_sharpe_diff"] = (
                model.sharpe_ratio - baseline.sharpe_ratio
            )
            metrics[f"{prefix}_win_rate_diff"] = model.win_rate - baseline.win_rate

            # Risk-adjusted superiority
            if baseline.max_drawdown != 0:
                model_RAR = model.total_return / abs(model.max_drawdown)
                baseline_RAR = baseline.total_return / abs(baseline.max_drawdown)
                metrics[f"{prefix}_rar_diff"] = model_RAR - baseline_RAR

        return metrics

    def _run_statistical_tests(
        self, model: BaselineResult, baselines: List[BaselineResult]
    ) -> Dict[str, float]:
        """Run simplified statistical tests."""
        # Placeholder for statistical significance tests
        # In practice, would use t-tests, bootstrap, etc.
        tests = {}

        for baseline in baselines:
            prefix = baseline.strategy_name.lower().replace(" ", "_")
            # Simplified: just check if difference is significant
            return_diff = model.total_return - baseline.total_return
            tests[f"{prefix}_return_significant"] = (
                1.0 if abs(return_diff) > 0.01 else 0.0
            )

        return tests

    def generate_report(
        self, comparison: ComparisonReport, output_path: Optional[str] = None
    ) -> str:
        """Generate human-readable comparison report."""
        lines = []
        lines.append("# Baseline Comparison Report")
        lines.append("")

        # Model results
        lines.append("## Model Performance")
        lines.append(f"- Strategy: {comparison.model_result.strategy_name}")
        lines.append(f"- Total Return: {comparison.model_result.total_return:.4f}")
        lines.append(f"- Sharpe Ratio: {comparison.model_result.sharpe_ratio:.4f}")
        lines.append(f"- Max Drawdown: {comparison.model_result.max_drawdown:.4f}")
        lines.append(f"- Win Rate: {comparison.model_result.win_rate:.4f}")
        lines.append(f"- Total Trades: {comparison.model_result.total_trades}")
        lines.append("")

        # Baseline results
        lines.append("## Baseline Strategies")
        for baseline in comparison.baseline_results:
            lines.append(f"### {baseline.strategy_name}")
            lines.append(f"- Total Return: {baseline.total_return:.4f}")
            lines.append(f"- Sharpe Ratio: {baseline.sharpe_ratio:.4f}")
            lines.append(f"- Max Drawdown: {baseline.max_drawdown:.4f}")
            lines.append(f"- Win Rate: {baseline.win_rate:.4f}")
            lines.append(f"- Total Trades: {baseline.total_trades}")
            lines.append("")

        # Superiority metrics
        lines.append("## Superiority Metrics")
        for key, value in comparison.superiority_metrics.items():
            lines.append(f"- {key}: {value:.4f}")
        lines.append("")

        # Statistical tests
        lines.append("## Statistical Tests")
        for key, value in comparison.statistical_tests.items():
            lines.append(f"- {key}: {value:.4f}")

        report = "\n".join(lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)

        return report


# Global instance
_baseline_engine = BaselineComparisonEngine()


def get_baseline_comparison_engine() -> BaselineComparisonEngine:
    """Get global baseline comparison engine."""
    return _baseline_engine
