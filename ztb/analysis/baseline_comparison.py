"""
Baseline comparison output for evaluation metrics.

Compares trained model performance against baseline strategies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ztb.io.text_io import write_text
# 年間取引日数

@dataclass
class BaselineResult:
    """Result from a baseline strategy."""

    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    metrics: dict[str, Any]

@dataclass
class ComparisonReport:
    """Comparison report between model and baselines."""

    model_result: BaselineResult
    baseline_results: list[BaselineResult]
    superiority_metrics: dict[str, float]
    statistical_tests: dict[str, float]

class BaselineStrategy:
    """Base class for baseline strategies."""

    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate(self, price_data: pd.DataFrame, **kwargs: Any) -> BaselineResult:
        """Evaluate strategy on price data."""
        raise NotImplementedError

class BuyAndHoldStrategy(BaselineStrategy):
    """Buy and hold baseline strategy."""

    def evaluate(self, price_data: pd.DataFrame, **kwargs: Any) -> BaselineResult:
        """Evaluate buy and hold strategy."""
        # Simple buy and hold return calculation
        start_price = price_data["close"].iloc[0]
        end_price = price_data["close"].iloc[-1]
        total_return = cast(float, (end_price - start_price) / start_price)

        # Calculate basic metrics
        returns = price_data["close"].pct_change().dropna()
        from ztb.metrics.metrics import sharpe_ratio as calc_sharpe_ratio

        sharpe_ratio = calc_sharpe_ratio(returns)

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = cast(float, drawdown.min())

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
        **kwargs: Any,
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

        total_return = (
            cast(float, (1 + price_data["strategy_returns"].dropna()).prod()) - 1
        )
        returns = price_data["strategy_returns"].dropna()

        from ztb.metrics.metrics import sharpe_ratio as calc_sharpe_ratio

        sharpe_ratio = calc_sharpe_ratio(returns)

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = cast(float, drawdown.min()) if len(drawdown) > 0 else 0

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

    def __init__(self) -> None:
        self.strategies = {
            "buy_hold": BuyAndHoldStrategy("Buy and Hold"),
            "sma_crossover": SMAStrategy("SMA Crossover"),
        }

    def add_strategy(self, strategy: BaselineStrategy) -> None:
        """Add a custom baseline strategy."""
        self.strategies[strategy.name] = strategy

    def compare(
        self,
        model_result: BaselineResult,
        price_data: pd.DataFrame,
        strategies: list[str] | None = None,
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
        self, model: BaselineResult, baselines: list[BaselineResult]
    ) -> dict[str, float]:
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
        self, model: BaselineResult, baselines: list[BaselineResult]
    ) -> dict[str, float]:
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
        self, comparison: ComparisonReport, output_path: str | None = None
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
            write_text(output_path, report)

        return report

class BaselineComparisonEngine:
    """Engine for comparing model performance against baseline strategies."""

    def __init__(self, price_data: pd.DataFrame | None = None):
        self.price_data = price_data
        self.baseline_strategies = [
            BuyAndHoldStrategy("Buy and Hold"),
            SMAStrategy("SMA Crossover"),
        ]

    def generate_baseline_results(self, price_data: pd.DataFrame) -> list[BaselineResult]:
        """Generate results for all baseline strategies."""
        results = []
        for strategy in self.baseline_strategies:
            try:
                result = strategy.evaluate(price_data)
                results.append(result)
            except Exception as e:
                print(f"Failed to evaluate {strategy.name}: {e}")
        return results

    def compare_with_baselines(
        self, model_result: BaselineResult, baseline_results: list[BaselineResult]
    ) -> ComparisonReport:
        """Compare model result with baseline results."""
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
        self, model: BaselineResult, baselines: list[BaselineResult]
    ) -> dict[str, float]:
        """Calculate superiority metrics over baselines."""
        metrics = {}
        for baseline in baselines:
            prefix = baseline.strategy_name.lower().replace(" ", "_")
            metrics[f"{prefix}_return_superiority"] = (
                model.total_return - baseline.total_return
            )
            metrics[f"{prefix}_sharpe_superiority"] = (
                model.sharpe_ratio - baseline.sharpe_ratio
            )
            metrics[f"{prefix}_win_rate_superiority"] = (
                model.win_rate - baseline.win_rate
            )
        return metrics

    def _run_statistical_tests(
        self, model: BaselineResult, baselines: list[BaselineResult]
    ) -> dict[str, float]:
        """Run simplified statistical tests."""
        tests = {}
        for baseline in baselines:
            prefix = baseline.strategy_name.lower().replace(" ", "_")
            return_diff = model.total_return - baseline.total_return
            tests[f"{prefix}_return_significant"] = (
                1.0 if abs(return_diff) > 0.01 else 0.0
            )
        return tests

    def compare_multiple_evaluations(self, evaluations: list["WalkForwardResult"]) -> list[str]:
        """Compare multiple WalkForwardResult evaluations."""
        from ztb.evaluation.walk_forward.types import WalkForwardResult
        
        if len(evaluations) < 2:
            return ["Need at least 2 results for comparison"]
        
        comparisons = []
        for i in range(len(evaluations)):
            for j in range(i + 1, len(evaluations)):
                eval1 = evaluations[i]
                eval2 = evaluations[j]
                roi_diff = eval1.average_test_roi - eval2.average_test_roi
                sharpe_diff = eval1.average_sharpe - eval2.average_sharpe
                comparisons.append(
                    f"Eval {i} vs {j}: ROI diff {roi_diff:.4f}, Sharpe diff {sharpe_diff:.4f}"
                )
        return comparisons

# Global instance
_baseline_engine = BaselineComparisonEngine()

def get_baseline_comparison_engine() -> BaselineComparisonEngine:
    """Get global baseline comparison engine."""
    return _baseline_engine
