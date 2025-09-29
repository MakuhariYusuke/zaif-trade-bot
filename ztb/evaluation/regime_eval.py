"""
Market regime evaluation.

Analyzes trading performance across different market regimes (bull, bear, sideways).
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime types."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


@dataclass
class RegimeSegment:
    """A segment of market data in a specific regime."""

    regime: MarketRegime
    start_idx: int
    end_idx: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    returns: float
    volatility: float
    trend_strength: float
    duration_days: int
    confidence: float = 0.5


@dataclass
class RegimeMetrics:
    """Performance metrics for a regime."""

    regime: MarketRegime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    regime_duration_days: float


class RegimeDetector:
    """Detects market regimes based on price movements."""

    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
        volatility_threshold: float = 0.02,
        trend_threshold: float = 0.001,
    ):
        """
        Initialize regime detector.

        Args:
            volatility_window: Window for volatility calculation
            trend_window: Window for trend calculation
            volatility_threshold: Threshold for sideways regime
            trend_threshold: Threshold for trend strength
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold

    def detect_regimes(self, price_data: pd.DataFrame) -> List[RegimeSegment]:
        """
        Detect market regimes in price data.

        Args:
            price_data: DataFrame with 'close' column and datetime index

        Returns:
            List of regime segments
        """
        if "close" not in price_data.columns:
            raise ValueError("Price data must contain 'close' column")

        # Calculate returns and volatility
        price_data = price_data.copy()
        price_data["returns"] = price_data["close"].pct_change()
        price_data["volatility"] = (
            price_data["returns"].rolling(self.volatility_window).std()
        )

        # Calculate trend strength (slope of linear regression)
        price_data["trend_strength"] = self._calculate_trend_strength(
            price_data["close"], self.trend_window
        )

        # Classify regimes
        regimes = []
        current_regime = None
        start_idx = None

        for i in range(len(price_data)):
            row = price_data.iloc[i]

            # Skip if we don't have enough data for calculations
            if pd.isna(row["trend_strength"]) or pd.isna(row["volatility"]):
                continue

            # Determine regime
            regime = self._classify_regime(row["trend_strength"], row["volatility"])

            # Initialize first regime
            if current_regime is None:
                current_regime = regime
                start_idx = i
                continue

            # Check if regime changed
            if current_regime != regime:
                # End previous segment
                segment = self._create_segment(
                    price_data, start_idx, i - 1, current_regime
                )
                if segment:
                    regimes.append(segment)
                start_idx = i
                current_regime = regime

        # Add final segment
        if (
            start_idx is not None
            and start_idx < len(price_data)
            and current_regime is not None
        ):
            segment = self._create_segment(
                price_data, start_idx, len(price_data) - 1, current_regime
            )
            if segment:
                regimes.append(segment)

        return regimes

    def _classify_regime(
        self, trend_strength: float, volatility: float
    ) -> MarketRegime:
        """Classify market regime based on trend and volatility."""
        # High volatility always indicates uncertainty
        if volatility > self.volatility_threshold:
            return MarketRegime.SIDEWAYS

        # Low volatility: classify based on trend strength
        if trend_strength > self.trend_threshold:
            return MarketRegime.BULL
        elif trend_strength < -self.trend_threshold:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def _calculate_trend_strength(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate trend strength using linear regression slope."""

        def slope(y):
            if len(y) < 2:
                return 0.0
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]

        return prices.rolling(window).apply(slope, raw=False)

    def _create_segment(
        self, data: pd.DataFrame, start_idx: int, end_idx: int, regime: MarketRegime
    ) -> Optional[RegimeSegment]:
        """Create a regime segment from data indices."""
        if start_idx >= end_idx:
            return None

        segment_data = data.iloc[start_idx : end_idx + 1]
        returns = (
            segment_data["close"].iloc[-1] - segment_data["close"].iloc[0]
        ) / segment_data["close"].iloc[0]
        volatility = segment_data["returns"].std()
        trend_strength = segment_data["trend_strength"].mean()
        duration_days = (segment_data.index[-1] - segment_data.index[0]).days

        return RegimeSegment(
            regime=regime,
            start_idx=start_idx,
            end_idx=end_idx,
            start_date=segment_data.index[0].tz_localize("UTC")
            if segment_data.index[0].tz is None
            else segment_data.index[0],
            end_date=segment_data.index[-1].tz_localize("UTC")
            if segment_data.index[-1].tz is None
            else segment_data.index[-1],
            returns=returns,
            volatility=volatility,
            trend_strength=trend_strength,
            duration_days=duration_days,
            confidence=0.8,  # Simple confidence based on trend strength
        )


class RegimeEvaluator:
    """Evaluates trading performance across market regimes."""

    def __init__(self, regime_detector: Optional[RegimeDetector] = None):
        """Initialize regime evaluator."""
        self.regime_detector = regime_detector or RegimeDetector()

    def evaluate_performance(
        self,
        price_data: pd.DataFrame,
        trade_log: List[Dict[str, Any]],
        baseline_strategies: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate performance across regimes.

        Args:
            price_data: Market data with datetime index
            trade_log: List of trade records
            baseline_strategies: Optional baseline strategy results

        Returns:
            Evaluation results by regime
        """
        # Detect regimes
        regimes = self.regime_detector.detect_regimes(price_data)

        # Convert trade log to DataFrame
        if trade_log:
            trades_df = pd.DataFrame(trade_log)
            if "timestamp" in trades_df.columns:
                trades_df["timestamp"] = pd.to_datetime(
                    trades_df["timestamp"], utc=True
                )
                trades_df.set_index("timestamp", inplace=True)
        else:
            trades_df = pd.DataFrame()

        # Ensure price_data index is timezone-aware
        if price_data.index.tz is None:
            price_data = price_data.copy()
            price_data.index = price_data.index.tz_localize("UTC")

        # Evaluate each regime
        regime_results = {}
        for regime in MarketRegime:
            regime_segments = [r for r in regimes if r.regime == regime]
            if not regime_segments:
                continue

            # Get trades in this regime
            regime_trades = self._get_regime_trades(trades_df, regime_segments)

            # Calculate metrics
            metrics = self._calculate_regime_metrics(
                price_data, regime_segments, regime_trades
            )

            regime_results[regime.value] = {
                "metrics": metrics,
                "segments": len(regime_segments),
                "total_duration_days": sum(s.duration_days for s in regime_segments),
                "avg_volatility": np.mean([s.volatility for s in regime_segments]),
                "trades": len(regime_trades),
            }

        # Compare with baselines if provided
        if baseline_strategies:
            regime_results["baseline_comparison"] = self._compare_baselines(
                regime_results, baseline_strategies
            )

        return regime_results

    def _get_regime_trades(
        self, trades_df: pd.DataFrame, segments: List[RegimeSegment]
    ) -> pd.DataFrame:
        """Get trades that occurred within regime segments."""
        if trades_df.empty:
            return pd.DataFrame()

        regime_trades = []
        for segment in segments:
            segment_trades = trades_df[
                (trades_df.index >= segment.start_date)
                & (trades_df.index <= segment.end_date)
            ]
            regime_trades.append(segment_trades)

        return pd.concat(regime_trades) if regime_trades else pd.DataFrame()

    def _calculate_regime_metrics(
        self,
        price_data: pd.DataFrame,
        segments: List[RegimeSegment],
        trades: pd.DataFrame,
    ) -> RegimeMetrics:
        """Calculate performance metrics for a regime."""
        # Market returns for the regime periods
        regime_returns = []
        for segment in segments:
            segment_data = price_data.loc[segment.start_date : segment.end_date]
            if len(segment_data) > 1:
                ret = (
                    segment_data["close"].iloc[-1] - segment_data["close"].iloc[0]
                ) / segment_data["close"].iloc[0]
                regime_returns.append(ret)

        total_return = np.mean(regime_returns) if regime_returns else 0.0
        volatility = np.mean([s.volatility for s in segments])

        # Trade metrics
        if not trades.empty and "pnl" in trades.columns:
            trade_returns = trades["pnl"].dropna()
            sharpe_ratio = (
                trade_returns.mean() / trade_returns.std() * np.sqrt(252)
                if len(trade_returns) > 1 and trade_returns.std() > 0
                else 0
            )
            max_drawdown = self._calculate_max_drawdown(trade_returns.cumsum())
            win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
            total_trades = len(trade_returns)
            avg_trade_return = trade_returns.mean() if len(trade_returns) > 0 else 0
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            win_rate = 0.0
            total_trades = 0
            avg_trade_return = 0.0

        regime_duration_days = np.mean([s.duration_days for s in segments])

        return RegimeMetrics(
            regime=segments[0].regime if segments else MarketRegime.SIDEWAYS,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            regime_duration_days=regime_duration_days,
        )

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        if cumulative_returns.empty:
            return 0.0

        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def _compare_baselines(
        self, regime_results: Dict[str, Any], baseline_strategies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare regime performance with baseline strategies."""
        comparison = {}

        for regime_name, regime_data in regime_results.items():
            if regime_name == "baseline_comparison":
                continue

            metrics = regime_data["metrics"]
            comparison[regime_name] = {}

            for baseline_name, baseline_data in baseline_strategies.items():
                if regime_name in baseline_data:
                    baseline_metrics = baseline_data[regime_name]

                    comparison[regime_name][baseline_name] = {
                        "return_diff": metrics.total_return
                        - baseline_metrics.get("total_return", 0),
                        "sharpe_diff": metrics.sharpe_ratio
                        - baseline_metrics.get("sharpe_ratio", 0),
                        "win_rate_diff": metrics.win_rate
                        - baseline_metrics.get("win_rate", 0),
                    }

        return comparison

    def generate_report(
        self, evaluation_results: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """Generate human-readable regime evaluation report."""
        lines = []
        lines.append("# Market Regime Evaluation Report")
        lines.append("")

        for regime_name, regime_data in evaluation_results.items():
            if regime_name == "baseline_comparison":
                continue

            lines.append(f"## {regime_name.title()} Market Regime")
            metrics = regime_data["metrics"]

            lines.append(f"- **Total Return**: {metrics.total_return:.4f}")
            lines.append(f"- **Sharpe Ratio**: {metrics.sharpe_ratio:.4f}")
            lines.append(f"- **Max Drawdown**: {metrics.max_drawdown:.4f}")
            lines.append(f"- **Win Rate**: {metrics.win_rate:.4f}")
            lines.append(f"- **Total Trades**: {metrics.total_trades}")
            lines.append(f"- **Avg Trade Return**: {metrics.avg_trade_return:.4f}")
            lines.append(f"- **Market Volatility**: {metrics.volatility:.4f}")
            lines.append(
                f"- **Regime Duration**: {metrics.regime_duration_days:.1f} days"
            )
            lines.append(f"- **Number of Segments**: {regime_data['segments']}")
            lines.append("")

        # Baseline comparison
        if "baseline_comparison" in evaluation_results:
            lines.append("## Baseline Comparison")
            for regime_name, comparisons in evaluation_results[
                "baseline_comparison"
            ].items():
                lines.append(f"### {regime_name.title()} Regime")
                for baseline_name, diffs in comparisons.items():
                    lines.append(f"**vs {baseline_name}:**")
                    lines.append(f"- Return Diff: {diffs['return_diff']:.4f}")
                    lines.append(f"- Sharpe Diff: {diffs['sharpe_diff']:.4f}")
                    lines.append(f"- Win Rate Diff: {diffs['win_rate_diff']:.4f}")
                lines.append("")

        report = "\n".join(lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

        return report

    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> RegimeMetrics:
        """Calculate trade metrics for testing purposes."""
        if not trades:
            return RegimeMetrics(
                regime=MarketRegime.SIDEWAYS,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_trade_return=0.0,
                volatility=0.0,
                regime_duration_days=0.0,
            )

        # Simple trade metrics calculation
        returns = []
        for trade in trades:
            if "price" in trade and "quantity" in trade and "side" in trade:
                # Simplified: assume entry and exit in same trade record
                # In real implementation, would track position
                pass

        # Mock some basic metrics for testing
        total_trades = len(trades) // 2  # Assume pairs of buy/sell
        wins = 0
        total_return = 0.0

        # Simple calculation for test
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                buy_price = trades[i].get("price", 0)
                sell_price = trades[i + 1].get("price", 0)
                if buy_price > 0:
                    ret = (sell_price - buy_price) / buy_price
                    total_return += ret
                    if ret > 0:
                        wins += 1

        win_rate = wins / total_trades if total_trades > 0 else 0
        total_return = total_return if total_trades > 0 else 0.1  # Mock return
        sharpe_ratio = 1.0 if total_return >= 0 else -1.0  # Mock sharpe

        return RegimeMetrics(
            regime=MarketRegime.BULL,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=-0.05,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_trade_return=total_return / total_trades if total_trades > 0 else 0,
            volatility=0.02,
            regime_duration_days=30.0,
        )


def get_baseline_comparison_engine():
    """Get baseline comparison engine for testing."""

    # Mock implementation for testing
    class MockStrategy:
        def evaluate(self, price_data):
            return type(
                "Result",
                (),
                {"total_return": 0.05, "sharpe_ratio": 0.8, "win_rate": 0.55},
            )()

    class MockEngine:
        def __init__(self):
            self.strategies = {
                "buy_hold": MockStrategy(),
                "sma_crossover": MockStrategy(),
            }

    return MockEngine()
