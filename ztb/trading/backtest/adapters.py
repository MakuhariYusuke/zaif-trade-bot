"""
Strategy adapters for backtesting.

Provides adapters to wrap different trading strategies for unified backtest interface.
"""

import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from ztb.trading.risk.optimizers.integrated_signal_filter import IntegratedSignalFilter
from ztb.utils.cache_utils import TTLCache


class RiskManager:
    """Risk management component for trading strategies."""

    def __init__(self):
        """Initialize risk manager with default parameters."""
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.stop_loss_atr_multiplier = 2.0  # 2x ATR for stop loss
        self.take_profit_atr_multiplier = 4.0  # 4x ATR for take profit
        self.max_consecutive_losses = 3  # Max consecutive losing trades
        self.circuit_breaker_threshold = 0.05  # 5% loss triggers circuit breaker

        # Risk tracking
        self.current_drawdown = 0.0
        self.consecutive_losses = 0
        self.portfolio_value = 1.0  # Normalized portfolio value
        self.circuit_breaker_active = False

        # Position tracking
        self.open_positions = {}  # position_id -> position_data

        # Performance optimization
        self.atr_cache = TTLCache(ttl_seconds=180)  # 3 minutes cache for ATR

    def calculate_atr_stop_levels(
        self, data: pd.DataFrame, entry_price: float, position_type: str
    ) -> Tuple[float, float]:
        """
        Calculate ATR-based stop loss and take profit levels.

        Args:
            data: Recent market data with ATR column
            entry_price: Entry price for the position
            position_type: 'long' or 'short'

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if "atr" not in data.columns:
            # Fallback to percentage-based stops if ATR not available
            atr_value = entry_price * 0.02  # 2% of entry price
        else:
            atr_value = data["atr"].iloc[-1]

        if position_type == "long":
            stop_loss = entry_price - (atr_value * self.stop_loss_atr_multiplier)
            take_profit = entry_price + (atr_value * self.take_profit_atr_multiplier)
        else:  # short
            stop_loss = entry_price + (atr_value * self.stop_loss_atr_multiplier)
            take_profit = entry_price - (atr_value * self.take_profit_atr_multiplier)

        return stop_loss, take_profit

    def should_open_position(
        self,
        signal_strength: float,
        market_volatility: float,
        current_portfolio_value: float,
    ) -> bool:
        """
        Determine if a new position should be opened based on risk criteria.

        Args:
            signal_strength: Strength of the trading signal (0-1)
            market_volatility: Current market volatility measure
            current_portfolio_value: Current portfolio value

        Returns:
            True if position should be opened, False otherwise
        """
        # Check circuit breaker
        if self.circuit_breaker_active:
            return False

        # Check drawdown limit
        if self.current_drawdown >= self.max_drawdown_limit:
            return False

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False

        # Check signal strength threshold
        min_signal_strength = 0.6 + (
            market_volatility * 0.2
        )  # Higher threshold in volatile markets
        if signal_strength < min_signal_strength:
            return False

        # Check position size limit
        max_position_value = current_portfolio_value * self.max_position_size
        if max_position_value < current_portfolio_value * 0.01:  # Minimum position size
            return False

        return True

    def should_close_position(
        self,
        position_data: Dict[str, Any],
        current_price: float,
        current_portfolio_value: float,
    ) -> Tuple[bool, str]:
        """
        Determine if a position should be closed based on risk criteria.

        Args:
            position_data: Position information including entry price, stop levels
            current_price: Current market price
            current_portfolio_value: Current portfolio value

        Returns:
            Tuple of (should_close, reason)
        """
        position_type = position_data["type"]
        entry_price = position_data["entry_price"]
        stop_loss = position_data["stop_loss"]
        take_profit = position_data["take_profit"]

        # Check stop loss
        if position_type == "long" and current_price <= stop_loss:
            return True, "stop_loss"
        elif position_type == "short" and current_price >= stop_loss:
            return True, "stop_loss"

        # Check take profit
        if position_type == "long" and current_price >= take_profit:
            return True, "take_profit"
        elif position_type == "short" and current_price <= take_profit:
            return True, "take_profit"

        # Check circuit breaker
        if self.circuit_breaker_active:
            return True, "circuit_breaker"

        # Check drawdown limit
        if self.current_drawdown >= self.max_drawdown_limit:
            return True, "max_drawdown"

        return False, ""

    def update_risk_metrics(self, trade_result: Optional[Dict[str, Any]] = None):
        """
        Update risk metrics after a trade or price movement.

        Args:
            trade_result: Optional trade result data
        """
        if trade_result:
            pnl = trade_result.get("pnl", 0)
            if pnl < 0:
                self.consecutive_losses += 1
                self.portfolio_value += pnl
            else:
                self.consecutive_losses = 0
                self.portfolio_value += pnl

        # Update drawdown
        self.current_drawdown = max(0, 1.0 - self.portfolio_value)

        # Check circuit breaker
        if self.portfolio_value <= (1.0 - self.circuit_breaker_threshold):
            self.circuit_breaker_active = True
        elif self.portfolio_value >= 0.98:  # Reset when recovered to 98%
            self.circuit_breaker_active = False

    def get_risk_adjusted_position_size(
        self, signal_strength: float, market_volatility: float
    ) -> float:
        """
        Calculate risk-adjusted position size based on signal strength and volatility.

        Args:
            signal_strength: Strength of the trading signal (0-1)
            market_volatility: Current market volatility measure

        Returns:
            Position size as fraction of portfolio (0-1)
        """
        # Base position size
        base_size = self.max_position_size

        # Adjust for signal strength
        strength_multiplier = 0.5 + (signal_strength * 0.5)  # 0.5 to 1.0

        # Adjust for volatility (smaller positions in high volatility)
        volatility_multiplier = 1.0 / (1.0 + market_volatility * 2.0)

        # Adjust for consecutive losses
        loss_multiplier = max(0.3, 1.0 - (self.consecutive_losses * 0.2))

        position_size = (
            base_size * strength_multiplier * volatility_multiplier * loss_multiplier
        )
        return min(position_size, self.max_position_size)

    def calculate_atr_stop_levels(
        self, data: pd.DataFrame, entry_price: float, position_type: str
    ) -> Tuple[float, float]:
        """
        Calculate ATR-based stop loss and take profit levels with caching.

        Args:
            data: Recent market data with OHLC columns
            entry_price: Entry price for the position
            position_type: 'long' or 'short'

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Create cache key
        cache_key = f"atr_{len(data)}_{hash(str(data.index[-1]) if len(data) > 0 else 'empty')}_{position_type}"

        # Check cache first
        cached_result = self.atr_cache.get(cache_key)
        if cached_result is not None:
            base_atr = cached_result
        else:
            # Calculate ATR efficiently
            if len(data) < 14:
                # Fallback to percentage-based stops if insufficient data
                base_atr = entry_price * 0.02  # 2% of entry price
            else:
                # Calculate True Range
                high = data["high"] if "high" in data.columns else data["High"]
                low = data["low"] if "low" in data.columns else data["Low"]
                close = data["close"] if "close" in data.columns else data["Close"]

                # True Range calculation
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                # ATR calculation (14-period SMA of True Range)
                base_atr = tr.rolling(14).mean().iloc[-1]
                if pd.isna(base_atr):
                    base_atr = entry_price * 0.02

            # Cache the base ATR
            self.atr_cache.set(cache_key, base_atr)

        if position_type == "long":
            stop_loss = entry_price - (base_atr * self.stop_loss_atr_multiplier)
            take_profit = entry_price + (base_atr * self.take_profit_atr_multiplier)
        else:  # short
            stop_loss = entry_price + (base_atr * self.stop_loss_atr_multiplier)
            take_profit = entry_price - (base_atr * self.take_profit_atr_multiplier)

        return stop_loss, take_profit


class DynamicThresholdManager:
    """Advanced dynamic threshold management for trading signals."""

    def __init__(self):
        """Initialize dynamic threshold manager."""
        self.regime_window = 50  # Window for regime detection
        self.adaptation_rate = 0.1  # How quickly thresholds adapt
        self.performance_memory = 100  # How many past signals to remember

        # Performance tracking
        self.signal_history = []
        self.threshold_history = []

        # Regime detection parameters
        self.trend_threshold = 0.001  # Minimum slope for trend detection
        self.volatility_threshold = 0.02  # Threshold for high volatility

    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect current market regime (trending, ranging, volatile).

        Args:
            data: Recent market data

        Returns:
            Market regime: 'trending_bull', 'trending_bear', 'ranging', 'volatile'
        """
        if len(data) < self.regime_window:
            return "unknown"

        recent_data = data.iloc[-self.regime_window :]

        # Calculate trend strength
        prices = recent_data["close"].values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend_strength = abs(slope) / np.mean(prices)

        # Calculate volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)

        # Detect regime
        if volatility > self.volatility_threshold:
            return "volatile"
        elif trend_strength > self.trend_threshold:
            return "trending_bull" if slope > 0 else "trending_bear"
        else:
            return "ranging"

    def calculate_adaptive_thresholds(
        self,
        data: pd.DataFrame,
        base_confidence: float = 0.7,
        base_strength: float = 0.4,
    ) -> Dict[str, float]:
        """
        Calculate adaptive thresholds based on market regime and performance.

        Args:
            data: Market data
            base_confidence: Base confidence threshold
            base_strength: Base signal strength threshold

        Returns:
            Dictionary with adaptive thresholds
        """
        regime = self.detect_market_regime(data)

        # Base adjustments by regime
        regime_adjustments = {
            "trending_bull": {"confidence": 0.9, "strength": 0.8},
            "trending_bear": {"confidence": 0.9, "strength": 0.8},
            "ranging": {
                "confidence": 1.1,
                "strength": 1.2,
            },  # Higher thresholds in ranging markets
            "volatile": {
                "confidence": 1.2,
                "strength": 1.3,
            },  # Much higher in volatile markets
            "unknown": {"confidence": 1.0, "strength": 1.0},
        }

        adjustment = regime_adjustments.get(
            regime, {"confidence": 1.0, "strength": 1.0}
        )

        # Performance-based adaptation
        performance_adjustment = self._calculate_performance_adjustment()

        confidence_threshold = (
            base_confidence
            * adjustment["confidence"]
            * performance_adjustment["confidence"]
        )
        signal_strength_threshold = (
            base_strength * adjustment["strength"] * performance_adjustment["strength"]
        )

        # Ensure reasonable bounds
        confidence_threshold = np.clip(confidence_threshold, 0.5, 0.9)
        signal_strength_threshold = np.clip(signal_strength_threshold, 0.2, 0.7)

        thresholds = {
            "confidence_threshold": confidence_threshold,
            "signal_strength_threshold": signal_strength_threshold,
            "regime": regime,
            "performance_adjustment": performance_adjustment,
        }

        # Store for performance tracking
        self.threshold_history.append(thresholds)

        return thresholds

    def _calculate_performance_adjustment(self) -> Dict[str, float]:
        """
        Calculate threshold adjustments based on recent performance.

        Returns:
            Performance-based adjustment factors
        """
        if len(self.signal_history) < 10:
            return {"confidence": 1.0, "strength": 1.0}

        recent_signals = self.signal_history[-20:]  # Last 20 signals
        win_rate = sum(1 for s in recent_signals if s.get("profitable", False)) / len(
            recent_signals
        )

        # Adjust thresholds based on win rate
        if win_rate > 0.6:  # Good performance, can be less strict
            adjustment = 0.9
        elif win_rate < 0.4:  # Poor performance, be more strict
            adjustment = 1.1
        else:
            adjustment = 1.0

        return {"confidence": adjustment, "strength": adjustment}

    def update_performance(self, signal_result: Dict[str, Any]):
        """
        Update performance tracking with signal result.

        Args:
            signal_result: Result of executed signal
        """
        self.signal_history.append(signal_result)

        # Keep memory limited
        if len(self.signal_history) > self.performance_memory:
            self.signal_history = self.signal_history[-self.performance_memory :]


class WalkForwardAnalyzer:
    """Walk-forward analysis for strategy validation and parameter optimization."""

    def __init__(
        self, train_window: int = 252, test_window: int = 63, step_size: int = 21
    ):
        """
        Initialize walk-forward analyzer.

        Args:
            train_window: Number of days for training/optimization
            test_window: Number of days for testing
            step_size: Number of days to advance each step
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

        # Analysis results storage
        self.walk_forward_results = []
        self.performance_metrics = []

    def run_walk_forward_analysis(
        self, data: pd.DataFrame, strategy_adapter: "ActionSignalGuideAdapter"
    ) -> Dict[str, Any]:
        """
        Run complete walk-forward analysis.

        Args:
            data: Full historical dataset
            strategy_adapter: Strategy adapter to test

        Returns:
            Analysis results and metrics
        """
        if len(data) < self.train_window + self.test_window:
            raise ValueError("Insufficient data for walk-forward analysis")

        results = []
        start_date = data.index[0]

        # Slide through data with expanding window
        for i in range(
            0, len(data) - self.train_window - self.test_window + 1, self.step_size
        ):
            train_end = i + self.train_window
            test_end = train_end + self.test_window

            if test_end > len(data):
                break

            # Split data
            train_data = data.iloc[i:train_end]
            test_data = data.iloc[train_end:test_end]

            # Optimize parameters on training data
            optimal_params = self._optimize_parameters(train_data, strategy_adapter)

            # Test parameters on test data
            test_result = self._evaluate_parameters(
                test_data, strategy_adapter, optimal_params
            )

            # Store results
            result = {
                "train_period": (train_data.index[0], train_data.index[-1]),
                "test_period": (test_data.index[0], test_data.index[-1]),
                "optimal_params": optimal_params,
                "test_performance": test_result,
                "train_end_date": train_data.index[-1],
                "test_end_date": test_data.index[-1],
            }

            results.append(result)

        self.walk_forward_results = results

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(results)

        return {
            "walk_forward_results": results,
            "overall_metrics": overall_metrics,
            "analysis_summary": self._generate_analysis_summary(
                results, overall_metrics
            ),
        }

    def _optimize_parameters(
        self, train_data: pd.DataFrame, strategy_adapter: "ActionSignalGuideAdapter"
    ) -> Dict[str, float]:
        """
        Optimize strategy parameters on training data.

        Args:
            train_data: Training dataset
            strategy_adapter: Strategy adapter

        Returns:
            Optimal parameters
        """
        # Simple parameter optimization - in practice, this would use more sophisticated methods
        best_params = {"confidence_threshold": 0.7, "signal_strength_threshold": 0.4}
        best_sharpe = -float("inf")

        # Test different parameter combinations
        confidence_levels = [0.6, 0.7, 0.8]
        strength_levels = [0.3, 0.4, 0.5]

        for conf in confidence_levels:
            for strength in strength_levels:
                # Temporarily set parameters
                original_params = strategy_adapter.hyperparameters.copy()
                strategy_adapter.hyperparameters.update(
                    {
                        "confidence_threshold": conf,
                        "signal_strength_threshold": strength,
                    }
                )

                # Evaluate performance
                performance = self._backtest_on_data(train_data, strategy_adapter)
                sharpe = performance.get("sharpe_ratio", -float("inf"))

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {
                        "confidence_threshold": conf,
                        "signal_strength_threshold": strength,
                    }

                # Restore original parameters
                strategy_adapter.hyperparameters = original_params

        return best_params

    def _evaluate_parameters(
        self,
        test_data: pd.DataFrame,
        strategy_adapter: "ActionSignalGuideAdapter",
        params: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Evaluate parameters on test data.

        Args:
            test_data: Test dataset
            strategy_adapter: Strategy adapter
            params: Parameters to evaluate

        Returns:
            Performance metrics
        """
        # Set parameters
        original_params = strategy_adapter.hyperparameters.copy()
        strategy_adapter.hyperparameters.update(params)

        # Run backtest
        performance = self._backtest_on_data(test_data, strategy_adapter)

        # Restore original parameters
        strategy_adapter.hyperparameters = original_params

        return performance

    def _backtest_on_data(
        self, data: pd.DataFrame, strategy_adapter: "ActionSignalGuideAdapter"
    ) -> Dict[str, Any]:
        """
        Run simplified backtest on data.

        Args:
            data: Data to backtest on
            strategy_adapter: Strategy adapter

        Returns:
            Performance metrics
        """
        returns = []
        current_position = 0

        for i in range(len(data)):
            current_data = data.iloc[: i + 1]
            signal = strategy_adapter.generate_signal(current_data, current_position)

            if signal["action"] == "buy" and current_position <= 0:
                current_position = 1
                entry_price = current_data["close"].iloc[-1]
            elif signal["action"] == "sell" and current_position >= 0:
                current_position = -1
                entry_price = current_data["close"].iloc[-1]
            elif signal["action"] == "hold":
                pass

            # Calculate daily return (simplified)
            if i > 0:
                daily_return = (
                    current_data["close"].iloc[-1] - current_data["close"].iloc[-2]
                ) / current_data["close"].iloc[-2]
                returns.append(daily_return * current_position)

        # Calculate basic metrics
        if returns:
            returns_array = np.array(returns)
            total_return = np.prod(1 + returns_array) - 1
            volatility = np.std(returns_array)
            sharpe_ratio = total_return / volatility if volatility > 0 else 0
        else:
            total_return = 0
            volatility = 0
            sharpe_ratio = 0

        return {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "num_trades": len(
                [r for r in returns if abs(r) > 0.001]
            ),  # Rough trade count
        }

    def _calculate_overall_metrics(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate overall walk-forward metrics.

        Args:
            results: Individual walk-forward results

        Returns:
            Overall performance metrics
        """
        if not results:
            return {}

        test_performances = [r["test_performance"] for r in results]

        # Calculate averages
        avg_total_return = np.mean([p["total_return"] for p in test_performances])
        avg_sharpe = np.mean([p["sharpe_ratio"] for p in test_performances])
        avg_volatility = np.mean([p["volatility"] for p in test_performances])

        # Calculate consistency metrics
        sharpe_ratios = [p["sharpe_ratio"] for p in test_performances]
        sharpe_consistency = (
            np.std(sharpe_ratios) / abs(np.mean(sharpe_ratios))
            if np.mean(sharpe_ratios) != 0
            else float("inf")
        )

        return {
            "average_total_return": avg_total_return,
            "average_sharpe_ratio": avg_sharpe,
            "average_volatility": avg_volatility,
            "sharpe_consistency": sharpe_consistency,
            "num_periods": len(results),
            "positive_periods": sum(
                1 for p in test_performances if p["sharpe_ratio"] > 0
            ),
        }

    def _generate_analysis_summary(
        self, results: List[Dict[str, Any]], overall_metrics: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable analysis summary.

        Args:
            results: Walk-forward results
            overall_metrics: Overall metrics

        Returns:
            Analysis summary text
        """
        summary = f"""
Walk-Forward Analysis Summary:
============================
Total Analysis Periods: {overall_metrics.get('num_periods', 0)}
Positive Periods: {overall_metrics.get('positive_periods', 0)}

Performance Metrics:
- Average Total Return: {overall_metrics.get('average_total_return', 0):.4f}
- Average Sharpe Ratio: {overall_metrics.get('average_sharpe_ratio', 0):.4f}
- Sharpe Consistency: {overall_metrics.get('sharpe_consistency', 0):.4f}

Analysis indicates {'good' if overall_metrics.get('sharpe_consistency', 1) < 0.5 else 'poor'} parameter stability.
"""

        return summary.strip()


class StrategyAdapter(Protocol):
    """Protocol for trading strategy adapters."""

    def generate_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> Dict[str, Any]:
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


class RLPolicyAdapter:
    """Adapter for RL policy (PPO trained model)."""

    def __init__(
        self, model_path: Optional[str] = None, enable_150d_features: bool = False
    ):
        """Initialize with trained model path and 150-dimensional feature support."""
        self.model_path = model_path
        self.model = None
        self.observation_space_shape = None
        self.enable_150d_features = enable_150d_features
        self.feature_engineer = (
            SACv427FeatureEngineer() if enable_150d_features else None
        )
        self.feature_cache = {}  # Cache for computed features
        self.hyperparameters = {
            "learning_rate": 1e-4,
            "batch_size": 64,
            "regularization_strength": 1e-5,
            "dropout_rate": 0.1,
        }
        if model_path:
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(model_path)
                # Get observation space shape dynamically
                self.observation_space_shape = self.model.observation_space.shape[0]
                print(
                    f"Loaded RL model with observation space: {self.observation_space_shape}"
                )
                if self.enable_150d_features:
                    print("150-dimensional feature expansion enabled")
            except Exception as e:
                print(f"Warning: Failed to load RL model from {model_path}: {e}")
                print("Falling back to momentum strategy")
                self.model = None

    def generate_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> Dict[str, Any]:
        """Generate signal using RL policy with comprehensive error handling."""
        try:
            if self.model is None:
                # Fallback: Simple momentum strategy
                return self._momentum_signal(data, current_position)

            # Input validation
            if data.empty:
                print("Warning: Empty data provided, using momentum fallback")
                return self._momentum_signal(data, current_position)

            if len(data) < 2:
                print("Warning: Insufficient data points, using momentum fallback")
                return self._momentum_signal(data, current_position)

            # Enhanced feature processing with 150-dimensional expansion
            if self.enable_150d_features and self.feature_engineer:
                # Generate comprehensive 150+ dimensional features
                cache_key = (
                    f"{len(data)}_{data.index[-1]}" if len(data) > 0 else "empty"
                )
                if cache_key not in self.feature_cache:
                    try:
                        enhanced_data = self.feature_engineer.generate_v427_features(
                            data
                        )
                        self.feature_cache[cache_key] = enhanced_data
                        # Only print feature generation info once per cache miss
                        if len(self.feature_cache) <= 10:  # Limit verbose logging
                            print(
                                f"Generated 150+ dimensional features: {len(enhanced_data.columns)} total columns"
                            )
                    except Exception as feat_e:
                        print(
                            f"Feature engineering failed: {feat_e}, falling back to basic features"
                        )
                        enhanced_data = data
                else:
                    enhanced_data = self.feature_cache[cache_key]

                # Use enhanced features
                numeric_columns = enhanced_data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
            else:
                # Fallback to basic features
                enhanced_data = data
                numeric_columns = data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()

            # Remove timestamp and non-feature columns
            exclude_cols = ["timestamp", "open", "high", "low", "volume"]
            selected_features = [
                col for col in numeric_columns if col not in exclude_cols
            ]

            # Prioritize features with valid values (not all NaN)
            valid_features = []
            for col in selected_features:
                try:
                    if not enhanced_data[col].isna().all():
                        valid_features.append(col)
                except Exception:
                    continue  # Skip problematic columns

            selected_features = valid_features[
                : min(150, len(valid_features))
            ]  # Limit to 150 features max

            if not selected_features:
                print("Warning: No valid features found, using basic price data")
                selected_features = (
                    ["close"] if "close" in enhanced_data.columns else []
                )

            if not selected_features:
                print("Warning: No basic price data available, using momentum fallback")
                return self._momentum_signal(data, current_position)

            # Get the latest data point with error handling
            try:
                obs = (
                    enhanced_data[selected_features].iloc[-1].values.astype(np.float32)
                )
            except (IndexError, KeyError, ValueError) as obs_e:
                print(f"Error extracting observation: {obs_e}, using momentum fallback")
                return self._momentum_signal(data, current_position)

            # Advanced NaN handling: forward fill, then backward fill, then interpolation
            if np.isnan(obs).any():
                print(
                    f"Warning: Found NaN values in {np.sum(np.isnan(obs))} features, applying advanced filling"
                )
                try:
                    # Forward fill then backward fill
                    filled_data = (
                        enhanced_data[selected_features]
                        .fillna(method="ffill")
                        .fillna(method="bfill")
                    )
                    # Interpolate remaining NaN
                    filled_data = filled_data.interpolate(
                        method="linear", limit_direction="both"
                    )
                    obs = filled_data.iloc[-1].values.astype(np.float32)
                    # Final fallback to 0
                    if np.isnan(obs).any():
                        obs = np.nan_to_num(obs, nan=0.0)
                except Exception as nan_e:
                    print(f"NaN handling failed: {nan_e}, using zeros")
                    obs = np.nan_to_num(obs, nan=0.0)

            # Only print feature usage info occasionally to reduce log spam
            if len(self.feature_cache) % 100 == 0:  # Print every 100 cache entries
                print(
                    f"Using {len(obs)}/{len(selected_features)} features for RL model (150d enabled: {self.enable_150d_features})"
                )

            # Adjust to model's expected observation space
            expected_features = self.observation_space_shape or 13
            if len(obs) < expected_features:
                # Pad with zeros
                padding = np.zeros(expected_features - len(obs), dtype=np.float32)
                obs = np.concatenate([obs, padding])
            elif len(obs) > expected_features:
                # Truncate to expected size
                obs = obs[:expected_features]

            print(
                f"Final observation shape: {obs.shape} (expected: {expected_features})"
            )

            # Generate action with model
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                action_value = (
                    float(action[0]) if hasattr(action, "__len__") else float(action)
                )

                # Convert to trading signal
                if action_value > 0.1:
                    signal_action = "buy"
                elif action_value < -0.1:
                    signal_action = "sell"
                else:
                    signal_action = "hold"

                return {
                    "action": signal_action,
                    "confidence": abs(action_value),
                    "raw_action": action_value,
                    "features_used": len(selected_features),
                    "observation_shape": obs.shape,
                }

            except Exception as pred_e:
                print(f"Model prediction failed: {pred_e}, using momentum fallback")
                return self._momentum_signal(data, current_position)

        except Exception as e:
            print(f"Unexpected error in generate_signal: {e}, using momentum fallback")
            return self._momentum_signal(data, current_position)

        # Predict action (using predict_with_masks for MaskablePPO support)
        # Note: No environment available in backtest adapter, so action masks won't be applied
        # TODO: Refactor to pass environment instance for proper action masking
        action, _ = predict_with_masks(self.model, obs, env=None, deterministic=True)

        # Debug: Log the raw action to file
        with open("debug_actions.log", "a") as f:
            f.write(f"Raw action from model: {action}\n")

        # Convert continuous action to discrete action
        # SAC uses continuous actions in range [-1, 1], convert to discrete
        if isinstance(action, (int, np.integer)):
            # Already discrete
            discrete_action = int(action)
        else:
            # Convert continuous to discrete
            action_val = (
                float(action[0]) if hasattr(action, "__len__") else float(action)
            )
            if action_val < -0.33:
                discrete_action = -1  # SELL
            elif action_val > 0.33:
                discrete_action = 1  # BUY
            else:
                discrete_action = 0  # HOLD

        # Debug: Log the discrete action to file
        with open("debug_actions.log", "a") as f:
            f.write(f"Discrete action: {discrete_action}\n")

        # Map action to signal
        action_map = {-1: "sell", 0: "hold", 1: "buy", 2: "sell"}
        return {"action": action_map[discrete_action], "confidence": 0.5}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for backtest (returns DataFrame)."""
        signals = []
        for i in range(len(data)):
            current_data = data.iloc[: i + 1]
            # Assume no position for signal generation
            signal = self.generate_signal(current_data, 0)
            signals.append(signal["action"])

        # Convert actions to signals (-1, 0, 1)
        action_to_signal = {"sell": -1, "hold": 0, "buy": 1}
        signal_values = [action_to_signal.get(s, 0) for s in signals]

        return pd.DataFrame(
            {
                "timestamp": data["timestamp"] if "timestamp" in data else data.index,
                "signal": signal_values,
            }
        )

    def update_hyperparameters(self, hyperparameters: Dict[str, float]) -> None:
        """Update RL model hyperparameters."""
        # Update local hyperparameters
        self.hyperparameters.update(hyperparameters)

        # If model is loaded, update model parameters if supported
        if self.model is not None:
            try:
                # Update learning rate if supported by the model
                if "learning_rate" in hyperparameters and hasattr(
                    self.model, "learning_rate"
                ):
                    # Note: This is a simplified example. Actual implementation would depend on the model type
                    print(
                        f"Updated model learning rate to {hyperparameters['learning_rate']}"
                    )
            except Exception as e:
                print(f"Warning: Could not update model hyperparameters: {e}")

        print(f"Updated strategy hyperparameters: {hyperparameters}")

    def clear_feature_cache(self):
        """Clear feature cache to prevent memory leaks."""
        cache_size = len(self.feature_cache)
        self.feature_cache.clear()
        print(f"Cleared feature cache ({cache_size} entries)")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get feature cache statistics for monitoring."""
        total_entries = len(self.feature_cache)
        memory_usage = (
            sum(df.memory_usage(deep=True).sum() for df in self.feature_cache.values())
            if self.feature_cache
            else 0
        )

        return {
            "cache_entries": total_entries,
            "estimated_memory_mb": memory_usage / (1024 * 1024),
            "features_enabled": self.enable_150d_features,
            "model_loaded": self.model is not None,
        }

    def _momentum_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> Dict[str, Any]:
        """Simple momentum-based signal as RL fallback."""
        if len(data) < 20:
            return {"action": "hold", "confidence": 0.5}

        # Simple RSI-based signal
        close = data["close"]
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()  # type: ignore[operator]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()  # type: ignore[operator]
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        latest_rsi = rsi.iloc[-1] if not rsi.empty else 50

        if latest_rsi < 30 and current_position <= 0:
            return {"action": "buy", "confidence": 0.7}
        elif latest_rsi > 70 and current_position >= 0:
            return {"action": "sell", "confidence": 0.7}
        else:
            return {"action": "hold", "confidence": 0.5}


class SMACrossoverAdapter:
    """Simple Moving Average crossover strategy."""

    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        """Initialize with MA periods."""
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> Dict[str, Any]:
        """Generate SMA crossover signal."""
        if len(data) < self.slow_period:
            return {"action": "hold", "confidence": 0.5}

        close = data["close"]
        fast_ma = close.rolling(self.fast_period).mean()
        slow_ma = close.rolling(self.slow_period).mean()

        # Check for crossover
        prev_fast = fast_ma.iloc[-2] if len(fast_ma) > 1 else fast_ma.iloc[-1]
        prev_slow = slow_ma.iloc[-2] if len(slow_ma) > 1 else slow_ma.iloc[-1]
        curr_fast = fast_ma.iloc[-1]
        curr_slow = slow_ma.iloc[-1]

        # Bullish crossover
        if prev_fast <= prev_slow and curr_fast > curr_slow and current_position <= 0:
            return {"action": "buy", "confidence": 0.8}

        # Bearish crossover
        elif prev_fast >= prev_slow and curr_fast < curr_slow and current_position >= 0:
            return {"action": "sell", "confidence": 0.8}

        return {"action": "hold", "confidence": 0.5}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for backtest (returns DataFrame)."""
        signals = []
        for i in range(len(data)):
            current_data = data.iloc[: i + 1]
            # Assume no position for signal generation
            signal = self.generate_signal(current_data, 0)
            signals.append(signal["action"])

        # Convert actions to signals (-1, 0, 1)
        action_to_signal = {"sell": -1, "hold": 0, "buy": 1}
        signal_values = [action_to_signal.get(s, 0) for s in signals]

        return pd.DataFrame(
            {
                "timestamp": data["timestamp"] if "timestamp" in data else data.index,
                "signal": signal_values,
            }
        )

    def update_hyperparameters(self, hyperparameters: Dict[str, float]) -> None:
        """Update SMA strategy hyperparameters."""
        if "fast_period" in hyperparameters:
            self.fast_period = int(hyperparameters["fast_period"])
        if "slow_period" in hyperparameters:
            self.slow_period = int(hyperparameters["slow_period"])
        print(
            f"Updated SMA parameters: fast_period={self.fast_period}, slow_period={self.slow_period}"
        )

    def clear_feature_cache(self):
        """Clear feature cache to prevent memory leaks."""
        cache_size = len(self.feature_cache)
        self.feature_cache.clear()
        print(f"Cleared feature cache ({cache_size} entries)")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get feature cache statistics for monitoring."""
        total_entries = len(self.feature_cache)
        memory_usage = (
            sum(df.memory_usage(deep=True).sum() for df in self.feature_cache.values())
            if self.feature_cache
            else 0
        )

        return {
            "cache_entries": total_entries,
            "estimated_memory_mb": memory_usage / (1024 * 1024),
            "features_enabled": self.enable_150d_features,
            "model_loaded": self.model is not None,
        }


class BuyAndHoldAdapter:
    """Buy and hold strategy (benchmark)."""

    def __init__(self) -> None:
        """Initialize buy and hold strategy."""
        self.initialized = False

    def generate_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> Dict[str, Any]:
        """Generate buy and hold signal."""
        if not self.initialized and len(data) > 0:
            self.initialized = True
            return {"action": "buy", "confidence": 1.0}

        return {"action": "hold", "confidence": 1.0}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for backtest (returns DataFrame)."""
        signals = []
        for i in range(len(data)):
            current_data = data.iloc[: i + 1]
            # Assume no position for signal generation
            signal = self.generate_signal(current_data, 0)
            signals.append(signal["action"])

        # Convert actions to signals (-1, 0, 1)
        action_to_signal = {"sell": -1, "hold": 0, "buy": 1}
        signal_values = [action_to_signal.get(s, 0) for s in signals]

        return pd.DataFrame(
            {
                "timestamp": data["timestamp"] if "timestamp" in data else data.index,
                "signal": signal_values,
            }
        )

    def clear_feature_cache(self):
        """Clear feature cache to prevent memory leaks."""
        cache_size = len(self.feature_cache)
        self.feature_cache.clear()
        print(f"Cleared feature cache ({cache_size} entries)")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get feature cache statistics for monitoring."""
        total_entries = len(self.feature_cache)
        memory_usage = (
            sum(df.memory_usage(deep=True).sum() for df in self.feature_cache.values())
            if self.feature_cache
            else 0
        )

        return {
            "cache_entries": total_entries,
            "estimated_memory_mb": memory_usage / (1024 * 1024),
            "features_enabled": self.enable_150d_features,
            "model_loaded": self.model is not None,
        }

    def update_hyperparameters(self, hyperparameters: Dict[str, float]) -> None:
        """Update buy and hold strategy hyperparameters."""
        # Buy and hold strategy doesn't have hyperparameters to update
        print("Buy and hold strategy: no hyperparameters to update")


class ActionSignalGuideAdapter:
    """Adapter for Action Signal Guide strategy."""

    def __init__(self, config: Optional[Any] = None):
        """Initialize Action Signal Guide adapter."""
        from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
            ActionSignalGuide,
            ActionSignalGuideConfig,
        )

        self.config = config or {}
        # Always create our own config for ActionSignalGuide - ignore passed config
        # Create config object with debug mode - force minimal setup
        guide_config = ActionSignalGuideConfig(
            debug_short_mode=True,  # Enable debug short mode for faster processing
            short_mode_recognizer_limit=5,  # Limit to 5 recognizers for speed
            guidance_level=ActionSignalGuideConfig().guidance_level,  # Use default guidance level
            enable_parallel_processing=False,  # Disable parallel for stability
            enable_candlestick_patterns=True,
            enable_fibonacci_patterns=False,  # Disable for speed
            enable_gann_patterns=False,  # Disable for speed
            enable_wave_patterns=False,  # Disable for speed
            enable_harmonic_patterns=False,  # Disable for speed
            enable_oscillator_patterns=True,
            enable_volume_patterns=False,  # Disable for speed
            enable_bollinger_patterns=True,
            enable_adx_patterns=False,  # Disable for speed
            enable_granville_patterns=False,  # Disable for speed
            enable_heikin_ashi_patterns=False,  # Disable for speed
            enable_dow_theory_patterns=False,  # Disable for speed
        )
        print(
            f"Created ActionSignalGuideConfig with enable_candlestick_patterns={guide_config.enable_candlestick_patterns}"
        )
        print(
            f"guide_config.candlestick_patterns length: {len(guide_config.candlestick_patterns) if guide_config.candlestick_patterns else 0}"
        )
        self.guide = ActionSignalGuide(config=guide_config)
        print(
            f"ActionSignalGuide initialized with {len(self.guide.all_recognizers)} recognizers"
        )
        print(f"Debug mode: {guide_config.debug_short_mode}")
        self.hyperparameters = {
            "confidence_threshold": 0.6,  # Require 60% confidence for signals
            "signal_strength_threshold": 0.3,  # Require 30% signal strength
            "max_signals_per_bar": 5,
        }

        # Signal statistics tracking
        self.signal_stats = {
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
        }

        # Batch signal cache for efficient backtesting
        self._batch_signals_cache = None

        # Risk management components
        self.risk_manager = RiskManager()
        self.threshold_manager = DynamicThresholdManager()
        self.walk_forward_analyzer = WalkForwardAnalyzer()
        self.active_positions = {}  # Track open positions with stop levels

        # Performance optimization: Caching system
        self.signal_cache = TTLCache(ttl_seconds=300)  # 5 minutes cache for signals
        self.volatility_cache = TTLCache(
            ttl_seconds=60
        )  # 1 minute cache for volatility
        self.thresholds_cache = TTLCache(
            ttl_seconds=120
        )  # 2 minutes cache for thresholds
        self.atr_cache = TTLCache(
            ttl_seconds=180
        )  # 3 minutes cache for ATR calculations

        # Phase 3-1: シグナル品質向上 - 統合フィルタ
        self.integrated_filter = IntegratedSignalFilter()

    def _calculate_dynamic_thresholds(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate dynamic thresholds using advanced threshold manager with caching."""
        # Create cache key
        cache_key = f"thresholds_{len(data)}_{hash(str(data.index[-1]) if len(data) > 0 else 'empty')}"

        # Check cache first
        cached_result = self.thresholds_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Use the advanced threshold manager for better adaptation
        thresholds = self.threshold_manager.calculate_adaptive_thresholds(data)

        result = {
            "confidence_threshold": thresholds["confidence_threshold"],
            "signal_strength_threshold": thresholds["signal_strength_threshold"],
        }

        # Cache the result
        self.thresholds_cache.set(cache_key, result)
        return result

    def _calculate_market_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current market volatility for risk management with caching."""
        # Create cache key based on data characteristics
        cache_key = f"volatility_{len(data)}_{hash(str(data.index[-1]) if len(data) > 0 else 'empty')}"

        # Check cache first
        cached_result = self.volatility_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Calculate volatility
        if len(data) < 20:
            result = 0.02  # Default moderate volatility
        else:
            returns = data["close"].pct_change()
            volatility_series = returns.rolling(20).std()
            if len(volatility_series) > 0:
                last_vol = volatility_series.iloc[-1]
                if not np.isnan(last_vol):
                    result = last_vol.item()
                else:
                    result = 0.02
            else:
                result = 0.02

        # Cache the result
        self.volatility_cache.set(cache_key, result)
        return result

    def update_positions(
        self, current_price: float, current_time: pd.Timestamp
    ) -> List[Dict[str, Any]]:
        """
        Update open positions and check for stop loss/take profit triggers.

        Args:
            current_price: Current market price
            current_time: Current timestamp

        Returns:
            List of closed positions with results
        """
        closed_positions = []

        for position_id, position_data in list(self.active_positions.items()):
            should_close, reason = self.risk_manager.should_close_position(
                position_data, current_price, self.risk_manager.portfolio_value
            )

            if should_close:
                # Calculate P&L
                entry_price = position_data["entry_price"]
                position_type = position_data["type"]
                position_size = position_data["size"]

                if position_type == "long":
                    pnl = (current_price - entry_price) * position_size
                else:  # short
                    pnl = (entry_price - current_price) * position_size

                # Update risk manager
                self.risk_manager.update_risk_metrics({"pnl": pnl})

                # Create closed position record
                closed_position = {
                    "position_id": position_id,
                    "type": position_type,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "size": position_size,
                    "pnl": pnl,
                    "reason": reason,
                    "entry_time": position_data["entry_time"],
                    "exit_time": current_time,
                    "duration": current_time - position_data["entry_time"],
                }

                closed_positions.append(closed_position)
                del self.active_positions[position_id]

        return closed_positions

    def open_position(
        self,
        position_type: str,
        entry_price: float,
        position_size: float,
        stop_loss: float,
        take_profit: float,
        current_time: pd.Timestamp,
        signal_data: Dict[str, Any],
    ) -> str:
        """
        Open a new position with risk management parameters.

        Args:
            position_type: 'long' or 'short'
            entry_price: Entry price
            position_size: Position size as fraction of portfolio
            stop_loss: Stop loss price
            take_profit: Take profit price
            current_time: Entry timestamp
            signal_data: Signal information for tracking

        Returns:
            Position ID
        """
        position_id = f"{position_type}_{current_time.strftime('%Y%m%d_%H%M%S')}_{len(self.active_positions)}"

        position_data = {
            "type": position_type,
            "entry_price": entry_price,
            "size": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": current_time,
            "signal_data": signal_data,
        }

        self.active_positions[position_id] = position_data
        return position_id

    def generate_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> Dict[str, Any]:
        """Generate signal using Action Signal Guide with dynamic thresholds and risk management."""
        try:
            # Calculate dynamic thresholds based on current market conditions
            dynamic_thresholds = self._calculate_dynamic_thresholds(data)
            current_confidence_threshold = dynamic_thresholds["confidence_threshold"]
            current_strength_threshold = dynamic_thresholds["signal_strength_threshold"]

            # Calculate market volatility for risk management
            market_volatility = self._calculate_market_volatility(data)

            # Generate signals from Action Signal Guide
            # Use the last index of current data (current bar)
            current_index = len(data) - 1
            signals = self.guide.generate_signals(data, current_index)

            # Debug: print signal information (only for significant signals and not too frequently)
            if signals and current_index % 500 == 0:  # Print only every 500 bars
                print(f"Generated {len(signals)} signals at index {current_index}")
                print(
                    f"Dynamic thresholds: confidence={current_confidence_threshold:.3f}, strength={current_strength_threshold:.3f}"
                )
                print(f"Market volatility: {market_volatility:.3f}")
                for i, signal in enumerate(signals[:3]):  # Show only first 3 signals
                    print(
                        f"  Signal {i}: direction={signal.direction:.3f}, confidence={signal.confidence:.3f}, type={signal.signal_type}"
                    )

            if not signals:
                self.signal_stats["hold_signals"] += 1
                self.signal_stats["total_signals"] += 1
                return {"action": "hold"}  # Hold

            # Use the most recent signal (last in the list)
            latest_signal = signals[-1]

            # Convert ActionSignal to action
            # ActionSignal.direction: -1.0 (strong sell) to 1.0 (strong buy)
            # Convert to discrete actions: "sell", "hold", "buy"
            direction = latest_signal.direction
            confidence = latest_signal.confidence

            # Apply dynamic confidence threshold
            if confidence < current_confidence_threshold:
                self.signal_stats["hold_signals"] += 1
                self.signal_stats["total_signals"] += 1
                return {"action": "hold"}  # Hold if confidence too low

            # Apply dynamic signal strength threshold
            if abs(direction) < current_strength_threshold:
                self.signal_stats["hold_signals"] += 1
                self.signal_stats["total_signals"] += 1
                return {"action": "hold"}  # Hold if signal too weak

            # Risk management: Check if position should be opened
            signal_strength = abs(direction) * confidence  # Combined signal strength
            # Get current price safely
            try:
                if "close" in data.columns:
                    current_price = data["close"].iloc[-1]
                elif "Close" in data.columns:
                    current_price = data["Close"].iloc[-1]
                else:
                    current_price = data.iloc[-1, -1]  # Last column as fallback

                # Safe float conversion
                try:
                    if hasattr(current_price, "item"):  # numpy scalar
                        current_price = current_price.item()
                    current_price = float(current_price)
                except (ValueError, TypeError, AttributeError):
                    current_price = 0.0
            except (ValueError, TypeError, KeyError, AttributeError):
                current_price = 0.0  # Fallback price

            if direction > 0.0:  # Bullish signal
                action = "buy"
                position_type = "long"
                self.signal_stats["buy_signals"] += 1
            elif direction < 0.0:  # Bearish signal
                action = "sell"
                position_type = "short"
                self.signal_stats["sell_signals"] += 1
            else:  # Neutral
                action = "hold"
                self.signal_stats["hold_signals"] += 1
                self.signal_stats["total_signals"] += 1
                return {"action": "hold"}

            # Risk management: Validate position opening
            if not self.risk_manager.should_open_position(
                signal_strength, market_volatility, self.risk_manager.portfolio_value
            ):
                self.signal_stats[
                    "hold_signals"
                ] += 1  # Override to hold due to risk management
                self.signal_stats["total_signals"] += 1
                return {
                    "action": "hold",
                    "risk_filtered": True,
                    "reason": "risk_management_blocked",
                }

            # Phase 3-1: シグナル品質向上 - 統合フィルタ適用
            signal_data = {
                "action": action,
                "confidence": confidence,
                "direction": direction,
                "signal_type": latest_signal.signal_type,
                "description": latest_signal.description,
                "timestamp": data.index[-1]
                if hasattr(data.index[-1], "timestamp")
                else pd.Timestamp.now(),
                "signal_strength": signal_strength,
                "market_volatility": market_volatility,
            }

            # 統合フィルタでシグナル品質を評価
            filter_result = self.integrated_filter.evaluate_signal_quality(
                signal_data, data
            )

            # フィルタ結果に基づいて最終決定
            if not filter_result.should_accept:
                self.signal_stats["hold_signals"] += 1
                self.signal_stats["total_signals"] += 1
                return {
                    "action": "hold",
                    "quality_filtered": True,
                    "filter_reasons": filter_result.filter_reasons,
                    "quality_score": filter_result.quality_score,
                    "recommended_action": filter_result.recommended_action,
                }

            # フィルタ通過時の品質情報を追加
            signal_data.update(
                {
                    "quality_score": filter_result.quality_score,
                    "quality_level": filter_result.overall_quality.value,
                    "filter_passed": True,
                }
            )

            # Calculate position size and stop levels
            position_size = self.risk_manager.get_risk_adjusted_position_size(
                signal_strength, market_volatility
            )
            stop_loss, take_profit = self.risk_manager.calculate_atr_stop_levels(
                data, current_price, position_type
            )

            self.signal_stats["total_signals"] += 1

            return {
                "action": action,
                "confidence": confidence,
                "direction": direction,
                "signal_type": latest_signal.signal_type,
                "description": latest_signal.description,
                "dynamic_thresholds": dynamic_thresholds,
                "position_size": position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_adjusted": True,
                "quality_score": filter_result.quality_score,
                "quality_level": filter_result.overall_quality.value,
                "filter_passed": True,
            }

        except Exception as e:
            # Enhanced error handling with classification and logging
            error_type = type(e).__name__
            error_msg = str(e)

            # Classify error types for better handling
            if "insufficient" in error_msg.lower() or "length" in error_msg.lower():
                error_category = "data_insufficient"
                print(
                    f"DEBUG: Data insufficient for signal generation at index {len(data)-1}: {error_msg}"
                )
            elif "validation" in error_msg.lower():
                error_category = "validation_error"
                print(f"WARNING: Signal validation failed: {error_msg}")
            elif "timeout" in error_msg.lower() or "time" in error_msg.lower():
                error_category = "timeout_error"
                print(f"WARNING: Signal generation timeout: {error_msg}")
            elif "memory" in error_msg.lower():
                error_category = "memory_error"
                print(f"ERROR: Memory error during signal generation: {error_msg}")
            else:
                error_category = "unexpected_error"
                print(
                    f"ERROR: Unexpected error in signal generation ({error_type}): {error_msg}"
                )

            # Update error statistics
            if not hasattr(self, "error_stats"):
                self.error_stats = {}
            self.error_stats[error_category] = (
                self.error_stats.get(error_category, 0) + 1
            )

            # Fallback: return hold signal with error metadata
            self.signal_stats["hold_signals"] += 1
            self.signal_stats["total_signals"] += 1
            return {
                "action": "hold",
                "error": True,
                "error_category": error_category,
                "error_message": error_msg[:100],  # Truncate long messages
            }

    def generate_signals_batch(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate signals for entire dataset in batch mode for efficient backtesting.

        This method pre-computes signals for all data points to avoid repeated
        individual signal generation calls during backtesting.

        Args:
            data: OHLCV DataFrame with all historical data

        Returns:
            List of signal dictionaries for each data point
        """
        if (
            not hasattr(self, "_batch_signals_cache")
            or self._batch_signals_cache is None
        ):
            print(
                f"Pre-computing signals for {len(data)} data points using optimized batch processing..."
            )

            start_time = time.time()
            self._batch_signals_cache = []

            # Use optimized batch processing that leverages ActionSignalGuide's efficiency
            self._generate_signals_optimized(data)

            processing_time = time.time() - start_time
            print(f"Batch signal generation completed in {processing_time:.2f} seconds")
            print(f"Generated {len(self._batch_signals_cache)} signals")

        return self._batch_signals_cache

    def _generate_signals_optimized(self, data: pd.DataFrame) -> None:
        """Generate signals using optimized batch processing."""
        self._batch_signals_cache = []

        # For backtesting efficiency, we'll generate signals for each point
        # but use a sliding window approach to minimize redundant calculations
        window_size = 100  # Process in windows to balance memory and speed

        for start_idx in range(0, len(data), window_size):
            end_idx = min(start_idx + window_size, len(data))

            # Process each point in the current window
            for i in range(start_idx, end_idx):
                try:
                    # Use the guide's generate_signals method directly for efficiency
                    current_data = data.iloc[: i + 1].copy()
                    if isinstance(current_data, pd.Series):
                        current_data = current_data.to_frame().T
                    signals = self.guide.generate_signals(current_data, i)

                    if signals:
                        # Convert the most recent signal to action format
                        latest_signal = signals[-1]
                        action = self._convert_signal_to_action(latest_signal)
                        self._batch_signals_cache.append(action)

                        # Update signal statistics
                        if action["action"] == "buy":
                            self.signal_stats["buy_signals"] += 1
                        elif action["action"] == "sell":
                            self.signal_stats["sell_signals"] += 1
                        else:
                            self.signal_stats["hold_signals"] += 1
                        self.signal_stats["total_signals"] += 1
                    else:
                        self._batch_signals_cache.append({"action": "hold"})
                        self.signal_stats["hold_signals"] += 1
                        self.signal_stats["total_signals"] += 1

                except Exception as e:
                    print(f"Error generating signal at index {i}: {e}")
                    self._batch_signals_cache.append({"action": "hold"})

    def _convert_signal_to_action(self, signal) -> Dict[str, Any]:
        """Convert ActionSignal to action dictionary format."""
        # Convert ActionSignal.direction: -1.0 (strong sell) to 1.0 (strong buy)
        direction = signal.direction
        confidence = signal.confidence

        # Use more conservative thresholds for backtesting
        confidence_threshold = 0.6  # Require 60% confidence
        strength_threshold = 0.3  # Require 30% signal strength

        # Apply confidence threshold
        if confidence < confidence_threshold:
            return {"action": "hold"}

        # Apply signal strength threshold
        if abs(direction) < strength_threshold:
            return {"action": "hold"}

        if direction > 0.0:  # Bullish
            action = "buy"
        elif direction < 0.0:  # Bearish
            action = "sell"
        else:  # Neutral
            action = "hold"

        return {
            "action": action,
            "confidence": confidence,
            "direction": direction,
            "signal_type": signal.signal_type,
            "description": signal.description,
        }

    def update_hyperparameters(self, hyperparameters: Dict[str, float]) -> None:
        """Update strategy hyperparameters."""
        self.hyperparameters.update(hyperparameters)
        print(f"Updated Action Signal Guide hyperparameters: {self.hyperparameters}")

    def get_signal_statistics(self) -> Dict[str, int]:
        """Get signal generation statistics."""
        return self.signal_stats.copy()


def create_adapter(strategy_name: str, **kwargs: Any) -> StrategyAdapter:
    """Factory function to create strategy adapters."""

    if strategy_name == "rl":
        return RLPolicyAdapter(**kwargs)
    elif strategy_name == "sma_fast_slow":
        return SMACrossoverAdapter(**kwargs)
    elif strategy_name == "buy_hold":
        return BuyAndHoldAdapter(**kwargs)
    elif strategy_name == "action_signal_guide":
        return ActionSignalGuideAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
