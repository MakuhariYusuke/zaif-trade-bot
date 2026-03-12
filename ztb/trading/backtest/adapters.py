"""
Strategy adapters for backtesting.

Provides adapters to wrap different trading strategies for unified backtest interface.
"""

import time
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ztb.analysis.walk_forward_analyzer import WalkForwardAnalyzer
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.threshold_manager import ThresholdManager
from ztb.trading.environment.constants import BYTES_PER_MB
from ztb.trading.risk.backtest_risk_manager import BacktestRiskManager
from ztb.trading.risk.optimizers.integrated_signal_filter import IntegratedSignalFilter
from ztb.utils.cache_utils import TTLCache

class StrategyAdapter(Protocol):
    """Protocol for trading strategy adapters."""

    def generate_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> dict[str, Any]:
        """
        Generate trading signal.

        Args:
            data: Market data with OHLCV and features
            current_position: Current position (-1, 0, 1 for short, flat, long)

        Returns:
            Signal dict with 'action' and optional parameters
        """
        ...

    def update_hyperparameters(self, hyperparameters: dict[str, float]) -> None:
        """
        Update strategy hyperparameters.

        Args:
            hyperparameters: Dictionary of hyperparameter names and values
        """
        ...

class RLPolicyAdapter:
    """Adapter for RL policy (PPO trained model)."""

    def __init__(
        self, model_path: str | None = None, enable_150d_features: bool = False
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
    ) -> dict[str, Any]:
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
                discrete_action = ACTION_SELL  # SELL
            elif action_val > 0.33:
                discrete_action = ACTION_BUY  # BUY
            else:
                discrete_action = ACTION_HOLD  # HOLD

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
        action_to_signal = {"sell": ACTION_SELL, "hold": ACTION_HOLD, "buy": ACTION_BUY}
        signal_values = [action_to_signal.get(s, 0) for s in signals]

        return pd.DataFrame(
            {
                "timestamp": data["timestamp"] if "timestamp" in data else data.index,
                "signal": signal_values,
            }
        )

    def update_hyperparameters(self, hyperparameters: dict[str, float]) -> None:
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

    def get_cache_stats(self) -> dict[str, Any]:
        """Get feature cache statistics for monitoring."""
        total_entries = len(self.feature_cache)
        memory_usage = (
            sum(df.memory_usage(deep=True).sum() for df in self.feature_cache.values())
            if self.feature_cache
            else 0
        )

        return {
            "cache_entries": total_entries,
            "estimated_memory_mb": memory_usage / BYTES_PER_MB,
            "features_enabled": self.enable_150d_features,
            "model_loaded": self.model is not None,
        }

    def _momentum_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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

    def update_hyperparameters(self, hyperparameters: dict[str, float]) -> None:
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

    def get_cache_stats(self) -> dict[str, Any]:
        """Get feature cache statistics for monitoring."""
        total_entries = len(self.feature_cache)
        memory_usage = (
            sum(df.memory_usage(deep=True).sum() for df in self.feature_cache.values())
            if self.feature_cache
            else 0
        )

        return {
            "cache_entries": total_entries,
            "estimated_memory_mb": memory_usage / BYTES_PER_MB,
            "features_enabled": self.enable_150d_features,
        }

class BuyAndHoldAdapter:
    """Buy and hold strategy (benchmark)."""

    def __init__(self) -> None:
        """Initialize buy and hold strategy."""
        self.initialized = False

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy and hold signals."""
        # Buy and hold: always buy at the beginning, hold forever
        signals = []
        for i, current_data in data.iterrows():
            if not self.initialized:
                # Buy on first data point
                signals.append(ACTION_BUY)
                self.initialized = True
            else:
                # Hold
                signals.append(ACTION_HOLD)

        return pd.DataFrame(
            {
                "timestamp": data["timestamp"] if "timestamp" in data else data.index,
                "signal": signals,
            }
        )

    def update_hyperparameters(self, hyperparameters: dict[str, float]) -> None:
        """Update buy and hold strategy hyperparameters."""
        # Buy and hold strategy doesn't have hyperparameters to update
        print("Buy and hold strategy: no hyperparameters to update")

class ActionSignalGuideAdapter:
    """Adapter for Action Signal Guide strategy."""

    def __init__(self, config: Any | None = None):
        """Initialize Action Signal Guide adapter."""
        from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
            ActionSignalGuideConfig,
        )

        self.config = config or {}
        # Respect a provided ActionSignalGuideConfig or dict, otherwise create a debug-light config
        if isinstance(self.config, ActionSignalGuideConfig):
            guide_config = self.config
        elif isinstance(self.config, dict):
            # Build a config object from the provided dict while preserving defaults
            base = ActionSignalGuideConfig()
            for k, v in self.config.items():
                if hasattr(base, k):
                    setattr(base, k, v)
            guide_config = base
            guide_config.short_mode_recognizer_limit = min(
                guide_config.short_mode_recognizer_limit, 5
            )
        else:
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
                signal_strength_threshold=0.3,  # Require 30% signal strength
                max_signals_per_bar=5,
                force_accept_signals=False,  # If True, bypass filters and accept valid signals
            )

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
        self.risk_manager = BacktestRiskManager({"test_mode": True})
        # When running inside backtests we want to relax risk checks for diagnostic runs;
        # enable test_mode by default for this adapter to avoid aggressive gating.
        try:
            self.risk_manager.test_mode = True
        except Exception:
            pass

        # Initialize ThresholdManager with default config for signal management
        threshold_config = type(
            "ThresholdConfig",
            (),
            {
                "continuous_to_discrete_threshold": 0.01,
                "adaptive_threshold_mode": True,
                "threshold_volatility_multiplier": 1.0,
                "min_action_threshold": 0.001,
                "max_action_threshold": 1.0,
                "regime_detection_window": 50,
                "threshold_adaptation_rate": 0.1,
                "performance_memory_size": 100,
                "trend_detection_threshold": 0.001,
                "volatility_detection_threshold": 0.02,
            },
        )()
        self.threshold_manager = ThresholdManager(threshold_config)

        # Hyperparameters for dynamic threshold calculation
        self.hyperparameters: dict[str, float] = {
            "confidence_threshold": 0.6,
            "signal_strength_threshold": 0.3,
        }

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

        # Backtest-time: 緩和された基準を使用してより多くのシグナルを検証できるようにする
        # - permissive_mode にして初期評価を緩くする
        # - バックテストでは初期閾値を下げて実際の取引生成を確認する
        try:
            self.integrated_filter.filter_criteria.permissive_mode = True
            self.integrated_filter.filter_criteria.min_quality_score = 0.45
            self.integrated_filter.filter_criteria.min_confidence_score = 0.55
            # Keep adaptive filtering ON so it can revert later if needed
            self.integrated_filter.filter_criteria.adaptive_filtering = True
        except Exception:
            # If for some reason the filter object doesn't have the expected attributes,
            # we just continue with defaults to avoid crashing the backtest
            pass

    def _calculate_dynamic_thresholds(self, data: pd.DataFrame) -> dict[str, float]:
        """Calculate dynamic thresholds using advanced threshold manager with caching."""
        # Create cache key
        cache_key = f"thresholds_{len(data)}_{hash(str(data.index[-1]) if len(data) > 0 else 'empty')}"

        # Check cache first
        cached_result = self.thresholds_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Use the advanced threshold manager for better adaptation
        # Phase 3: Lower base thresholds to allow more signals while maintaining quality
        # Use hyperparameters override when present so experiments/backtests can set
        # desired baselines (e.g. for more permissive testing).
        base_confidence = self.hyperparameters.get("confidence_threshold", 0.6)
        base_strength = self.hyperparameters.get("signal_strength_threshold", 0.3)
        # If in backtest/test mode, allow tests to override computed thresholds
        # using hyperparameters (safer for diagnostics) to avoid overly strict
        # dynamic thresholds during controlled debugging runs.
        if getattr(self, "risk_manager", None) is not None and getattr(
            self.risk_manager, "test_mode", False
        ):
            thresholds = {
                "confidence_threshold": base_confidence,
                "signal_strength_threshold": base_strength,
            }
            try:
                print(
                    f"[DEBUG-ADAPTER-THRESHOLDS] Using test_mode override thresholds: confidence={base_confidence:.3f}, strength={base_strength:.3f}"
                )
            except Exception:
                pass
        else:
            thresholds = self.threshold_manager.calculate_adaptive_signal_thresholds(
                data, base_confidence=base_confidence, base_strength=base_strength
            )

        result = {
            "confidence_threshold": thresholds["confidence_threshold"],
            "signal_strength_threshold": thresholds["signal_strength_threshold"],
        }

        # Cache the result
        self.thresholds_cache.set(cache_key, result)
        try:
            print(
                f"[DEBUG-ADAPTER-THRESHOLDS] base_confidence={base_confidence:.3f}, base_strength={base_strength:.3f}, thresholds={{'confidence':{result['confidence_threshold']:.3f}, 'strength':{result['signal_strength_threshold']:.3f}}}"
            )
        except Exception:
            pass
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
    ) -> list[dict[str, Any]]:
        """
        Update open positions and check for stop loss/take profit triggers.

        Args:
            current_price: Current market price
            current_time: Current timestamp

        Returns:
            list of closed positions with results
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
        signal_data: dict[str, Any],
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

    def _log_and_return(
        self, result: dict[str, Any], tag: str = "ADAPTER-RET"
    ) -> dict[str, Any]:
        """Log the adapter return and return the provided result.

        This is a lightweight helper allowing consistent debug prints for the
        numerous return locations inside signal generation.
        """
        try:
            print(f"[DEBUG-ADAPTER-RETURN] {tag}: {result}")
        except Exception:
            # Don't let logging cause a secondary failure
            pass
        return result

    def generate_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> dict[str, Any]:
        """Generate signal using Action Signal Guide with dynamic thresholds and risk management."""
        try:
            # Debug: ensure adapter generate_signal is being called
            print(f"[DEBUG-ADAPTER-GEN] Called generate_signal for index {len(data)-1}")
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
            print(
                f"[DEBUG-ADAPTER-SIGCNT] idx={current_index} signals_count={len(signals) if signals is not None else 0}"
            )

            # Debug: log any non-neutral signals to trace behavior
            if signals:
                try:
                    latest = signals[-1]
                    print(
                        f"[DEBUG-ADAPTER] {current_index} signals={len(signals)} latest: direction={latest.direction:.3f}, confidence={latest.confidence:.3f}, strength={abs(latest.direction):.3f}"
                    )
                except Exception:
                    print(
                        f"[DEBUG-ADAPTER] {current_index} signals exist but could not read details"
                    )

            # Debug: print signal information (only for significant signals and not too frequently)
            if current_index % 100 == 0:  # Print every 100 bars for debugging
                print(f"[{current_index}] Generated {len(signals)} signals")
                if signals:
                    print(
                        f"  First signal: direction={signals[0].direction:.3f}, confidence={signals[0].confidence:.3f}"
                    )
                print(
                    f"  Dynamic thresholds: confidence={current_confidence_threshold:.3f}, strength={current_strength_threshold:.3f}"
                )
                print(f"  Market volatility: {market_volatility:.3f}")

            if not signals:
                self.signal_stats["hold_signals"] += 1
                self.signal_stats["total_signals"] += 1
                return self._log_and_return({"action": "hold"}, "no_signals")  # Hold

            # Use the most recent signal (last in the list)
            latest_signal = signals[-1]

            # Convert ActionSignal to action
            # ActionSignal.direction: -1.0 (strong sell) to 1.0 (strong buy)
            # Convert to discrete actions: "sell", "hold", "buy"
            direction = latest_signal.direction
            confidence = latest_signal.confidence

            print(
                f"[DEBUG] Latest signal: direction={direction:.3f}, confidence={confidence:.3f}, strength={abs(direction):.3f}"
            )  # Debug output
            print(
                f"[DEBUG] Thresholds: confidence_threshold={current_confidence_threshold:.3f}, strength_threshold={current_strength_threshold:.3f}"
            )  # Debug output

            # Apply dynamic confidence threshold
            if confidence < current_confidence_threshold:
                print(
                    f"[DEBUG] Signal filtered by confidence threshold: {confidence:.3f} < {current_confidence_threshold:.3f}"
                )  # Debug output
                self.signal_stats["hold_signals"] += 1
                self.signal_stats["total_signals"] += 1
                return self._log_and_return(
                    {"action": "hold"}, "confidence_filtered"
                )  # Hold if confidence too low

            # Apply dynamic signal strength threshold
            if abs(direction) < current_strength_threshold:
                print(
                    f"[DEBUG] Signal filtered by strength threshold: {abs(direction):.3f} < {current_strength_threshold:.3f}"
                )  # Debug output
                self.signal_stats["hold_signals"] += 1
                self.signal_stats["total_signals"] += 1
                return self._log_and_return(
                    {"action": "hold"}, "strength_filtered"
                )  # Hold if signal too weak

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
                return self._log_and_return({"action": "hold"}, "neutral")

            # Risk management: Validate position opening
            current_portfolio_value = getattr(self.risk_manager, "portfolio_value", 1.0)
            # If forcing acceptance, skip risk checks
            if self.hyperparameters.get("force_accept_signals"):
                should_open = True
            else:
                should_open = True
                try:
                    should_open = self.risk_manager.should_open_position(
                        signal_strength, market_volatility, current_portfolio_value
                    )
                except AttributeError as e:
                    print(
                        f"[DEBUG-ADAPTER-RISK] Risk manager missing attribute in should_open_position: {e}"
                    )
                    should_open = getattr(self.risk_manager, "test_mode", False)
                except Exception as e:
                    print(
                        f"[DEBUG-ADAPTER-RISK] Unexpected error calling should_open_position: {e}"
                    )
                    should_open = getattr(self.risk_manager, "test_mode", False)
            if not should_open:
                self.signal_stats[
                    "hold_signals"
                ] += 1  # Override to hold due to risk management
                self.signal_stats["total_signals"] += 1
                return self._log_and_return(
                    {
                        "action": "hold",
                        "risk_filtered": True,
                        "reason": "risk_management_blocked",
                    },
                    "risk_filtered",
                )

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

            print(
                f"[DEBUG] Integrated filter result: should_accept={filter_result.should_accept}, quality_score={filter_result.quality_score:.3f}"
            )  # Debug output
            if not filter_result.should_accept:
                print(
                    f"[DEBUG] Signal filtered by integrated filter: reasons={filter_result.filter_reasons}"
                )  # Debug output

            # フィルタ結果に基づいて最終決定
            if not filter_result.should_accept:
                self.signal_stats["hold_signals"] += 1
                self.signal_stats["total_signals"] += 1
                return self._log_and_return(
                    {
                        "action": "hold",
                        "quality_filtered": True,
                        "filter_reasons": filter_result.filter_reasons,
                        "quality_score": filter_result.quality_score,
                        "recommended_action": filter_result.recommended_action,
                    },
                    "quality_filtered",
                )

            # フィルタ通過時の品質情報を追加
            signal_data.update(
                {
                    "quality_score": filter_result.quality_score,
                    "quality_level": filter_result.overall_quality.value,
                    "filter_passed": True,
                }
            )

            # Calculate position size and stop levels
            # Defensive call: some RiskManager implementations do not expose
            # get_risk_adjusted_position_size (e.g. repository-level RiskManager).
            # Use fallback sizing in that case to keep backtests running.
            if self.hyperparameters.get("force_accept_signals"):
                position_size = 0.05
            elif hasattr(self.risk_manager, "get_risk_adjusted_position_size"):
                try:
                    position_size = self.risk_manager.get_risk_adjusted_position_size(
                        signal_strength, market_volatility
                    )
                except AttributeError as e:
                    print(
                        f"[DEBUG-ADAPTER-RISK] Risk manager method raised AttributeError in get_risk_adjusted_position_size: {e}"
                    )
                    base_max = getattr(self.risk_manager, "max_position_size", 0.1)
                    position_size = min(
                        base_max, max(0.01, signal_strength * (1.0 - market_volatility))
                    )
                except Exception as e:
                    print(
                        f"[DEBUG-ADAPTER-RISK] Unexpected error calling get_risk_adjusted_position_size: {e}"
                    )
                    base_max = getattr(self.risk_manager, "max_position_size", 0.1)
                    position_size = min(
                        base_max, max(0.01, signal_strength * (1.0 - market_volatility))
                    )
            else:
                # Fallback: basic percentage sizing based on signal strength and volatility
                base_max = getattr(self.risk_manager, "max_position_size", 0.1)
                try:
                    position_size = min(
                        base_max, max(0.01, signal_strength * (1.0 - market_volatility))
                    )
                except Exception:
                    position_size = 0.01
            if self.hyperparameters.get("force_accept_signals"):
                # Provide generous stop/take profit when forcing acceptance
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.02
            elif hasattr(self.risk_manager, "calculate_atr_stop_levels"):
                try:
                    (
                        stop_loss,
                        take_profit,
                    ) = self.risk_manager.calculate_atr_stop_levels(
                        data, current_price, position_type
                    )
                except AttributeError as e:
                    print(
                        f"[DEBUG-ADAPTER-RISK] Risk manager method raised AttributeError in calculate_atr_stop_levels: {e}"
                    )
                    stop_pct = 0.02
                    if position_type == "long":
                        stop_loss = current_price * (1 - stop_pct)
                        take_profit = current_price * (1 + stop_pct * 2)
                    else:
                        stop_loss = current_price * (1 + stop_pct)
                        take_profit = current_price * (1 - stop_pct * 2)
                except Exception as e:
                    print(
                        f"[DEBUG-ADAPTER-RISK] Unexpected error calling calculate_atr_stop_levels: {e}"
                    )
                    stop_pct = 0.02
                    if position_type == "long":
                        stop_loss = current_price * (1 - stop_pct)
                        take_profit = current_price * (1 + stop_pct * 2)
                    else:
                        stop_loss = current_price * (1 + stop_pct)
                        take_profit = current_price * (1 - stop_pct * 2)
            else:
                # Fallback: simple percentage-based stop/take levels
                stop_pct = 0.02
                if position_type == "long":
                    stop_loss = current_price * (1 - stop_pct)
                    take_profit = current_price * (1 + stop_pct * 2)
                else:
                    stop_loss = current_price * (1 + stop_pct)
                    take_profit = current_price * (1 - stop_pct * 2)

            self.signal_stats["total_signals"] += 1

            return self._log_and_return(
                {
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
                },
                "accepted",
            )

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
            return self._log_and_return(
                {
                    "action": "hold",
                    "error": True,
                    "error_category": error_category,
                    "error_message": error_msg[:100],  # Truncate long messages
                },
                "error",
            )

    def generate_signals_batch(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Generate signals for entire dataset in batch mode for efficient backtesting.

        This method pre-computes signals for all data points to avoid repeated
        individual signal generation calls during backtesting.

        Args:
            data: OHLCV DataFrame with all historical data

        Returns:
            list of signal dictionaries for each data point
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

    def _convert_signal_to_action(self, signal) -> dict[str, Any]:
        """Convert ActionSignal to action dictionary format."""
        # Convert ActionSignal.direction: -1.0 (strong sell) to 1.0 (strong buy)
        direction = signal.direction
        confidence = signal.confidence

        # Use configured hyperparameters for thresholds (defaults kept as fallback)
        confidence_threshold = self.hyperparameters.get("confidence_threshold", 0.6)
        strength_threshold = self.hyperparameters.get("signal_strength_threshold", 0.3)

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

    def update_hyperparameters(self, hyperparameters: dict[str, float]) -> None:
        """Update strategy hyperparameters."""
        self.hyperparameters.update(hyperparameters)
        print(f"Updated Action Signal Guide hyperparameters: {self.hyperparameters}")

    def get_signal_statistics(self) -> dict[str, int]:
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
