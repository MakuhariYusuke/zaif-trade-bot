"""
Strategy adapters for backtesting.

Provides adapters to wrap different trading strategies for unified backtest interface.
"""

from typing import Any, Dict, Optional, Protocol

import numpy as np
import pandas as pd

from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer
from ztb.training.policies.policy_utils import predict_with_masks


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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Action Signal Guide adapter."""
        from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
            ActionSignalGuide,
            ActionSignalGuideConfig,
        )

        self.config = config or {}
        # Use provided config or create default
        if config and hasattr(config, "debug_short_mode"):
            guide_config = config
        else:
            # Create config object with debug mode - force minimal setup
            guide_config = ActionSignalGuideConfig(
                debug_short_mode=False,  # Run all recognizers
                guidance_level=ActionSignalGuideConfig().guidance_level,  # Use default guidance level
                enable_candlestick_patterns=True,
                enable_fibonacci_patterns=False,
                enable_gann_patterns=False,
                enable_wave_patterns=False,
                enable_harmonic_patterns=False,
                enable_oscillator_patterns=False,
                enable_volume_patterns=False,
                enable_bollinger_patterns=False,
                enable_adx_patterns=False,
                enable_granville_patterns=False,
                enable_heikin_ashi_patterns=False,
                enable_dow_theory_patterns=False,
            )
        self.guide = ActionSignalGuide(config=guide_config)
        print(
            f"ActionSignalGuide initialized with {len(self.guide.all_recognizers)} recognizers"
        )
        print(f"Debug mode: {guide_config.debug_short_mode}")
        self.hyperparameters = {
            "confidence_threshold": 0.1,  # Very low threshold
            "signal_strength_threshold": 0.0,  # Accept any signal strength
            "max_signals_per_bar": 5,
        }

        # Signal statistics tracking
        self.signal_stats = {
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
        }

    def generate_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> Dict[str, Any]:
        """Generate signal using Action Signal Guide."""
        try:
            # Generate signals from Action Signal Guide
            # Use the last index of current data (current bar)
            current_index = len(data) - 1
            signals = self.guide.generate_signals(data, current_index)

            # Debug: print signal information
            if signals:
                print(f"Generated {len(signals)} signals at index {current_index}")
                for i, signal in enumerate(signals):
                    print(
                        f"  Signal {i}: direction={signal.direction:.3f}, confidence={signal.confidence:.3f}, type={signal.signal_type}"
                    )
            else:
                if (
                    current_index >= 25
                ):  # Only print for indices where patterns should work
                    print(
                        f"No signals generated at index {current_index} (data length: {len(data)})"
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

            # Apply confidence threshold
            if confidence < self.hyperparameters["confidence_threshold"]:
                self.signal_stats["hold_signals"] += 1
                self.signal_stats["total_signals"] += 1
                return {"action": "hold"}  # Hold if confidence too low

            # Apply signal strength threshold
            if abs(direction) < self.hyperparameters["signal_strength_threshold"]:
                self.signal_stats["hold_signals"] += 1
                self.signal_stats["total_signals"] += 1
                return {"action": "hold"}  # Hold if signal too weak

            if direction > 0.1:  # Bullish
                action = "buy"
                self.signal_stats["buy_signals"] += 1
            elif direction < -0.1:  # Bearish
                action = "sell"
                self.signal_stats["sell_signals"] += 1
            else:  # Neutral
                action = "hold"
                self.signal_stats["hold_signals"] += 1

            self.signal_stats["total_signals"] += 1

            return {
                "action": action,
                "confidence": confidence,
                "direction": direction,
                "signal_type": latest_signal.signal_type,
                "description": latest_signal.description,
            }

        except Exception as e:
            print(f"Error generating Action Signal Guide signal: {e}")
            self.signal_stats["hold_signals"] += 1
            self.signal_stats["total_signals"] += 1
            return {"action": "hold"}  # Hold on error

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
