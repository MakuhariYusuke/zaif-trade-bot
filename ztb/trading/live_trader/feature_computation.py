"""Feature computation implementation for live trading."""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader


class FeatureComputation:
    """Handles feature computation for model prediction."""

    def __init__(self, live_trader: "LiveTrader") -> None:
        """Initialize feature computation with reference to live trader."""
        self.live_trader = live_trader
        self.logger = get_logger(__name__)

    def compute_features(self) -> np.ndarray[Any]:
        """Compute features for model prediction using full feature engine when available."""
        try:
            # Use cached price history
            prices = list(self.live_trader.price_history)

            if len(prices) < 10:
                logger = self.logger
                logger.warning(f"Insufficient price history: {len(prices)} points")
                # Pad with current price if needed
                current_price = (
                    self.live_trader._last_valid_price
                    if self.live_trader._last_valid_price > 0
                    else 5000000.0
                )
                while len(prices) < 10:
                    prices.append(current_price)

            # Validate prices are reasonable
            prices = [
                max(1000.0, min(10000000.0, p)) for p in prices
            ]  # Clamp to reasonable range

            # Try to use full feature engine if schema is available
            if (
                hasattr(self.live_trader, "schema_available")
                and self.live_trader.schema_available
                and hasattr(self.live_trader, "expected_features")
                and self.live_trader.expected_features
                and self.live_trader.features_available
                and self.live_trader.compute_features_batch is not None
            ):
                try:
                    # Create DataFrame for feature computation
                    df = pd.DataFrame(
                        {
                            "timestamp": pd.date_range(
                                start=pd.Timestamp.now()
                                - pd.Timedelta(minutes=len(prices)),
                                periods=len(prices),
                                freq="1min",
                            ),
                            "open": prices,
                            "high": prices,
                            "low": prices,
                            "close": prices,
                            "volume": [1000] * len(prices),  # Mock volume
                        }
                    )

                    # Compute all features using the feature engine
                    result = self.live_trader.compute_features_batch(df, verbose=False)

                    # Handle different return types
                    if isinstance(result, tuple) and len(result) >= 1:
                        features_df = result[0]
                    else:
                        features_df = result

                    if hasattr(features_df, "columns") and len(features_df) > 0:
                        # Extract feature columns (exclude OHLCV)
                        feature_cols = [
                            col
                            for col in features_df.columns
                            if col
                            not in [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ]
                        ]

                        if feature_cols:
                            # Get the latest row features
                            latest_features = []
                            for col in feature_cols:
                                try:
                                    value = float(features_df[col].iloc[-1])
                                    if np.isfinite(value):
                                        latest_features.append(value)
                                    else:
                                        latest_features.append(0.0)
                                except (ValueError, TypeError, IndexError):
                                    latest_features.append(0.0)

                            # Ensure we have the expected number of features
                            expected_count = self.live_trader.expected_features
                            if len(latest_features) < expected_count:
                                # Pad with zeros if needed
                                latest_features.extend(
                                    [0.0] * (expected_count - len(latest_features))
                                )
                            elif len(latest_features) > expected_count:
                                # Truncate if too many
                                latest_features = latest_features[:expected_count]

                            logger = self.logger
                            logger.debug(
                                f"Computed {len(latest_features)} features using full feature engine"
                            )
                            return np.array(latest_features, dtype=np.float32)

                except Exception as e:
                    logger = self.logger
                    logger.warning(
                        f"Failed to compute features with full engine, falling back to basic: {e}"
                    )

            # Fallback to basic feature computation
            features = []

            # Price-based features
            if len(prices) >= 2:
                features.append(prices[-1] / prices[-2] - 1)  # Return
            else:
                features.append(0.0)

            if len(prices) >= 2:
                features.append((prices[-1] - prices[0]) / prices[0])  # Total return
            else:
                features.append(0.0)

            # Simple moving averages
            if len(prices) >= 5:
                sma5 = sum(prices[-5:]) / 5
                features.append(sma5 / prices[-1] - 1)  # SMA5 ratio
            else:
                features.append(0.0)

            if len(prices) >= 10:
                sma10 = sum(prices[-10:]) / 10
                features.append(sma10 / prices[-1] - 1)  # SMA10 ratio
            else:
                features.append(0.0)

            # RSI (simplified)
            if len(prices) >= 14:
                rsi = self.live_trader._compute_rsi(prices[-14:])
                features.append(rsi / 100.0 - 0.5)  # Normalize around 0
            else:
                features.append(0.0)

            # Pad to expected feature count
            expected_features = getattr(self.live_trader, "expected_features", 64) or 64
            while len(features) < expected_features:
                features.append(0.0)

            features = features[:expected_features]

            # Validate features are finite
            features = [0.0 if not np.isfinite(f) else f for f in features]

            logger = self.logger
            logger.debug(f"Computed {len(features)} basic features as fallback")
            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger = self.logger
            logger.error(f"Error in compute_features: {e}")
            # Return zero features as safe fallback
            expected_features = getattr(self.live_trader, "expected_features", 64) or 64
            return np.zeros(expected_features, dtype=np.float32)
