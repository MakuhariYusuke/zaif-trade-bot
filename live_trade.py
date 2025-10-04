#!/usr/bin/env python3
"""
Live Trading Bot for BTC/JPY using Trained PPO Model.

This script performs live trading on Coincheck exchange using the trained ML model.
Implements sell-biased strategy as requested.
Cross-platform compatible (Windows/Raspberry Pi).
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
import requests
from dotenv import load_dotenv
from stable_baselines3 import PPO

# Optional health check endpoint
try:
    from flask import Flask, jsonify

    flask_available = True
except ImportError:
    flask_available = False

# Optional metrics collection
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    prometheus_available = True
except ImportError:
    prometheus_available = False

# Load environment variables from .env file
load_dotenv()

# Cross-platform path handling
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Initialize compute_features_batch to None for linter recognition
compute_features_batch = None

# Import feature computation
try:
    from ztb.features.feature_engine import (
        compute_features_batch,
    )  # type: ignore[assignment]
    from ztb.features.momentum.rsi import compute_rsi

    features_available = True
except ImportError:
    features_available = False

# Import trading adapters
try:
    from ztb.trading.live.coincheck_adapter import CoincheckAdapter

    coincheck_available = True
except ImportError:
    coincheck_available = False

# Import Discord notifier
from ztb.utils import DiscordNotifier

# Import configuration management
from ztb.utils.config import ZTBConfig

# Action constants for better readability
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2

ACTION_NAMES = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}

# Configure logging with cross-platform path handling
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"

# Create logger with file and console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# File handler
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# Console handler - only add if not already configured by basicConfig
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

logger.addHandler(file_handler)

# Import risk management
try:
    from ztb.risk.advanced_auto_stop import create_production_auto_stop

    auto_stop_available = True
    # Check if the function actually exists
    if not hasattr(create_production_auto_stop, "__call__"):
        auto_stop_available = False
        logger.warning("create_production_auto_stop function not found in module")
except ImportError:
    auto_stop_available = False


class LiveTrader:
    """
    Live trading bot for BTC/JPY using trained PPO model.

    If COINCHECK_API_KEY and COINCHECK_API_SECRET are not set, the bot runs in demo mode and does not execute real trades.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        disable_risk_limits: bool = False,
        dry_run: bool = False,
    ):
        # Cross-platform path handling
        self.model_path = Path(model_path)
        self.ztb_config = ZTBConfig()
        self.config = config or self._get_default_config()
        self.disable_risk_limits = disable_risk_limits
        self.dry_run = dry_run
        self.notifier: Optional[DiscordNotifier] = None

        # Initialize metrics if Prometheus is available
        if prometheus_available:
            self._setup_metrics()
        else:
            self.metrics = None

        # Adjust risk limits if disabled
        if self.disable_risk_limits:
            logger.warning(
                "RISK LIMITS DISABLED - Operating without safety restrictions"
            )
            self.config.update(
                {
                    "max_daily_loss": float("inf"),
                    "max_daily_trades": float("inf"),
                    "emergency_stop_loss": float("inf"),
                }
            )

        # Coincheck API settings (initialize early)
        self.api_key = os.getenv("COINCHECK_API_KEY", "").strip()
        self.api_secret = os.getenv("COINCHECK_API_SECRET", "").strip()
        self.base_url = "https://coincheck.com"

        # Validate API credentials for live trading (check for non-empty values)
        self.demo_mode = not (self.api_key and self.api_secret) or self.dry_run
        if self.demo_mode:
            if self.dry_run:
                logger.info("DRY RUN MODE - No real trades will be executed")
            elif not (self.api_key and self.api_secret):
                logger.warning(
                    "COINCHECK_API_KEY and/or COINCHECK_API_SECRET not set or empty - running in DEMO mode"
                )
                logger.warning(
                    "Set environment variables or create .env file with API credentials for live trading"
                )

        # Initialize Discord notifier with error handling
        webhook_url = os.getenv("DISCORD_WEBHOOK")
        if webhook_url:
            try:
                self.notifier = DiscordNotifier(webhook_url=webhook_url)
            except Exception as e:
                logger.warning(f"Failed to initialize Discord notifier: {e}")
                self.notifier = None
        else:
            logger.info("Discord webhook not configured - notifications disabled")
            self.notifier = None

        # Initialize Coincheck adapter
        if coincheck_available:
            try:
                self.coincheck_adapter = CoincheckAdapter(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    dry_run=self.dry_run,
                )
                logger.info("Coincheck adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Coincheck adapter: {e}")
                self.coincheck_adapter = None
        else:
            self.coincheck_adapter = None
            logger.warning("Coincheck adapter not available")

        # Load and validate model (after coincheck adapter initialization)
        self.model = self._load_model()

        # Initialize trading state
        self.position = 0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades_count = 0
        self.daily_start_pnl = 0.0
        self.daily_trades = 0
        self.daily_start_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Initialize price validation
        self._last_valid_price = 0.0

        # Initialize advanced auto-stop system
        if auto_stop_available:
            try:
                self.auto_stop = create_production_auto_stop()
                logger.info("Advanced auto-stop system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize auto-stop system: {e}")
                self.auto_stop = None
        else:
            self.auto_stop = None
            logger.warning("Advanced auto-stop system not available")

        # Cache for historical prices to avoid repeated API calls
        self.price_history = []
        self._update_price_history()

        # Send startup notification
        self._send_notification(
            "🚀 BTC/JPY Live Trading Started",
            f"Model: {model_path}\nStrategy: Sell-biased\nMode: {'DEMO' if self.demo_mode else 'LIVE'}\nFeatures: 68 technical indicators\nTrading Mode: Normal (1M timeframe)",
            "info",
        )

    def _send_notification(self, title: str, message: str, level: str = "info"):
        """Send notification with error handling."""
        if self.notifier:
            try:
                self.notifier.send_notification(title, message, level)
            except Exception as e:
                logger.warning(f"Failed to send notification: {e}")
        else:
            # Log to console if Discord is not available
            log_level = getattr(logging, level.upper(), logging.INFO)
            logger.log(log_level, f"{title}: {message}")

    def _update_price_history(self) -> None:
        """Update cached price history for technical indicators."""
        try:
            self.price_history = self._get_historical_prices(
                limit=self.config["price_history_length"]
            )
            logger.info(
                f"Updated price history with {len(self.price_history)} data points"
            )
        except Exception as e:
            logger.warning(f"Failed to update price history: {e}")
            # Fallback to current price
            current_price = self._get_current_price()
            self.price_history = [current_price] * self.config["price_history_length"]

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default trading configuration with safety limits using ZTBConfig."""
        return {
            "reward_scaling": self.ztb_config.get_float("ZTB_REWARD_SCALING", 1.0),
            "transaction_cost": self.ztb_config.get_float(
                "ZTB_TRANSACTION_COST", 0.001
            ),  # 0.1%
            "max_position_size": self.ztb_config.get_float(
                "ZTB_MAX_POSITION_SIZE", 0.1
            ),  # Max 10% of available BTC (conservative)
            "sell_bias_multiplier": self.ztb_config.get_float(
                "ZTB_SELL_BIAS_MULTIPLIER", 2.0
            ),  # Bias towards selling
            "min_trade_amount": self.ztb_config.get_float(
                "ZTB_MIN_TRADE_AMOUNT", 0.001
            ),  # Minimum BTC trade
            "max_trades_per_hour": self.ztb_config.get_int(
                "ZTB_MAX_TRADES_PER_HOUR", 6
            ),  # Conservative trading frequency
            "price_check_interval": self.ztb_config.get_int(
                "ZTB_PRICE_CHECK_INTERVAL", 60
            ),  # Check price every 60 seconds (conservative)
            "max_daily_loss": self.ztb_config.get_float(
                "ZTB_MAX_DAILY_LOSS", 10000.0
            ),  # Max daily loss in JPY
            "max_daily_trades": self.ztb_config.get_int(
                "ZTB_MAX_DAILY_TRADES", 50
            ),  # Max trades per day
            "emergency_stop_loss": self.ztb_config.get_float(
                "ZTB_EMERGENCY_STOP_LOSS", 0.05
            ),  # 5% emergency stop loss
            # Technical analysis parameters
            "price_history_length": self.ztb_config.get_int(
                "ZTB_PRICE_HISTORY_LENGTH", 50
            ),  # Length of price history for indicators
            "rsi_neutral_value": self.ztb_config.get_float(
                "ZTB_RSI_NEUTRAL_VALUE", 50.0
            ),  # Neutral RSI value
            "rsi_period": self.ztb_config.get_int(
                "ZTB_RSI_PERIOD", 14
            ),  # RSI calculation period
            # Price validation parameters
            "fallback_price": self.ztb_config.get_float(
                "ZTB_FALLBACK_PRICE", 5000000.0
            ),  # Fallback price for initialization
            "price_min": self.ztb_config.get_int(
                "ZTB_PRICE_MIN", 1000000
            ),  # Minimum valid price (1M JPY)
            "price_max": self.ztb_config.get_int(
                "ZTB_PRICE_MAX", 50000000
            ),  # Maximum valid price (50M JPY)
            "price_change_threshold": self.ztb_config.get_float(
                "ZTB_PRICE_CHANGE_THRESHOLD", 0.20
            ),  # 20% price change threshold
        }

    def _load_model(self) -> PPO:
        """Load the trained PPO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")
        model = PPO.load(str(self.model_path))

        # Check observation space compatibility
        # Get sample features to determine feature count
        try:
            # Temporarily initialize price history for feature checking
            if not hasattr(self, "price_history"):
                current_price = 1000000.0  # Dummy price for checking
                self.price_history = [current_price] * self.config[
                    "price_history_length"
                ]

            # Use dummy price for model loading, real adapter will be used later
            sample_features = self._get_market_features()
            expected_features = len(sample_features)
            actual_obs_space = model.observation_space.shape[0]

            if expected_features != actual_obs_space:
                logger.warning(
                    f"Feature count mismatch: model expects {actual_obs_space}, got {expected_features}"
                )
                logger.warning("Using only basic features to match training data")

        except Exception as e:
            logger.warning(f"Could not verify feature compatibility: {e}")

        # Send model loaded notification
        self._send_notification(
            "✅ Model Loaded Successfully", f"Model path: {self.model_path}", "success"
        )

        return model

    def _get_current_price(self) -> float:
        """Get current BTC/JPY price from Coincheck adapter."""
        start_time = time.time()
        if self.coincheck_adapter:
            try:
                # Run async method synchronously
                import asyncio

                price = asyncio.run(self.coincheck_adapter.get_current_price("btc_jpy"))
                duration = time.time() - start_time

                if price and price > 0:
                    # Record successful price fetch metrics
                    if self.metrics:
                        self.metrics["price_fetches"].labels(success="true").inc()
                        self.metrics["price_fetch_duration"].observe(duration)
                        self.metrics["price_current"].set(price)

                    # Validate price is reasonable (between configured min and max JPY)
                    # Validate price is reasonable (between configured min and max JPY)
                    if not (
                        self.config["price_min"] <= price <= self.config["price_max"]
                    ):
                        logger.critical(
                            f"Invalid price received: {price} JPY (expected range: {self.config['price_min']}-{self.config['price_max']} JPY)"
                        )
                        self._send_notification(
                            "🚨 Critical Error: Invalid Price",
                            f"Received invalid price: {price} JPY\nTerminating trading to prevent errors",
                            "error",
                        )
                        raise SystemExit(
                            "Invalid price received - terminating for safety"
                        )

                    # Check for extreme price changes (more than configured threshold from last known price)
                    if (
                        hasattr(self, "_last_valid_price")
                        and self._last_valid_price > 0
                    ):
                        price_change_pct = (
                            abs(price - self._last_valid_price) / self._last_valid_price
                        )
                        if (
                            price_change_pct > self.config["price_change_threshold"]
                        ):  # Configurable threshold
                            logger.critical(
                                f"Extreme price change detected: {price_change_pct:.1%} from {self._last_valid_price} to {price}"
                            )
                            self._send_notification(
                                "🚨 Critical Error: Extreme Price Change",
                                f"Price changed by {price_change_pct:.1%}\nFrom: {self._last_valid_price} JPY\nTo: {price} JPY\nTerminating for safety",
                                "error",
                            )
                            raise SystemExit(
                                "Extreme price change detected - terminating for safety"
                            )

                    self._last_valid_price = price
                    return price
                else:
                    logger.error("Coincheck adapter returned invalid price")
                    if self.metrics:
                        self.metrics["price_fetches"].labels(success="false").inc()
                    return 0.0
            except SystemExit:
                raise  # Re-raise SystemExit to terminate program
            except Exception as e:
                logger.error(f"Failed to get price from Coincheck adapter: {e}")
                if self.metrics:
                    self.metrics["price_fetches"].labels(success="false").inc()
                return 0.0
        else:
            # Fallback to direct API call
            try:
                response = requests.get(f"{self.base_url}/api/ticker", timeout=10)
                response.raise_for_status()
                data = response.json()

                if not isinstance(data, dict) or "last" not in data:
                    logger.error(f"Invalid response format from Coincheck API: {data}")
                    return 0.0

                price = float(data["last"])
                if price <= 0:
                    logger.error(f"Invalid price received: {price}")
                    return 0.0

                return price

            except requests.exceptions.Timeout:
                logger.error("Timeout while fetching price from Coincheck API")
                return 0.0
            except requests.exceptions.ConnectionError:
                logger.error("Connection error while fetching price from Coincheck API")
                return 0.0
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error while fetching price: {e}")
                return 0.0
            except ValueError as e:
                logger.error(f"Failed to parse price data: {e}")
                return 0.0
            except Exception as e:
                logger.error(f"Unexpected error while fetching price: {e}")
                return 0.0

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status for monitoring."""
        try:
            current_price = self._get_current_price()
            price_healthy = current_price > 0

            return {
                "status": "healthy" if price_healthy else "degraded",
                "timestamp": datetime.now().isoformat(),
                "price_feed": {
                    "healthy": price_healthy,
                    "current_price": current_price if price_healthy else None,
                    "last_update": getattr(self, "_last_price_update", None),
                },
                "model_loaded": hasattr(self, "model") and self.model is not None,
                "coincheck_adapter": self.coincheck_adapter is not None,
                "auto_stop": hasattr(self, "auto_stop") and self.auto_stop is not None,
                "dry_run": self.dry_run,
                "risk_limits_disabled": self.disable_risk_limits,
                "total_pnl": getattr(self, "total_pnl", 0.0),
                "trades_today": getattr(self, "daily_trade_count", 0),
                "price_history_length": len(getattr(self, "price_history", [])),
            }
        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    def _setup_metrics(self) -> None:
        """Set up Prometheus metrics for monitoring."""
        if not prometheus_available:
            return

        self.metrics = {
            "trades_total": Counter(
                "ztb_trades_total",
                "Total number of trades executed",
                ["action", "dry_run"],
            ),
            "trade_profit": Histogram(
                "ztb_trade_profit", "Profit/loss per trade", ["action"]
            ),
            "price_fetches": Counter(
                "ztb_price_fetches_total", "Total price fetch attempts", ["success"]
            ),
            "price_fetch_duration": Histogram(
                "ztb_price_fetch_duration_seconds", "Price fetch duration"
            ),
            "current_pnl": Gauge("ztb_current_pnl", "Current total profit/loss"),
            "daily_trades": Gauge("ztb_daily_trades", "Trades executed today"),
            "price_current": Gauge("ztb_price_current", "Current BTC/JPY price"),
            "model_predictions": Counter(
                "ztb_model_predictions_total", "Total model predictions", ["action"]
            ),
        }

    def _get_historical_prices(self, limit: int = 100) -> List[float]:
        """Get historical BTC/JPY prices from Coincheck."""
        try:
            # Use Coincheck's trades API for historical data
            response = requests.get(
                f"{self.base_url}/api/trades",
                params={"pair": "btc_jpy", "limit": min(limit, 100)},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success", False):
                logger.warning("Coincheck API returned success=False")
                return [self._get_current_price()] * 14

            trades = data.get("data", [])
            if not trades:
                logger.warning("No trade data received from Coincheck")
                return [self._get_current_price()] * 14

            # Extract prices (most recent first, we want oldest first for calculations)
            prices = []
            for trade in trades:
                if isinstance(trade, dict) and "rate" in trade:
                    prices.append(float(trade["rate"]))

            if not prices:
                logger.warning("No valid price data in trades")
                return [self._get_current_price()] * 14

            # Reverse to get chronological order (oldest first)
            prices.reverse()
            logger.info(f"Successfully fetched {len(prices)} historical prices")
            return prices

        except Exception as e:
            logger.warning(f"Failed to get historical prices: {e}, using fallback")
            current_price = self._get_current_price()
            return [current_price] * 14

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index) using existing utility."""
        if len(prices) < self.config["rsi_period"] + 1:
            return self.config["rsi_neutral_value"]  # Configurable neutral RSI

        try:
            # Create DataFrame for compute_rsi
            df = pd.DataFrame({"close": prices})
            rsi_series = compute_rsi(df, period=period)
            return (
                float(rsi_series.iloc[-1])
                if not rsi_series.empty
                else self.config["rsi_neutral_value"]
            )
        except Exception as e:
            logger.warning(
                f"Failed to compute RSI with utility: {e}, falling back to manual calculation"
            )
            # Fallback to manual calculation
            gains = []
            losses = []

            for i in range(1, len(prices)):
                change = prices[i] - prices[i - 1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return max(0, min(100, rsi))  # Clamp between 0-100

    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0

        return sum(prices[-period:]) / period

    def _compute_live_features(self, prices: List[float]) -> Dict[str, float]:
        """Compute features available for live trading from price data."""
        if not features_available or len(prices) < 14:
            return {}

        # Create a DataFrame with OHLCV-like structure for feature computation
        # Use price as open/high/low/close, and mock volume data
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start=pd.Timestamp.now() - pd.Timedelta(minutes=len(prices)),
                    periods=len(prices),
                    freq="1min",
                ),
                "open": prices,
                "high": prices,  # Mock high as current price
                "low": prices,  # Mock low as current price
                "close": prices,
                "volume": [1000] * len(prices),  # Mock volume
            }
        )

        # Check if compute_features_batch is available before calling
        if compute_features_batch is None:
            logger.warning("compute_features_batch not available")
            return {}

        try:
            # Compute features using the feature engine
            result = compute_features_batch(df, verbose=False)

            # Handle different return types from compute_features_batch
            if isinstance(result, tuple) and len(result) >= 1:
                features_df = result[0]  # First element is DataFrame
            else:
                features_df = result

            if not hasattr(features_df, "columns"):
                logger.warning("Feature computation returned unexpected format")
                return {}

            # Extract the latest feature values
            latest_features = {}
            for col in features_df.columns:
                if col not in ["timestamp", "open", "high", "low", "close", "volume"]:
                    try:
                        latest_features[col] = float(features_df[col].iloc[-1])
                    except (ValueError, TypeError):
                        continue

            return latest_features

        except Exception as e:
            logger.warning(f"Failed to compute advanced features: {e}")
            return {}

    def _get_market_features(self) -> NDArray[np.floating]:
        """Get current market features for model prediction with comprehensive indicators."""
        max_retries = 3
        current_price = 0.0

        # If coincheck_adapter is not initialized yet (during model loading), use dummy price
        if self.coincheck_adapter is None:
            current_price = self.config[
                "fallback_price"
            ]  # Configurable fallback price for initialization
            logger.debug("Using fallback price during initialization")
        else:
            for attempt in range(max_retries):
                try:
                    current_price = self._get_current_price()
                    if current_price > 0:
                        break
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch current price (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(2)

            if current_price <= 0:
                logger.error(
                    "Unable to fetch current price after multiple attempts. Using fallback price."
                )
                current_price = self.config[
                    "fallback_price"
                ]  # Configurable fallback price
                self._send_notification(
                    "⚠️ Price Fetch Failed",
                    "Using fallback price for feature calculation",
                    "warning",
                )

        # Update price history with current price
        if self.price_history:
            self.price_history.append(current_price)
            # Keep only last configured number of prices
            self.price_history = self.price_history[
                -self.config["price_history_length"] :
            ]
        else:
            self.price_history = [current_price] * self.config["price_history_length"]

        # Calculate basic technical indicators
        rsi = self._calculate_rsi(self.price_history, period=14)
        sma_short = self._calculate_sma(self.price_history, period=5)  # 5-period SMA
        sma_long = self._calculate_sma(self.price_history, period=20)  # 20-period SMA

        # Price (normalized)
        price_norm = current_price / 1000000.0  # Similar to training data

        # Volume/quantity (mock for now - would need order book data)
        qty = np.random.uniform(0.001, 0.01)

        # PnL and win flag (based on recent price movement)
        recent_prices = (
            self.price_history[-10:]
            if len(self.price_history) >= 10
            else self.price_history
        )
        if len(recent_prices) >= 2:
            pnl = (recent_prices[-1] - recent_prices[0]) * 0.001  # Small position PnL
            win = 1 if pnl > 0 else 0
        else:
            pnl = 0.0
            win = 0

        # Start with basic features and extend to 68 dimensions
        features = [
            rsi,  # RSI (14-period)
            sma_short,  # Short SMA (5-period)
            sma_long,  # Long SMA (20-period)
            price_norm,  # Normalized price
            qty,  # Quantity
            pnl,  # Recent PnL
            win,  # Win flag
        ]

        # Add advanced features if available
        if features_available and len(self.price_history) >= 20:
            advanced_features = self._compute_live_features(self.price_history)
            if advanced_features:
                logger.debug(f"Adding {len(advanced_features)} advanced features")
                # Add advanced features in sorted order for consistency
                for feature_name in sorted(advanced_features.keys()):
                    features.append(advanced_features[feature_name])

        # Ensure we have the correct number of features for the model
        expected_features = 68  # Default fallback
        try:
            if (
                self.model
                and hasattr(self.model, "observation_space")
                and hasattr(self.model.observation_space, "shape")
            ):
                shape = getattr(self.model.observation_space, "shape", None)
                if shape is not None and len(shape) > 0:
                    expected_features = shape[0]
        except Exception:
            logger.warning(
                "Could not determine model observation space, using default feature count"
            )

        if len(features) < expected_features:
            # Pad with zeros
            padding_needed = expected_features - len(features)
            features.extend([0.0] * padding_needed)
            logger.debug(
                f"Padded features to {expected_features} dimensions (added {padding_needed})"
            )
        elif len(features) > expected_features:
            features = features[:expected_features]
            logger.debug(f"Truncated features to {expected_features} dimensions")

        logger.debug(f"Total features: {len(features)} (expected: {expected_features})")
        return np.array(features, dtype=np.float32)

    def _should_trade_sell_bias(self, action: int) -> bool:
        """
        Apply sell bias to trading decisions with BUY promotion to balance trades.
        Returns True if trade should be executed.
        """
        if action == ACTION_HOLD:  # Hold
            return False

        # Apply sell bias multiplier
        sell_bias = self.config["sell_bias_multiplier"]

        if action == ACTION_SELL:  # Sell signal
            # Don't allow sell if position is flat (0) - prevent immediate shorting
            if self.position == 0:
                logger.info("Suppressing SELL signal when position is flat")
                return False
            # Don't allow sell in first few trades to stabilize
            if self.trades_count < 2:
                logger.info(
                    f"Suppressing SELL signal in early trades (trade #{self.trades_count + 1})"
                )
                return False
            return True  # Allow sell signals otherwise
        elif action == ACTION_BUY:  # Buy signal
            # Promote BUY actions to balance with SELL bias
            # Use higher probability for BUY to counteract SELL bias from reward function
            buy_probability = min(
                1.0, 1.0 / sell_bias * 1.5
            )  # Boost BUY probability by 1.5x
            return np.random.random() < buy_probability

        return False

    def _execute_trade(self, side: str, amount: float) -> bool:
        """Execute trade on Coincheck."""
        if self.demo_mode:
            logger.info(f"DEMO MODE: Would execute {side} {amount} BTC")
            return True

        # TODO: Implement actual Coincheck API trading calls
        logger.warning(
            f"LIVE MODE: Trade execution not implemented yet - {side} {amount} BTC"
        )
        self._send_notification(
            "⚠️ Live Trade Not Implemented",
            f"Would execute: {side.upper()} {amount} BTC\nPlease implement actual API calls",
            "warning",
        )
        return False

    def _update_position(self, action: int, current_price: float) -> None:
        """Update position based on model action."""
        old_position = self.position

        # Handle position changes based on action and current position
        if action == ACTION_BUY:
            if self.position <= 0:  # Enter long position or reverse from short
                self.position = 1
                self.entry_price = current_price
                self.trades_count += 1
                self.daily_trades += 1
                self._execute_trade("buy", self.config["min_trade_amount"])
            # If already long, do nothing (hold)

        elif action == ACTION_SELL:
            if self.position >= 0:  # Enter short position or reverse from long
                self.position = -1
                self.entry_price = current_price
                self.trades_count += 1
                self.daily_trades += 1
                self._execute_trade("sell", self.config["min_trade_amount"])
            # If already short, do nothing (hold)

        # Calculate PnL if position was closed/reversed
        if old_position != self.position and old_position != 0:
            # Calculate PnL based on the closed position
            if old_position > 0:  # Closing long position
                pnl = (current_price - self.entry_price) * self.config[
                    "min_trade_amount"
                ]
            else:  # Closing short position
                pnl = (self.entry_price - current_price) * self.config[
                    "min_trade_amount"
                ]

            self.total_pnl += pnl

            # Update auto-stop system with trade result
            if self.auto_stop:
                self.auto_stop.update_trade_result(
                    pnl,
                    {
                        "action": action,
                        "entry_price": self.entry_price,
                        "exit_price": current_price,
                        "old_position": old_position,
                        "new_position": self.position,
                        "timestamp": datetime.now(),
                    },
                )

            self._send_notification(
                "📊 Position Update",
                f"PnL: {pnl:.2f} JPY\nTotal PnL: {self.total_pnl:.2f} JPY\nTrades: {self.trades_count}",
                "info",
            )

    def run_trading_loop(self, duration_hours: int = 1) -> None:
        """Run the main trading loop."""
        logger.info(f"Starting live trading for {duration_hours} hours")
        logger.info("Strategy: Sell-biased BTC/JPY trading")

        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)

        trades_this_hour = 0
        hour_start = datetime.now()
        last_price_update = datetime.now() - timedelta(
            minutes=10
        )  # Force initial update

        while datetime.now() < end_time:
            try:
                current_time = datetime.now()

                # Reset daily counters if new day (use UTC for consistency)
                current_date = current_time.date()
                daily_start_date = self.daily_start_time.date()
                if current_date > daily_start_date:
                    self.daily_start_pnl = self.total_pnl
                    self.daily_trades = 0
                    self.daily_start_time = current_time.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    logger.info("Daily counters reset")

                # Check daily risk limits
                if not self.disable_risk_limits:
                    daily_loss = self.total_pnl - self.daily_start_pnl
                    if daily_loss <= -self.config["max_daily_loss"]:
                        logger.error(f"Daily loss limit reached: {daily_loss:.2f} JPY")
                        self._send_notification(
                            "🚨 Daily Loss Limit Reached",
                            f"Daily Loss: {daily_loss:.2f} JPY\nStopping trading for safety",
                            "error",
                        )
                        break

                    if self.daily_trades >= self.config["max_daily_trades"]:
                        logger.warning(
                            f"Daily trade limit reached: {self.daily_trades}"
                        )
                        time.sleep(self.config["price_check_interval"])
                        continue

                    # Check emergency stop loss (percentage based)
                    if self.entry_price > 0:
                        current_price = self._get_current_price()
                        if current_price > 0:
                            loss_pct = (
                                abs(current_price - self.entry_price) / self.entry_price
                            )
                            if loss_pct >= self.config["emergency_stop_loss"]:
                                logger.critical(
                                    f"Emergency stop loss triggered: {loss_pct:.1%} loss"
                                )
                                self._send_notification(
                                    "🚨 Emergency Stop Loss Triggered",
                                    f"Loss: {loss_pct:.1%}\nEntry: {self.entry_price:.0f}\nCurrent: {current_price:.0f}",
                                    "error",
                                )
                                # Close position immediately
                                if self.position != 0:
                                    self._update_position(
                                        (
                                            ACTION_SELL
                                            if self.position > 0
                                            else ACTION_BUY
                                        ),
                                        current_price,
                                    )
                                break

                # Update price history every 5 minutes
                if (current_time - last_price_update).seconds >= 300:
                    self._update_price_history()
                    last_price_update = current_time

                # Reset hourly trade counter
                if (current_time - hour_start).seconds >= 3600:
                    trades_this_hour = 0
                    hour_start = current_time

                # Check trading frequency limit
                if trades_this_hour >= self.config["max_trades_per_hour"]:
                    time.sleep(self.config["price_check_interval"])
                    continue

                # Get current market features
                features = self._get_market_features()

                # Get model prediction
                obs = features.reshape(1, -1)
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action[0])

                # Balance SELL bias by converting some SELL predictions to BUY
                if action == ACTION_SELL and self.position == 0:
                    # Convert 40% of SELL predictions to BUY when position is flat to promote buying
                    if np.random.random() < 0.4:
                        action = ACTION_BUY
                        logger.info("Converted SELL prediction to BUY for balance")

                # Apply sell bias
                should_trade = self._should_trade_sell_bias(action)

                # Debug logging
                logger.info(
                    f"Model prediction: {ACTION_NAMES[action]}, should_trade: {should_trade}, position: {self.position}, trades: {self.trades_count}"
                )

                if should_trade:
                    current_price = self._get_current_price()
                    if current_price > 0:
                        old_pnl = self.total_pnl
                        old_trades = self.trades_count
                        self._update_position(action, current_price)
                        trades_this_hour += 1

                        # トレード結果の詳細なログ出力
                        trade_success = self.trades_count > old_trades
                        if trade_success:
                            pnl_change = self.total_pnl - old_pnl
                            logger.info(
                                f"✅ Trade executed successfully: {ACTION_NAMES[action]} at {current_price:.0f} JPY"
                            )
                            logger.info(
                                f"💰 PnL Change: {pnl_change:+.2f} JPY | Total PnL: {self.total_pnl:.2f} JPY"
                            )
                            logger.info(
                                f"📊 Total Trades: {self.trades_count} | Daily Trades: {self.daily_trades}"
                            )

                            # Send Discord notification for successful trade
                            self._send_notification(
                                "💹 Trade Executed",
                                f"Action: {ACTION_NAMES[action]}\nPrice: {current_price:.0f} JPY\nPnL Change: {pnl_change:+.2f} JPY\nTotal PnL: {self.total_pnl:.2f} JPY\nDaily Trades: {self.daily_trades}",
                                "success",
                            )
                        else:
                            logger.warning(
                                f"⚠️ Trade attempted but not executed: {ACTION_NAMES[action]} at {current_price:.0f} JPY"
                            )

                # Wait before next check
                time.sleep(self.config["price_check_interval"])

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping trading...")
                self._send_notification(
                    "⏹️ Trading Stopped", "Manual stop requested", "info"
                )
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self._send_notification(
                    "❌ Trading Error",
                    f"Error: {str(e)}\nWill continue after delay",
                    "error",
                )
                # Wait longer on error to avoid rapid error loops
                time.sleep(60)

        # Send completion notification
        auto_stop_status = ""
        if self.auto_stop:
            status = self.auto_stop.get_status()
            auto_stop_status = (
                f"\nAuto-Stop Status: {'Active' if status['is_active'] else 'Stopped'}"
            )

        self._send_notification(
            "🏁 Live Trading Completed",
            f"Duration: {duration_hours} hours\nTotal PnL: {self.total_pnl:.2f} JPY\nTotal Trades: {self.trades_count}{auto_stop_status}",
            "info",
        )

        logger.info(f"Live trading completed. Total PnL: {self.total_pnl:.2f} JPY")


def main() -> None:
    # Configure logging with dynamic log level from environment variable
    log_level = os.getenv("ZTB_LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Live BTC/JPY Trading Bot")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=1,
        help="Trading duration in hours (default: 1)",
    )
    parser.add_argument(
        "--disable-risk-limits",
        action="store_true",
        help="Disable all risk limits (daily loss, trade count, emergency stop) - USE WITH CAUTION",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enable dry run mode - no real trades will be executed",
    )

    args = parser.parse_args()

    # Start Prometheus metrics server if available
    if (
        prometheus_available
        and os.getenv("ZTB_ENABLE_METRICS", "false").lower() == "true"
    ):
        metrics_port = int(os.getenv("ZTB_METRICS_PORT", "8000"))
        start_http_server(metrics_port)
        logger.info(f"Prometheus metrics server started on port {metrics_port}")

    # Start health check endpoint if Flask is available
    health_app = None
    if (
        flask_available
        and os.getenv("ZTB_ENABLE_HEALTH_CHECK", "false").lower() == "true"
    ):
        health_app = Flask(__name__)

        @health_app.route("/health")
        def health_check() -> Any:
            if "trader" in globals():
                return jsonify(trader.get_health_status())
            return jsonify({"status": "initializing"})

        health_port = int(os.getenv("ZTB_HEALTH_PORT", "8080"))
        import threading

        health_thread = threading.Thread(
            target=lambda: health_app.run(
                host="0.0.0.0", port=health_port, debug=False
            ),
            daemon=True,
        )
        health_thread.start()
        logger.info(f"Health check endpoint started on port {health_port}")

    if args.dry_run:
        logger.info("DRY RUN MODE - No real trades will be executed")

    try:
        trader = LiveTrader(
            args.model_path,
            disable_risk_limits=args.disable_risk_limits,
            dry_run=args.dry_run,
        )
        trader.run_trading_loop(args.duration_hours)

    except Exception as e:
        logger.error(f"Failed to start live trading: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
