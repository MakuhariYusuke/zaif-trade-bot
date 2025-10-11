#!/usr/bin/env python3
"""
Live Trading Bot for BTC/JPY using Trained PPO Model.

This script performs live trading on Coincheck exchange using the trained ML model.
Implements sell-biased strategy as requested.
Cross-platform compatible (Windows/Raspberry Pi).
"""

import argparse
import gc
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from numpy.typing import NDArray
import requests
from dotenv import load_dotenv  # type: ignore[import-untyped]
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from ztb.training.policies.policy_utils import predict_with_masks  # type: ignore[attr-defined]
from ztb.trading.live.action_mask_provider import (
    ActionMaskProvider,
    ActionMaskConfig,
)
from ztb.utils.logging_utils import get_logger
from ztb.utils.performance_utils import timed
from ztb.utils.errors import safe_operation

# Optional health check endpoint
try:
    from flask import Flask, jsonify  # type: ignore[import-untyped]

    flask_available = True
except ImportError:
    flask_available = False

# Optional metrics collection
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server  # type: ignore[import-untyped]

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
    from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

    coincheck_available = True
except ImportError:
    coincheck_available = False

# Import position management
try:
    from ztb.trading.environment.components.position_manager import PositionManager

    position_manager_available = True
except ImportError:
    position_manager_available = False
    # Logger will be initialized later, so we can't log here

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
logger = get_logger(__name__)
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
            self.metrics: Optional[Dict[str, Any]] = None

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
                self.coincheck_adapter = CoincheckAdapter(  # type: ignore[name-defined]
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

        # Initialize PositionManager (Bug #25, #26 fix)
        if position_manager_available:
            # Create a simple config object for PositionManager
            class LivePositionConfig:
                """Configuration for PositionManager in live trading."""
                def __init__(self, config_dict: Dict[str, Any]) -> None:
                    self.allow_reverse = config_dict.get("allow_reverse", False)
                    self.transaction_cost = config_dict.get("transaction_cost", 0.001)
                    # Bug #28 fix: Pass max_position_size to prevent scale mismatch
                    self.max_position_size = config_dict.get(
                        "max_position_size", 
                        config_dict.get("min_trade_amount", 0.001)
                    )
                    
            position_config = LivePositionConfig(self.config)
            
            self.position_manager = PositionManager(  # type: ignore[name-defined]
                config=position_config,
                get_price_callback=lambda: self._last_valid_price if self._last_valid_price > 0 else self._get_current_price()
            )
            logger.info(f"PositionManager initialized for live trading (position_size={position_config.max_position_size})")
        else:
            self.position_manager = None
            logger.warning("PositionManager not available, using legacy position logic")

        # Initialize price validation
        self._last_valid_price = 0.0
        
        # Initialize action mask provider for MaskablePPO support (Bug #27 fix)
        mask_config = ActionMaskConfig(
            min_holding_period=self.config.get("min_holding_period", 5),
            enable_forced_close=True,
            max_position_age=self.config.get("max_position_age", 1000)
        )
        self.mask_provider = ActionMaskProvider(mask_config)
        self._is_maskable_ppo = False  # Will be set in _load_model()
        self._current_step = 0
        self._position_entry_step = 0
        logger.info(
            f"ActionMaskProvider initialized (min_holding={mask_config.min_holding_period}, "
            f"max_age={mask_config.max_position_age})"
        )

        # Initialize advanced auto-stop system
        if auto_stop_available:
            try:
                self.auto_stop = create_production_auto_stop()  # type: ignore[name-defined]
                logger.info("Advanced auto-stop system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize auto-stop system: {e}")
                self.auto_stop = None
        else:
            self.auto_stop = None
            logger.warning("Advanced auto-stop system not available")

        # Cache for historical prices to avoid repeated API calls
        # Memory optimization: Use deque with maxlen for automatic size limiting
        self._price_history_max_size = self.config.get("price_history_length", 30)
        self.price_history: deque[float] = deque(maxlen=self._price_history_max_size)
        self._update_price_history()
        
        # Memory optimization: Periodic cleanup counter
        self._cleanup_counter = 0
        self._cleanup_interval = 100  # Clean up every 100 iterations

        # Send startup notification (with schema info if available)
        feature_info = "68 technical indicators (default)"
        if hasattr(self, 'schema_available') and self.schema_available and self.expected_features:
            feature_info = f"{self.expected_features} features (schema-validated ✅)"
        elif hasattr(self, 'expected_features') and self.expected_features:
            feature_info = f"{self.expected_features} features (detected)"
        
        self._send_notification(
            "🚀 BTC/JPY Live Trading Started",
            f"Model: {model_path}\nStrategy: Sell-biased\nMode: {'DEMO' if self.demo_mode else 'LIVE'}\nFeatures: {feature_info}\nTrading Mode: Normal (1M timeframe)\nMemory: Optimized (history={self._price_history_max_size})",
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
            prices = self._get_historical_prices(
                limit=self.config["price_history_length"]
            )
            # Convert list to deque
            self.price_history.clear()
            self.price_history.extend(prices)
            logger.info(
                f"Updated price history with {len(self.price_history)} data points"
            )
        except Exception as e:
            logger.warning(f"Failed to update price history: {e}")
            # Fallback to current price
            current_price = self._get_current_price()
            self.price_history.clear()
            self.price_history.extend([current_price] * self.config["price_history_length"])
    
    def _periodic_cleanup(self) -> None:
        """Perform periodic memory cleanup to prevent accumulation."""
        self._cleanup_counter += 1
        if self._cleanup_counter >= self._cleanup_interval:
            # deque already has maxlen, so no manual trimming needed
            
            # Force garbage collection periodically
            gc.collect()
            
            self._cleanup_counter = 0
            logger.debug(f"Periodic cleanup: price_history={len(self.price_history)} items")

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
                "ZTB_PRICE_HISTORY_LENGTH",  64
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

    def _load_model(self) -> PPO | MaskablePPO:
        """Load the trained PPO or MaskablePPO model.
        
        Bug #27 Fix: Now properly loads MaskablePPO models and uses
        ActionMaskProvider for action masking in production.
        
        Schema Integration: Load schema information for feature validation.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")
        
        # Try loading as MaskablePPO first, fallback to PPO
        try:
            model = MaskablePPO.load(str(self.model_path))
            logger.info("Model loaded as MaskablePPO with action masking support")
            self._is_maskable_ppo = True
        except Exception as e:
            logger.info(f"Not a MaskablePPO model ({e}), loading as standard PPO")
            model = PPO.load(str(self.model_path))
            logger.info("Model loaded as standard PPO (no action masking)")
            self._is_maskable_ppo = False

        # ========================================================================
        # Schema-based feature validation (Phase 3 Integration)
        # ========================================================================
        try:
            from ztb.trading.environment.schema_env_factory import create_env_from_model_path
            from ztb.training.core.feature_schema_manager import FeatureSchemaManager
            
            # Load model schema
            model_name = self.model_path.stem
            schema_manager = FeatureSchemaManager(model_name)
            
            try:
                metadata = schema_manager.load_schema()
                logger.info(f"✅ Schema loaded for model: {model_name}")
                logger.info(f"   Expected features: {metadata.num_features}")
                logger.info(f"   Schema hash: {metadata.schema_hash}")
                logger.info(f"   Created at: {metadata.created_at}")
                
                # Store schema info for feature validation
                self.expected_features = metadata.num_features
                self.feature_names = metadata.feature_names
                self.model_schema_hash = metadata.schema_hash
                self.schema_available = True
                
                logger.info(f"📋 Model feature requirements:")
                logger.info(f"   Total: {len(self.feature_names)} features")
                logger.info(f"   First 5: {self.feature_names[:5]}")
                logger.info(f"   Last 5: {self.feature_names[-5:]}")
                
            except FileNotFoundError:
                logger.warning(f"⚠️  Schema not found for model: {model_name}")
                logger.warning(f"   Schema file expected at: {self.ztb_config.get_model_dir()}/schemas/{model_name}/")
                logger.warning(f"   Falling back to legacy validation")
                logger.warning(f"   Recommendation: Run migration if this is an old model")
                
                self.expected_features = None
                self.feature_names = None
                self.model_schema_hash = None
                self.schema_available = False
                
        except ImportError as e:
            logger.warning(f"Schema system not available: {e}")
            logger.warning("Using legacy feature validation")
            self.expected_features = None
            self.feature_names = None
            self.model_schema_hash = None
            self.schema_available = False

        # Legacy feature validation (fallback)
        try:
            # Temporarily initialize price history for feature checking
            if not hasattr(self, "price_history"):
                current_price = 1000000.0  # Dummy price for checking
                self.price_history = [current_price] * self.config[
                    "price_history_length"
                ]

            # Use dummy price for model loading, real adapter will be used later
            sample_features = self._get_market_features()
            actual_features = len(sample_features)
            
            # Check against schema-based expectation if available
            if self.expected_features is not None:
                if actual_features != self.expected_features:
                    logger.warning(
                        f"Feature count mismatch: model expects {self.expected_features} (schema), "
                        f"got {actual_features} (computed)"
                    )
                else:
                    logger.info(f"Feature count validated: {actual_features} features")
            elif model.observation_space.shape is not None:
                expected_obs_space = model.observation_space.shape[0]
                if actual_features != expected_obs_space:
                    logger.warning(
                        f"Feature count mismatch: model expects {expected_obs_space}, got {actual_features}"
                    )
                    logger.warning("Using only basic features to match training data")
            else:
                logger.warning("Could not determine model observation space shape")

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
    
    async def get_account_balance(self, currency: Optional[str] = None) -> Dict[str, float]:
        """
        Get account balance from exchange using existing adapter.
        
        Args:
            currency: Optional currency filter (e.g., 'BTC', 'JPY')
            
        Returns:
            Dict mapping currency to available balance
            
        Example:
            >>> balances = await trader.get_account_balance()
            >>> print(f"BTC: {balances.get('BTC', 0.0)}, JPY: {balances.get('JPY', 0.0)}")
        """
        if not self.coincheck_adapter:
            logger.warning("Coincheck adapter not available, cannot fetch balance")
            return {}
        
        try:
            balances = await self.coincheck_adapter.get_balance(currency=currency)
            result = {}
            for balance in balances:
                result[balance.currency] = balance.free
            logger.info(f"Account balance: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            self._send_notification(
                "⚠️ Balance Fetch Error",
                f"Failed to retrieve account balance: {str(e)}",
                "error"
            )
            return {}

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
                "model_loaded": hasattr(self, "model"),
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
            "trades_total": Counter(  # type: ignore[name-defined]
                "ztb_trades_total",
                "Total number of trades executed",
                ["action", "dry_run"],
            ),
            "trade_profit": Histogram(  # type: ignore[name-defined]
                "ztb_trade_profit", "Profit/loss per trade", ["action"]
            ),
            "price_fetches": Counter(  # type: ignore[name-defined]
                "ztb_price_fetches_total", "Total price fetch attempts", ["success"]
            ),
            "price_fetch_duration": Histogram(  # type: ignore[name-defined]
                "ztb_price_fetch_duration_seconds", "Price fetch duration"
            ),
            "current_pnl": Gauge("ztb_current_pnl", "Current total profit/loss"),  # type: ignore[name-defined]
            "daily_trades": Gauge("ztb_daily_trades", "Trades executed today"),  # type: ignore[name-defined]
            "price_current": Gauge("ztb_price_current", "Current BTC/JPY price"),  # type: ignore[name-defined]
            "model_predictions": Counter(  # type: ignore[name-defined]
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
            rsi_series = compute_rsi(df, period=period)  # type: ignore[name-defined]
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

    @timed
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

    @timed
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
            # deque with maxlen automatically removes old items
        else:
            # Initialize deque with initial values
            self.price_history.clear()
            self.price_history.extend([current_price] * self.config["price_history_length"])

        # Convert deque to list for calculation functions
        price_list = list(self.price_history)

        # Calculate basic technical indicators
        rsi = self._calculate_rsi(price_list, period=14)
        sma_short = self._calculate_sma(price_list, period=5)  # 5-period SMA
        sma_long = self._calculate_sma(price_list, period=20)  # 20-period SMA

        # Price (normalized)
        price_norm = current_price / 1000000.0  # Similar to training data

        # Volume/quantity (mock for now - would need order book data)
        qty = np.random.uniform(0.001, 0.01)

        # PnL and win flag (based on recent price movement)
        recent_prices = (
            list(self.price_history)[-10:]
            if len(self.price_history) >= 10
            else list(self.price_history)
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
            advanced_features = self._compute_live_features(price_list)
            if advanced_features:
                logger.debug(f"Adding {len(advanced_features)} advanced features")
                # Add advanced features in sorted order for consistency
                for feature_name in sorted(advanced_features.keys()):
                    features.append(advanced_features[feature_name])

        # ========================================================================
        # Feature count validation (Schema-aware)
        # ========================================================================
        # Determine expected feature count from schema or model
        expected_features = 68  # Default fallback
        
        if self.schema_available and self.expected_features is not None:
            # Use schema information (most reliable)
            expected_features = self.expected_features
            logger.debug(f"Using schema-defined feature count: {expected_features}")
        elif (
            self.model
            and hasattr(self.model, "observation_space")
            and hasattr(self.model.observation_space, "shape")
        ):
            # Fallback to model observation space
            shape = getattr(self.model.observation_space, "shape", None)
            if shape is not None and len(shape) > 0:
                expected_features = shape[0]
                logger.debug(f"Using model observation space: {expected_features}")
        else:
            logger.debug(f"Using default feature count: {expected_features}")

        # Validate and adjust feature count
        if len(features) < expected_features:
            # Pad with zeros
            padding_needed = expected_features - len(features)
            features.extend([0.0] * padding_needed)
            logger.debug(
                f"Padded features to {expected_features} dimensions (added {padding_needed})"
            )
        elif len(features) > expected_features:
            # Truncate
            features = features[:expected_features]
            logger.debug(f"Truncated features to {expected_features} dimensions")

        # Schema-based feature order validation (if available)
        if self.schema_available and self.feature_names is not None:
            # TODO: Future enhancement - reorder features based on schema
            # This would ensure features are in the exact order expected by the model
            logger.debug(f"Feature schema available with {len(self.feature_names)} named features")

        logger.debug(f"✅ Final features: {len(features)} (expected: {expected_features})")
        features_array: NDArray[np.float32] = np.array(features, dtype=np.float32)
        return features_array

    def _should_trade_sell_bias(self, action: int) -> bool:
        """
        Apply sell bias to trading decisions with BUY promotion to balance trades.
        Returns True if trade should be executed.
        
        Bug #31 Fix: Allow short position opening after warmup period.
        """
        if action == ACTION_HOLD:  # Hold
            return False

        # Apply sell bias multiplier
        sell_bias = self.config["sell_bias_multiplier"]

        if action == ACTION_SELL:  # Sell signal
            # Bug #33 Fix: Warmup only restricts SHORT opening, not position closing
            # Allow warmup period before enabling short positions
            sell_warmup_trades = self.config.get("sell_warmup_trades", 2)
            
            # Check if this SELL would OPEN a short position (flat → short)
            if self.position == 0 and self.trades_count < sell_warmup_trades:
                # Only suppress SELL when opening new short during warmup
                logger.info(
                    f"Suppressing SHORT opening in warmup period (trade #{self.trades_count + 1}/{sell_warmup_trades})"
                )
                return False
            
            # After warmup OR when closing long: allow SELL
            # (position > 0: closing long, position == 0 and trades >= warmup: opening short OK)
            return True
            
        elif action == ACTION_BUY:  # Buy signal
            # Bug #41 Fix: Always allow BUY when closing short position
            if self.position < 0:
                # Closing short position (short → flat or short → long)
                # Always allow position closing regardless of probability filter
                return True
            
            # Promote BUY actions to balance with SELL bias
            # Use higher probability for BUY to counteract SELL bias from reward function
            buy_probability = min(
                1.0, 1.0 / sell_bias * 1.5
            )  # Boost BUY probability by 1.5x
            return np.random.random() < buy_probability

        return False

    def _execute_trade(self, side: str, amount: float) -> bool:
        """Execute trade on Coincheck with enhanced error handling and notifications."""
        if self.demo_mode:
            logger.info(f"DEMO MODE: Would execute {side} {amount} BTC")
            return True

        # Enhanced error notification for live trading
        try:
            # TODO: Implement actual Coincheck API trading calls
            logger.warning(
                f"LIVE MODE: Trade execution not implemented yet - {side} {amount} BTC"
            )
            self._send_notification(
                "⚠️ Live Trade Not Implemented",
                f"Would execute: {side.upper()} {amount} BTC\n"
                f"Please implement actual API calls\n"
                f"Position: {self.position}, Entry: ¥{self.entry_price:,.0f}",
                "warning",
            )
            return False
        except Exception as e:
            # Critical error notification
            error_msg = f"CRITICAL: Trade execution failed - {str(e)}"
            logger.error(error_msg)
            self._send_notification(
                "🚨 CRITICAL: Trade Execution Error",
                f"Side: {side.upper()}\n"
                f"Amount: {amount} BTC\n"
                f"Error: {str(e)}\n"
                f"Position: {self.position}",
                "error",
            )
            return False

    def _update_position(self, action: int, current_price: float) -> None:
        """Update position based on model action using PositionManager.
        
        Args:
            action: Trading action (0=HOLD, 1=BUY, 2=SELL)
            current_price: Current BTC/JPY price
            
        Fixes:
            Bug #25: PnL calculation now uses PositionManager (prevents entry_price overwrite bug)
            Bug #26: Position closes go to flat (prevents immediate reversal bug)
        """
        if self.position_manager:
            # Use PositionManager for correct position and PnL management
            old_position = self.position_manager.position
            old_trades = self.position_manager.trades_count
            
            # Execute action through PositionManager
            # Note: Using _current_step=0 since live trading doesn't have step counter
            # and min_holding_period=0 to allow flexible trading
            trade_pnl = self.position_manager.execute_action(
                action=action,
                current_step=0,  # Live trading doesn't use step-based timing
                min_holding_period=0  # No min holding period for live trading
            )
            
            # Sync state from PositionManager
            self.position = self.position_manager.position
            self.entry_price = self.position_manager.entry_price
            
            # Update position entry step for mask provider (Bug #27 fix)
            if old_position == 0 and self.position != 0:
                # Position opened
                self._position_entry_step = self._current_step
            elif old_position != 0 and self.position == 0:
                # Position closed
                self._position_entry_step = 0
            
            # Convert numeric position to string representation
            if self.position > 0:
                position_str = "long"
            elif self.position < 0:
                position_str = "short"
            else:
                position_str = "flat"
            
            # Update trade counters
            if self.position_manager.trades_count > old_trades:
                self.trades_count = self.position_manager.trades_count
                self.daily_trades += 1
            
            # Execute actual trade if position changed
            if old_position != self.position_manager.position:
                if action == ACTION_BUY:
                    self._execute_trade("buy", self.config["min_trade_amount"])
                elif action == ACTION_SELL:
                    self._execute_trade("sell", self.config["min_trade_amount"])
            
            # Bug #29 fix: Always sync realized_pnl, even when trade_pnl is 0
            # (Opening positions have 0 trade_pnl but negative entry fees)
            old_total_pnl = self.total_pnl
            self.total_pnl = self.position_manager.realized_pnl
            pnl_change = self.total_pnl - old_total_pnl
            
            # Validate PnL if it changed
            if pnl_change != 0.0:
                # Validate PnL calculation (Reviewer B recommendation)
                if not np.isfinite(self.total_pnl):
                    logger.error(
                        f"Invalid PnL calculation: {self.total_pnl}. Reverting to previous value."
                    )
                    self.total_pnl = old_total_pnl
                    pnl_change = 0.0
                
                # Sanity check: PnL change shouldn't exceed 10x estimated portfolio value
                # Estimate portfolio as 1M JPY base + total accumulated PnL
                estimated_portfolio = 1_000_000.0 + old_total_pnl
                if abs(pnl_change) > estimated_portfolio * 10:
                    logger.warning(
                        f"Suspiciously large PnL change detected: {pnl_change:.2f} JPY "
                        f"(estimated portfolio: {estimated_portfolio:.2f} JPY). "
                        f"This may indicate a calculation bug. Please verify."
                    )
                
                # Update auto-stop system with PnL change
                if self.auto_stop and pnl_change != 0.0:
                    self.auto_stop.update_trade_result(
                        pnl_change,
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
                    f"PnL: {trade_pnl:.2f} JPY\nTotal PnL: {self.total_pnl:.2f} JPY\nPosition: {position_str}\nTrades: {self.trades_count}",
                    "info",
                )
        else:
            # Legacy fallback (should not be used if PositionManager is available)
            logger.warning("Using legacy position management (PositionManager not available)")
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
                
                # Periodic memory cleanup
                self._periodic_cleanup()

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

                # Get model prediction with action masking support (Bug #27 fix)
                obs = features.reshape(1, -1)
                
                # Update mask provider state before prediction
                if hasattr(self, 'mask_provider'):
                    # Sync state with current position
                    self.mask_provider.update_state(
                        current_position=self.position,
                        position_entry_step=getattr(self, '_position_entry_step', 0),
                        current_step=getattr(self, '_current_step', 0),
                        forced_close_reason=None  # TODO: Add forced close detection
                    )
                
                # Predict action based on model type
                if self._is_maskable_ppo:
                    # Use predict_with_masks utility for MaskablePPO
                    # This handles action_masks parameter correctly
                    action, _ = predict_with_masks(
                        self.model,
                        obs,
                        env=self.mask_provider,
                        deterministic=True
                    )
                    mask_info = self.mask_provider.get_mask_info()
                    logger.debug(f"Action mask: {mask_info['mask_human']}")
                else:
                    # Standard PPO without masking
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                action = int(action[0])  # Convert from numpy array to int

                # Increment step counter
                if not hasattr(self, '_current_step'):
                    self._current_step = 0
                self._current_step += 1

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
    """Main entry point for live trading."""
    safe_operation(
        _main_impl,
        logger=get_logger(__name__),
        context="Live trading execution",
    )


def _main_impl() -> None:
    """Implementation of main function"""
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
        start_http_server(metrics_port)  # type: ignore[name-defined]
        logger.info(f"Prometheus metrics server started on port {metrics_port}")

    # Start health check endpoint if Flask is available
    health_app = None
    if (
        flask_available
        and os.getenv("ZTB_ENABLE_HEALTH_CHECK", "false").lower() == "true"
    ):
        health_app = Flask(__name__)  # type: ignore[name-defined]

        @health_app.route("/health")  # type: ignore[misc]
        def health_check() -> Any:
            if "trader" in globals():
                return jsonify(trader.get_health_status())  # type: ignore[name-defined]
            return jsonify({"status": "initializing"})  # type: ignore[name-defined]

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

    trader = LiveTrader(
        args.model_path,
        disable_risk_limits=args.disable_risk_limits,
        dry_run=args.dry_run,
    )
    trader.run_trading_loop(args.duration_hours)


if __name__ == "__main__":
    main()
