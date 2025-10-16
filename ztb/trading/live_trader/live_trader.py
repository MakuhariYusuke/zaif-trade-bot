#!/usr/bin/env python3
"""
Live Trader implementation for BTC/JPY trading.
"""

import asyncio
import gc
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

# Type definitions for better type safety
TradingConfig = Dict[str, Any]  # Configuration dictionary for trading parameters

# Add project root to path
from ztb.utils.path_utils import get_project_root
project_root = get_project_root()
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO, SAC

from ztb.trading.live.action_mask_provider import (
    ActionMaskConfig,
    ActionMaskProvider,
)
from ztb.trading.live_trader.action_prediction import ActionPrediction
from ztb.trading.live_trader.config import LiveTradingOptions, prometheus_available
from ztb.trading.live_trader.feature_computation import FeatureComputation
from ztb.trading.live_trader.health_monitoring import HealthMonitoring
from ztb.trading.live_trader.model_loading import ModelLoading
from ztb.trading.live_trader.trading_loop import TradingLoop
from ztb.trading.live.registry.broker_registry import get_broker_registry
from ztb.utils.errors import InsufficientFundsError, MinimumSizeError, OrderNotFoundError

# Import feature computation
try:
    from ztb.features.feature_engine import compute_features_batch
    from ztb.features.momentum.rsi import compute_rsi

    features_available = True
except ImportError:
    features_available = False
    compute_features_batch = None

# Import trading adapters
try:
    from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter
    from ztb.trading.live.registry.broker_registry import get_broker_registry

    coincheck_available = True
except ImportError:
    coincheck_available = False

# Import position management
try:
    from ztb.trading.environment.components.position_manager import PositionManager

    position_manager_available = True
except ImportError:
    position_manager_available = False

# Import Discord notifier
from ztb.utils.notify.discord import DiscordNotifier

# Import utility modules
from ztb.utils.cache_utils import TTLCache
from ztb.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from ztb.utils.logging_utils import get_logger
from ztb.utils.performance_utils import PerformanceMonitor, timed
from ztb.utils.rate_limiter import TokenBucketRateLimiter, RateLimitConfig

# Import configuration management
from ztb.utils.config import ZTBConfig

# Import risk management
try:
    from ztb.risk.advanced_auto_stop import create_production_auto_stop

    auto_stop_available = True
    # Check if the function actually exists
    if not hasattr(create_production_auto_stop, "__call__"):
        auto_stop_available = False
        logger = get_logger(__name__)
        logger.warning("create_production_auto_stop function not found in module")
except ImportError:
    auto_stop_available = False

# Load environment variables from .env file
load_dotenv()

# Cross-platform path handling - add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.constants import (
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_SELL,
    ACTION_NAMES,
)
from ztb.trading.live.exchanges.coincheck.config import (
    get_coincheck_credentials,
    get_coincheck_credentials_optional,
    validate_credentials,
)
from ztb.trading.live.exchanges.bitflyer.config import (
    get_bitflyer_credentials,
)


class LiveTrader:
    """
    Live trading bot for BTC/JPY using trained reinforcement learning models.

    This class implements a production-ready trading system that can operate in live markets
    using pre-trained PPO or SAC models. It includes comprehensive risk management, monitoring,
    and error handling features.

    Architecture:
    - Modular design with separate components for different responsibilities
    - TradingLoop: Main trading execution logic
    - FeatureComputation: Technical indicator calculation
    - ActionPrediction: Model inference and action selection
    - HealthMonitoring: System health checks and metrics
    - ModelLoading: Model loading and validation

    Key Features:
    - Support for PPO and SAC algorithms
    - Multiple exchange integrations (Coincheck, Bitflyer)
    - Comprehensive risk management with configurable limits
    - Real-time monitoring and health checks
    - Discord notifications for trade alerts
    - Dry-run mode for testing without real trades
    - Automatic position management and PnL tracking
    - Memory-efficient operation with periodic cleanup
    - Configurable caching, rate limiting, and circuit breaking

    Safety Features:
    - Emergency stop loss protection
    - Daily loss and trade count limits
    - API credential validation
    - Circuit breaker pattern for error handling
    - Comprehensive logging and error reporting
    - Graceful degradation on component failures

    Configuration:
    The system supports extensive configuration through the config parameter:
    - price_cache_ttl: Price cache TTL in seconds (default: 30.0)
    - requests_per_second: API rate limit (default: 1.0)
    - burst_limit: API burst limit (default: 5)
    - circuit_failure_threshold: Circuit breaker threshold (default: 5)
    - circuit_recovery_timeout: Circuit breaker recovery time (default: 60.0)
    - cleanup_interval: Memory cleanup interval (default: 100)
    - max_consecutive_errors: Max consecutive errors before shutdown (default: 5)

    Args:
        model_path: Path to trained model or LiveTradingOptions configuration
        config: Optional trading configuration overrides
        disable_risk_limits: Disable all risk management limits (dangerous!)
        dry_run: Operate in simulation mode without real trades

    Environment Variables:
        COINCHECK_API_KEY: Coincheck exchange API key
        COINCHECK_API_SECRET: Coincheck exchange API secret
        BITFLYER_API_KEY: Bitflyer exchange API key
        BITFLYER_API_SECRET: Bitflyer exchange API secret
        DISCORD_WEBHOOK_URL: Discord webhook for notifications

    Note:
        If exchange API credentials are not set, the bot runs in demo mode
        and does not execute real trades. This is the default safe mode.
    """

    compute_features_batch: Optional[Callable[[pd.DataFrame, Optional[List[str]], int, bool, Optional[bool], Optional[int], Optional[bool]], Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, float]]]]]

    # Instance attributes with type hints
    config: TradingConfig

    def __init__(
        self,
        model_path: Union[str, Path, LiveTradingOptions],
        config: Optional[Dict[str, Any]] = None,
        disable_risk_limits: bool = False,
        dry_run: bool = False,
    ) -> None:
        logger = get_logger(__name__)
        logger.info("LiveTrader.__init__ started")
        print("LiveTrader.__init__ started")
        
        # Quick dry-run initialization
        if dry_run:
            self._init_dry_run(model_path, disable_risk_limits)
            return

        self._init_normal(model_path, config, disable_risk_limits)

    def _init_dry_run(self, model_path: Union[str, Path, LiveTradingOptions], disable_risk_limits: bool) -> None:
        """Initialize for dry-run mode with live components."""
        logger = get_logger(__name__)
        logger.info("Dry-run mode: using simplified initialization with live components")
        self.dry_run = True
        if isinstance(model_path, LiveTradingOptions):
            options = model_path
            self.model_path = Path(options.model_path).expanduser()
        else:
            self.model_path = Path(model_path).expanduser()
            options = LiveTradingOptions(
                model_path=self.model_path,
                algorithm="sac" if "sac" in str(self.model_path).lower() else "ppo",
                venue="coincheck",
                disable_risk_limits=disable_risk_limits,
                dry_run=True,
            )
        self.options = options
        self.disable_risk_limits = options.disable_risk_limits
        self.algorithm = options.algorithm
        self.ACTION_NAMES = ACTION_NAMES
        self.price_history: deque[float] = deque(maxlen=100)
        self.expected_features = 64
        self.feature_names = None
        self.schema_available = False
        
        # Initialize essential attributes for dry-run
        self.price_cache = TTLCache(ttl_seconds=30.0)
        self._last_valid_price = 0.0
        self.total_pnl = 0.0
        self.position = 0
        self.entry_price = 0.0
        self.trades_count = 0
        self.daily_start_pnl = 0.0
        self.daily_trades = 0
        self.notifier = None
        self.config = {'price_history_length': 100}  # Basic config for dry-run
        self._current_step = 0
        self._cleanup_counter = 0
        self._cleanup_interval = 100
        
        # Initialize exchange adapter for live price access in dry-run
        venue = self.options.venue.lower()
        if venue == "coincheck":
            self.api_key, self.api_secret = get_coincheck_credentials_optional()
            self.base_url = "https://coincheck.com"
            adapter_name = "coincheck"
        else:
            raise ValueError(f"Unsupported venue for dry-run: {venue}")
        
        try:
            broker_registry = get_broker_registry()
            self.exchange_adapter = broker_registry.get_broker(
                adapter_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                dry_run=True,  # Force dry-run mode for adapter
            )
            logger.info(f"{venue.upper()} adapter initialized for dry-run")
        except Exception as e:
            logger.warning(f"Failed to initialize {venue.upper()} adapter for dry-run: {e}")
            self.exchange_adapter = None
        
        # Initialize PositionManager for dry-run
        if position_manager_available:
            try:
                # Create a simple config object for PositionManager
                class LivePositionConfig:
                    """Configuration for PositionManager in live trading."""
                    def __init__(self, config_dict: Dict[str, Any]) -> None:
                        self.allow_reverse = config_dict.get("allow_reverse", False)
                        self.transaction_cost = config_dict.get("transaction_cost", 0.001)
                        self.max_position_size = config_dict.get(
                            "max_position_size", 
                            config_dict.get("min_trade_amount", 0.001)
                        )
                        self.enforce_reverse_cooldown = config_dict.get("enforce_reverse_cooldown", False)
                        self.initial_portfolio_value = config_dict.get("initial_portfolio_value", 200000.0)
                        
                position_config = LivePositionConfig(self.config)
                
                self.position_manager = PositionManager(
                    config=position_config,
                    get_price_callback=lambda: self._last_valid_price if hasattr(self, '_last_valid_price') and self._last_valid_price > 0 else 5000000.0
                )
                logger.info(f"PositionManager initialized for dry-run (position_size={position_config.max_position_size})")
            except Exception as e:
                logger.warning(f"Failed to initialize PositionManager for dry-run: {e}")
                self.position_manager = None
        else:
            self.position_manager = None
            logger.warning("PositionManager not available, using legacy position logic")
        
        # Initialize action mask provider for dry-run
        mask_config = ActionMaskConfig(
            min_holding_period=self.config.get("min_holding_period", 5),
            enable_forced_close=True,
            max_position_age=self.config.get("max_position_age", 1000)
        )
        self.mask_provider = ActionMaskProvider(mask_config)
        self._is_maskable_ppo = False
        self._position_entry_step = 0
        
        # Initialize minimal components for dry-run
        self.trading_loop = TradingLoop(self)
        self.feature_computation = FeatureComputation(self)
        self.action_prediction = ActionPrediction(self)
        self.health_monitoring = HealthMonitoring(self)
        self.model_loading = ModelLoading(self)
        # Load model for dry-run
        self.model = self.model_loading.load_model()
        
        # Initialize price history with current live price for dry-run
        try:
            # Get live price synchronously for dry-run initialization
            import requests
            response = requests.get("https://coincheck.com/api/ticker", timeout=5)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "last" in data:
                current_price = float(data["last"])
                self._last_valid_price = current_price
                self.price_history.clear()
                self.price_history.extend([current_price] * 100)  # Fill with current price
                logger.info(f"Initialized price history with live price: ¥{current_price:,.0f}")
            else:
                raise ValueError("Invalid API response")
        except Exception as e:
            logger.warning(f"Failed to initialize price history with live price: {e}, using fallback")
            fallback_price = 5000000.0
            self._last_valid_price = fallback_price
            self.price_history.clear()
            self.price_history.extend([fallback_price] * 100)
        
        logger.info("Dry-run initialization completed with live components")

    def _init_exchange_adapter(self) -> None:
        """Initialize exchange adapter for live trading."""
        logger = get_logger(__name__)
        
        # Exchange API settings based on venue
        venue = self.options.venue.lower()
        if venue == "coincheck":
            self.api_key, self.api_secret = get_coincheck_credentials()
            self.base_url = "https://coincheck.com"
            adapter_name = "coincheck"
        elif venue == "bitflyer":
            self.api_key, self.api_secret = get_bitflyer_credentials()
            self.base_url = "https://api.bitflyer.com"
            adapter_name = "bitflyer"
        else:
            raise ValueError(f"Unsupported venue: {venue}")

        # Validate API credentials for live trading (check for non-empty values)
        self.demo_mode = not (self.api_key and self.api_secret) or self.dry_run
        if self.demo_mode:
            if self.dry_run:
                logger.info("DRY RUN MODE - No real trades will be executed")
            elif not (self.api_key and self.api_secret):
                logger.warning(
                    f"{venue.upper()}_API_KEY and/or {venue.upper()}_API_SECRET not set or empty - running in DEMO mode"
                )
                logger.warning(
                    "Set environment variables or create .env file with API credentials for live trading"
                )

        try:
            broker_registry = get_broker_registry()
            self.exchange_adapter = broker_registry.get_broker(
                adapter_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                dry_run=self.dry_run,
            )
            logger.info(f"{venue.upper()} adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize {venue.upper()} adapter: {e}")
            self.exchange_adapter = None

    def _init_position_manager(self) -> None:
        """Initialize PositionManager for live trading."""
        logger = get_logger(__name__)
        logger.info(f"PositionManager available: {position_manager_available}")
        
        if position_manager_available:
            try:
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
                        self.enforce_reverse_cooldown = config_dict.get("enforce_reverse_cooldown", False)
                        self.initial_portfolio_value = config_dict.get("initial_portfolio_value", 200000.0)
                        
                position_config = LivePositionConfig(self.config)
                
                self.position_manager = PositionManager(  # type: ignore[name-defined]
                    config=position_config,
                    get_price_callback=lambda: self._last_valid_price if hasattr(self, '_last_valid_price') and self._last_valid_price > 0 else 5000000.0
                )
                logger.info(f"PositionManager initialized for live trading (position_size={position_config.max_position_size})")
            except Exception as e:
                logger.warning(f"Failed to initialize PositionManager: {e}")
                self.position_manager = None
        else:
            self.position_manager = None
            logger.warning("PositionManager not available, using legacy position logic")

    def _init_trading_state(self) -> None:
        """Initialize trading state variables."""
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
        
        # Initialize schema attributes
        self.schema_available = False
        self.expected_features = 64  # Default
        self.feature_names = None
        
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
        logger = get_logger(__name__)
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

    def _init_normal(self, model_path: Union[str, Path, LiveTradingOptions], config: Optional[Dict[str, Any]], disable_risk_limits: bool) -> None:
        """Initialize for normal live trading mode."""
        logger = get_logger(__name__)
        logger.info("Starting normal live trader initialization")
        
        # Store model_path for later use
        self.model_path = model_path

        try:
            # Initialize basic attributes
            self.compute_features_batch = compute_features_batch
            self._setup_archive_config()
            self._setup_model_options(model_path, disable_risk_limits)
            self._validate_model_path()
            self._setup_config(config)

            # Initialize core components
            self._init_core_components()
            self._init_trading_components()
            self._init_monitoring_components()

            logger.info("Normal live trader initialization completed")
            logger.debug(f"Trading config loaded: {self.config}")
            self.disable_risk_limits = self.options.disable_risk_limits
            self.dry_run = self.options.dry_run

            # Initialize Discord notifications
            if not self.dry_run and os.getenv("DISCORD_WEBHOOK_URL"):
                try:
                    self.notifier = DiscordNotifier(os.getenv("DISCORD_WEBHOOK_URL"))
                    logger.info("Discord notifications enabled")
                except Exception as e:
                    logger.warning(f"Failed to initialize Discord notifier: {e}")
                    self.notifier = None
            else:
                logger.info("Discord notifications disabled (dry-run or no webhook)")
                self.notifier = None

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
                        "max_daily_loss": float("inf"),  # type: ignore[dict-item]
                        "max_daily_trades": float("inf"),  # type: ignore[dict-item]
                        "emergency_stop_loss": float("inf"),  # type: ignore[dict-item]
                    }
                )

            # Initialize exchange adapter
            self._init_exchange_adapter()

            # Load and validate model (after coincheck adapter initialization)
            # Initialize model_loading first
            self.algorithm = "ppo"  # Default, will be updated by model_loading
            self.ACTION_NAMES = ACTION_NAMES
            self.model_loading = ModelLoading(self)
            self.model = self._load_model()

            # Initialize PositionManager (Bug #25, #26 fix) - moved after model loading
            self._init_position_manager()

        except Exception as e:
            logger.error(f"Critical error during live trader initialization: {e}")
            logger.error("Initialization failed - trader will not function properly")
            raise RuntimeError(f"Failed to initialize LiveTrader: {e}") from e
        else:
            self.position_manager = None
            logger.warning("PositionManager not available, using legacy position logic")
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
        
        # Initialize schema attributes
        self.schema_available = False
        self.expected_features = 64  # Default
        self.feature_names = None
        
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

    def _init_utility_components(self) -> None:
        """Initialize utility components (cache, rate limiter, circuit breaker)."""
        logger = get_logger(__name__)
        
        # Get configuration values with defaults
        price_cache_ttl = self.config.get('price_cache_ttl', 30.0)
        requests_per_second = self.config.get('requests_per_second', 1.0)
        burst_limit = self.config.get('burst_limit', 5)
        circuit_failure_threshold = self.config.get('circuit_failure_threshold', 5)
        circuit_recovery_timeout = self.config.get('circuit_recovery_timeout', 60.0)
        
        # Price cache for API response caching
        self.price_cache = TTLCache(ttl_seconds=price_cache_ttl)
        
        # Rate limiter for API calls
        rate_limit_config = RateLimitConfig(
            requests_per_second=requests_per_second,
            burst_limit=burst_limit,
            window_seconds=1.0
        )
        self.rate_limiter = TokenBucketRateLimiter(rate_limit_config)
        
        # Circuit breaker for API protection
        circuit_config = CircuitBreakerConfig(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_timeout,
            success_threshold=3,
            timeout=10.0
        )
        self.api_circuit_breaker = CircuitBreaker("coincheck_api", circuit_config)
        logger.info(f"Utility components initialized (Cache TTL: {price_cache_ttl}s, Rate: {requests_per_second}/s, Circuit: {circuit_failure_threshold} failures)")

    def _init_price_history(self) -> None:
        """Initialize price history for live trading."""
        logger = get_logger(__name__)
        
        # Update price history if adapter is available (skip in dry-run)
        if self.exchange_adapter and not self.dry_run:
            self._update_price_history()
        elif self.dry_run:
            logger.info("Dry-run mode: using mock price history")
            # Initialize with mock prices for dry-run
            mock_price = 5000000.0  # Mock BTC/JPY price
            self.price_history.clear()
            self.price_history.extend([mock_price] * self.config.get("price_history_length", 100))
        else:
            logger.info("Price history update deferred until adapter is available")

        # Initialize utility components
        self._init_utility_components()
        
        # Initialize price history
        self._init_price_history()
        
        # Memory optimization: Periodic cleanup counter
        self._cleanup_counter = 0
        self._cleanup_interval = 100  # Clean up every 100 iterations

        # Initialize component modules
        self.trading_loop = TradingLoop(self)
        self.feature_computation = FeatureComputation(self)
        self.action_prediction = ActionPrediction(self)
        self.health_monitoring = HealthMonitoring(self)
        # self.model_loading = ModelLoading(self)  # Already initialized above

        # Send startup notification (with schema info if available)
        feature_info = "68 technical indicators (default)"
        if hasattr(self, 'schema_available') and self.schema_available and self.expected_features:
            feature_info = f"{self.expected_features} features (schema-validated ✅)"
        elif hasattr(self, 'expected_features') and self.expected_features:
            feature_info = f"{self.expected_features} features (detected)"
        
        self._send_notification(
            "🚀 BTC/JPY Live Trading Started",
            f"Model: {self.model_path}\nStrategy: Sell-biased\nMode: {'DEMO' if self.demo_mode else 'LIVE'}\nFeatures: {feature_info}\nTrading Mode: Normal (1M timeframe)\nUtils: RateLimiter+CircuitBreaker+Cache enabled",
            "info",
        )

    def _send_notification(self, title: str, message: str, level: str = "info") -> None:
        """Send notification with error handling."""
        logger = get_logger(__name__)
        logger.debug(f"Sending notification: {title} - {level}")
        if self.notifier:
            try:
                self.notifier.send_notification(title, message, level)
                logger.debug("Notification sent successfully")
            except Exception as e:
                logger.warning(f"Failed to send notification: {e}")
        else:
            # Log to console if Discord is not available
            log_level = getattr(logging, level.upper(), logging.INFO)
            logger.log(log_level, f"{title}: {message}")
            logger.debug("Notification logged to console")

    def set_log_level(self, level: str) -> None:
        """Dynamically adjust logging level for this trader instance."""
        logger = get_logger(__name__)
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        logger.info(f"Log level changed to {level.upper()}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for monitoring."""
        return {
            'position': self.position,
            'total_pnl': self.total_pnl,
            'trades_count': self.trades_count,
            'last_valid_price': self._last_valid_price,
            'demo_mode': getattr(self, 'demo_mode', True),
            'dry_run': getattr(self, 'dry_run', False),
            'exchange_adapter': self.exchange_adapter is not None,
            'position_manager': self.position_manager is not None,
            'price_cache_size': len(getattr(self.price_cache, '_cache', {})) if hasattr(self, 'price_cache') else 0,
            'uptime': (datetime.now() - getattr(self, 'daily_start_time', datetime.now())).total_seconds()
        }


    async def _get_current_price(self) -> float:
        """Get current BTC/JPY price with timeout and error handling."""
        if not self.exchange_adapter:
            raise ValueError("Exchange adapter not initialized")
        
        try:
            # Add timeout to prevent hanging
            price = await asyncio.wait_for(
                self.exchange_adapter.get_current_price("BTC/JPY"),
                timeout=10.0  # 10 second timeout
            )
            if price is None or not isinstance(price, (int, float)) or not np.isfinite(price):
                raise ValueError(f"Invalid price received: {price}")
            self._last_valid_price = price
            return price
        except asyncio.TimeoutError:
            logger = get_logger(__name__)
            logger.error("Timeout getting current price from exchange")
            raise
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Failed to get current price: {e}")
            raise

    def _get_current_price_sync(self) -> float:
        """Synchronous wrapper for async _get_current_price method."""
        try:
            return asyncio.run(self._get_current_price())
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Failed to get current price synchronously: {e}")
            # Fallback to last valid price or mock price
            if self._last_valid_price > 0:
                return self._last_valid_price
            return 5000000.0

    def _compute_features(self) -> NDArray[np.float32]:
        """Compute features for model prediction using full feature engine when available."""
        return self.feature_computation.compute_features()

    def _compute_rsi(self, prices: List[float]) -> float:
        """Compute RSI indicator."""
        try:
            if len(prices) < 2:
                return 50.0
                
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-change)
            
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Validate RSI is in reasonable range
            return max(0.0, min(100.0, rsi))
            
        except Exception as e:
            logger.warning(f"Error computing RSI: {e}")
            return 50.0

    def _predict_action(self, features: NDArray[np.float32]) -> int:
        """Predict trading action using the model."""
        return self.action_prediction.predict_action(features)

    def _execute_action(self, action: int) -> float:
        """
        Execute trading action using PositionManager if available.
        
        Args:
            action: Action to execute (0=HOLD, 1=BUY, 2=SELL)
            
        Returns:
            pnl: PnL from the action
        """
        logger = get_logger(__name__)
        
        if self.position_manager is not None:
            # Use PositionManager for execution
            try:
                pnl = self.position_manager.execute_action(
                    action, 
                    self._current_step,
                    min_holding_period=self.config.get("min_holding_period", 5)
                )
                
                # Sync position state with PositionManager
                self.position = self.position_manager.position
                self.entry_price = self.position_manager.entry_price
                self.total_pnl = self.position_manager.total_pnl
                self.trades_count = self.position_manager.trades_count
                
                logger.info(f"Action executed via PositionManager: {self.ACTION_NAMES.get(action, f'UNKNOWN({action})')}, PnL: {pnl:.2f}")
                return pnl
                
            except Exception as e:
                logger.error(f"Failed to execute action via PositionManager: {e}")
                return 0.0
        else:
            # Legacy execution logic (fallback)
            logger.warning("PositionManager not available, using legacy execution logic")
            
            old_position = self.position
            current_price = self._last_valid_price
            
            if action == 1:  # BUY
                if self.position <= 0:  # Flat or short
                    self.position = 1
                    self.entry_price = current_price
                    self.trades_count += 1
                    logger.info(f"BUY executed: {self.config.get('min_trade_amount', 0.001)} BTC at ¥{current_price:,.0f}")
                    
                    if not self.dry_run:
                        self._send_notification(
                            "📈 Trade Executed",
                            f"BUY {self.config.get('min_trade_amount', 0.001)} BTC\n"
                            f"Price: ¥{current_price:,.0f}\n"
                            f"New Position: long",
                            "success",
                        )
                    
                    return -current_price * self.config.get('transaction_cost', 0.001)  # Entry cost
                    
            elif action == 2:  # SELL
                if self.position >= 0:  # Flat or long
                    self.position = -1
                    self.entry_price = current_price
                    self.trades_count += 1
                    logger.info(f"SELL executed: {self.config.get('min_trade_amount', 0.001)} BTC at ¥{current_price:,.0f}")
                    
                    if not self.dry_run:
                        self._send_notification(
                            "📈 Trade Executed",
                            f"SELL {self.config.get('min_trade_amount', 0.001)} BTC\n"
                            f"Price: ¥{current_price:,.0f}\n"
                            f"New Position: short",
                            "success",
                        )
                    
                    return -current_price * self.config.get('transaction_cost', 0.001)  # Entry cost
            
            # Calculate PnL if position was closed/reversed
            if old_position != self.position and old_position != 0:
                if old_position > 0:  # Closing long position
                    pnl = (current_price - self.entry_price) * self.config.get("min_trade_amount", 0.001)
                else:  # Closing short position
                    pnl = (self.entry_price - current_price) * self.config.get("min_trade_amount", 0.001)
                
                self.total_pnl += pnl
                logger.info(f"Position closed, PnL: {pnl:.2f}, Total PnL: {self.total_pnl:.2f}")
                
                if not self.dry_run:
                    self._send_notification(
                        "📊 Position Update",
                        f"PnL: {pnl:.2f} JPY\nTotal PnL: {self.total_pnl:.2f} JPY\nTrades: {self.trades_count}",
                        "info",
                    )
                
                return pnl
            
            return 0.0  # HOLD or no action

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for monitoring."""
        return self.health_monitoring.get_health_status()

    def _update_price_history(self) -> None:
        """Update cached price history for technical indicators."""
        logger = get_logger(__name__)
        try:
            prices = self._get_historical_prices(
                limit=self.config["price_history_length"]
            )
            self._safe_update_price_history(prices)
        except Exception as e:
            logger.warning(f"Failed to update price history: {e}")
            # Fallback to current price
            current_price = asyncio.run(self._get_current_price())
            self._safe_update_price_history([current_price] * self.config["price_history_length"])
    
    def _safe_update_price_history(self, prices: List[float]) -> None:
        """Safely update price history with None checks."""
        logger = get_logger(__name__)
        if prices and len(prices) > 0:
            # Convert list to deque
            self.price_history.clear()
            self.price_history.extend(prices)
            logger.info(
                f"Updated price history with {len(self.price_history)} data points"
            )
        else:
            logger.warning("No valid prices to update history")
    
    def _archive_price_history(self) -> None:
        """Archive current price history to disk to free memory."""
        if not self.price_history:
            return
        
        # Create DataFrame with timestamps
        timestamps = pd.date_range(end=datetime.now(), periods=len(self.price_history), freq='1min')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': list(self.price_history)
        })
        
        # Save to parquet file
        archive_file = self.archive_dir / f"price_history_{int(time.time())}.parquet"
        df.to_parquet(archive_file, index=False)
        
        logger = get_logger(__name__)
        logger.info(f"Archived price history to {archive_file}")
        
        # Clear memory (though deque has maxlen, this ensures fresh start)
        self.price_history.clear()
    
    def _periodic_cleanup(self) -> None:
        """Perform periodic memory cleanup to prevent accumulation."""
        self._cleanup_counter += 1
        if self._cleanup_counter >= self._cleanup_interval:
            # Clear any accumulated caches that might grow (skip clear_expired in dry-run)
            if not self.dry_run:
                try:
                    self.price_cache.clear_expired()
                except AttributeError:
                    # clear_expired method not available, skip
                    pass

            # Clean up old items from deque (though maxlen should prevent this)
            # deque automatically handles maxlen, so no manual trimming needed

            # Only force garbage collection if memory usage is high
            # Avoid frequent GC as it can impact performance
            try:
                import psutil  # type: ignore[import-untyped]
                import os

                process = psutil.Process(os.getpid())
                memory_percent = process.memory_percent()
                if memory_percent > 80.0:  # Only GC if memory usage > 80%
                    gc.collect()
                    logger = get_logger(__name__)
                    logger.info(f"Periodic cleanup: memory usage was {memory_percent:.1f}%, forced GC")
                else:
                    logger = get_logger(__name__)
                    logger.debug(f"Periodic cleanup: memory usage {memory_percent:.1f}%, no GC needed")
            except ImportError:
                # psutil not available, skip memory check
                pass
            except Exception as e:
                logger = get_logger(__name__)
                logger.debug(f"Memory check failed: {e}")

            # Clear any accumulated references that might prevent GC
            # This helps prevent memory leaks in long-running processes
            if hasattr(self, '_feature_cache'):
                # Clear feature cache if it gets too large
                if len(getattr(self, '_feature_cache', {})) > 1000:
                    self._feature_cache.clear()
                    logger = get_logger(__name__)
                    logger.debug("Cleared feature cache to prevent memory accumulation")

            self._cleanup_counter = 0

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate trading configuration for safety and correctness."""
        logger = get_logger(__name__)

        # Required configuration keys
        required_keys = [
            "price_history_length",
            "max_daily_loss",
            "max_daily_trades",
            "emergency_stop_loss"
        ]

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Required configuration key missing: {key}")

        # Validate numeric ranges
        if config.get("price_history_length", 0) <= 0:
            raise ValueError("price_history_length must be positive")

        if config.get("max_daily_loss", 0) < 0:
            raise ValueError("max_daily_loss cannot be negative")

        if config.get("max_daily_trades", 0) < 0:
            raise ValueError("max_daily_trades cannot be negative")

        if not (0 < config.get("emergency_stop_loss", 0) < 1):
            raise ValueError("emergency_stop_loss must be between 0 and 1")

        # Validate transaction costs are reasonable
        if config.get("transaction_cost", 0) > 0.01:  # More than 1%
            logger.warning(f"High transaction cost detected: {config['transaction_cost']:.4f}")

        # Validate position sizes are reasonable
        max_pos = config.get("max_position_size", 0)
        if max_pos > 1.0:
            raise ValueError("max_position_size cannot exceed 1.0 (100%)")
        elif max_pos > 0.5:
            logger.warning(f"Large position size detected: {max_pos:.2f}")

        logger.debug("Configuration validation passed")

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
            # Action mask parameters
            "min_holding_period": self.ztb_config.get_int(
                "ZTB_MIN_HOLDING_PERIOD", 5
            ),  # Minimum holding period for action masking
            "max_position_age": self.ztb_config.get_int(
                "ZTB_MAX_POSITION_AGE", 1000
            ),  # Maximum position age before forced close
            "allow_reverse": self.ztb_config.get_bool(
                "ZTB_ALLOW_REVERSE", False
            ),  # Allow position reversal
            "enforce_reverse_cooldown": self.ztb_config.get_bool(
                "ZTB_ENFORCE_REVERSE_COOLDOWN", False
            ),  # Enforce cooldown before position reversal
            "initial_portfolio_value": self.ztb_config.get_float(
                "ZTB_INITIAL_PORTFOLIO_VALUE", 200000.0
            ),  # Initial portfolio value for position sizing
        }

    def _load_model(self) -> PPO | MaskablePPO | SAC:
        """Load the trained PPO, MaskablePPO, or SAC model.

        Bug #27 Fix: Now properly loads MaskablePPO models and uses
        ActionMaskProvider for action masking in production.

        Schema Integration: Load schema information for feature validation.
        """
        return self.model_loading.load_model()

        logger = get_logger(__name__)
        logger.info(f"Loading {self.options.algorithm.upper()} model from {self.model_path}")

        if self.options.algorithm.lower() == "sac":
            model = SAC.load(str(self.model_path))
            logger.info("Model loaded as SAC")
            self._is_maskable_ppo = False  # SAC doesn't use masks
            self.algorithm = "sac"
        else:
            # Try loading as MaskablePPO first, fallback to PPO
            try:
                model = MaskablePPO.load(str(self.model_path))
                logger.info("Model loaded as MaskablePPO with action masking support")
                self._is_maskable_ppo = True
                self.algorithm = "ppo"
            except Exception as e:
                logger.info(f"Not a MaskablePPO model ({e}), loading as standard PPO")
                model = PPO.load(str(self.model_path))
                logger.info("Model loaded as standard PPO (no action masking)")
                self._is_maskable_ppo = False
                self.algorithm = "ppo"

        # Log model spaces
        obs_space = model.observation_space
        action_space = model.action_space
        logger.info(f"Model observation space: {obs_space}")
        logger.info(f"Observation shape: {obs_space.shape}")
        logger.info(f"Model action space: {action_space}")
        logger.info(f"Action space type: {type(action_space)}")
        
        # Check if action space is continuous
        try:
            import gymnasium as gym
            Box = gym.spaces.Box
            Discrete = gym.spaces.Discrete
        except ImportError:
            try:
                import gym
                Box = gym.spaces.Box
                Discrete = gym.spaces.Discrete
            except ImportError:
                # Fallback: check by type name
                Box = type(None)
                Discrete = type(None)
                logger.warning("Could not import gym/gymnasium spaces, cannot detect action space type")
        
        if isinstance(action_space, Box):
            self.is_continuous_action = True
            logger.info("Detected continuous action space - will discretize actions")
        elif isinstance(action_space, Discrete):
            self.is_continuous_action = False
            logger.info("Detected discrete action space")
        else:
            self.is_continuous_action = False
            logger.warning(f"Unknown action space type: {type(action_space)} - assuming discrete")

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

            # Skip feature validation during model loading - will be done after adapter initialization
            logger.info("Feature validation deferred until after adapter initialization")

        except Exception as e:
            logger.warning(f"Could not prepare for feature validation: {e}")

        # Send model loaded notification
        self._send_notification(
            "✅ Model Loaded Successfully", f"Model path: {self.model_path}", "success"
        )

        return model

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
        if not self.exchange_adapter:
            logger = get_logger(__name__)
            logger.warning("Exchange adapter not available, cannot fetch balance")
            return {}
        
        try:
            balances = await self.exchange_adapter.get_balance(currency=currency)
            result = {}
            for balance in balances:
                result[balance.currency] = balance.free
            logger = get_logger(__name__)
            logger.info(f"Account balance: {result}")
            return result
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Failed to get account balance: {e}")
            self._send_notification(
                "⚠️ Balance Fetch Error",
                f"Failed to retrieve account balance: {str(e)}",
                "error"
            )
            return {}

    def _setup_metrics(self) -> None:
        """Set up Prometheus metrics for monitoring."""
        if not prometheus_available:
            return

        # Import prometheus classes only when available
        from prometheus_client import Counter, Gauge, Histogram  # type: ignore[import-untyped]

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
                logger = get_logger(__name__)
                logger.warning("Coincheck API returned success=False")
                return [self._get_current_price_sync()] * 14

            trades = data.get("data", [])
            if not trades:
                logger = get_logger(__name__)
                logger.warning("No trade data received from Coincheck")
                return [self._get_current_price_sync()] * 14

            # Extract prices (most recent first, we want oldest first for calculations)
            prices = []
            for trade in trades:
                if isinstance(trade, dict) and "rate" in trade:
                    prices.append(float(trade["rate"]))

            if not prices:
                logger = get_logger(__name__)
                logger.warning("No valid price data in trades")
                return [self._get_current_price_sync()] * 14

            # Reverse to get chronological order (oldest first)
            prices.reverse()
            logger = get_logger(__name__)
            logger.info(f"Successfully fetched {len(prices)} historical prices")
            return prices

        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to get historical prices: {e}, using fallback")
            current_price = self._get_current_price_sync()
            return [current_price] * 14

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:  # type: ignore[no-any-return]
        """Calculate RSI (Relative Strength Index) using existing utility."""
        if len(prices) < self.config["rsi_period"] + 1:
            return self.config["rsi_neutral_value"]  # Configurable neutral RSI

        try:
            # Create DataFrame for compute_rsi
            df = pd.DataFrame({"close": prices})
            rsi_series = cast(pd.Series, compute_rsi(df, period=period))  # type: ignore[name-defined]
            return rsi_value if not rsi_series.empty else neutral_value  # type: ignore[no-any-return]
        except Exception as e:
            logger = get_logger(__name__)
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

            avg_gain = cast(float, sum(gains[-period:]) / period)
            avg_loss = cast(float, sum(losses[-period:]) / period)

            if avg_loss == 0:
                return cast(float, 100.0)  # type: ignore[no-any-return]

            rs = cast(float, avg_gain / avg_loss)
            rsi = cast(float, 100 - (100 / (1 + rs)))
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
            logger = get_logger(__name__)
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
                logger = get_logger(__name__)
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
            logger = get_logger(__name__)
            logger.warning(f"Failed to compute advanced features: {e}")
            return {}

    @timed
    def _get_market_features(self) -> NDArray[np.floating]:
        """Get current market features for model prediction with comprehensive indicators."""
        max_retries = 3
        current_price = 0.0

        # If exchange_adapter is not initialized yet (during model loading), use dummy price
        if self.exchange_adapter is None:
            current_price = self.config[
                "fallback_price"
            ]  # Configurable fallback price for initialization
            logger = get_logger(__name__)
            logger.debug("Using fallback price during initialization")
        else:
            for attempt in range(max_retries):
                try:
                    current_price = self._get_current_price_sync()
                    if current_price > 0:
                        break
                except Exception as e:
                    logger = get_logger(__name__)
                    logger.warning(
                        f"Failed to fetch current price (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(2)

            if current_price <= 0:
                logger = get_logger(__name__)
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
                logger = get_logger(__name__)
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
            logger = get_logger(__name__)
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
                logger = get_logger(__name__)
                logger.debug(f"Using model observation space: {expected_features}")
        else:
            logger = get_logger(__name__)
            logger.debug(f"Using default feature count: {expected_features}")

        # Validate and adjust feature count
        if len(features) < expected_features:
            # Pad with zeros
            padding_needed = expected_features - len(features)
            features.extend([0.0] * padding_needed)
            logger = get_logger(__name__)
            logger.debug(
                f"Padded features to {expected_features} dimensions (added {padding_needed})"
            )
        elif len(features) > expected_features:
            # Truncate
            features = features[:expected_features]
            logger = get_logger(__name__)
            logger.debug(f"Truncated features to {expected_features} dimensions")

        # Schema-based feature order validation (if available)
        if self.schema_available and self.feature_names is not None:
            # TODO: Future enhancement - reorder features based on schema
            # This would ensure features are in the exact order expected by the model
            logger = get_logger(__name__)
            logger.debug(f"Feature schema available with {len(self.feature_names)} named features")

        logger = get_logger(__name__)
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
        sell_bias = cast(float, self.config["sell_bias_multiplier"])

        if action == ACTION_SELL:  # Sell signal
            # Bug #33 Fix: Warmup only restricts SHORT opening, not position closing
            # Allow warmup period before enabling short positions
            sell_warmup_trades = self.config.get("sell_warmup_trades", 2)
            
            # Check if this SELL would OPEN a short position (flat → short)
            if self.position == 0 and self.trades_count < sell_warmup_trades:
                # Only suppress SELL when opening new short during warmup
                logger = get_logger(__name__)
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

    def _execute_trade(self, side: Literal["buy", "sell"], amount: float) -> bool:
        """Execute trade using exchange adapter with enhanced error handling and notifications."""
        logger = get_logger(__name__)
        
        if self.demo_mode:
            logger.info(f"DEMO MODE: Would execute {side} {amount} BTC")
            # Send notification for demo trades
            self._send_notification(
                "📈 Demo Trade Executed",
                f"Side: {side.upper()}\n"
                f"Amount: {amount} BTC\n"
                f"Mode: DEMO (no real trade)",
                "info",
            )
            return True

        # Use exchange adapter for live trading
        if not self.exchange_adapter:
            logger.error("No exchange adapter available for live trading")
            self._send_notification(
                "🚨 CRITICAL: No Exchange Adapter",
                "Exchange adapter not available for live trading",
                "error",
            )
            return False

        try:
            # Execute trade using exchange adapter
            order = self.exchange_adapter.place_order(
                symbol="BTC_JPY",
                side=side,
                quantity=amount,
                order_type="market",  # Use market orders for live trading
                sizing_reason="live_trading",
            )
            
            if order:
                logger.info(f"LIVE TRADE EXECUTED: {side.upper()} {amount} BTC, Order ID: {order.order_id}")
                self._send_notification(
                    "📈 Live Trade Executed",
                    f"Side: {side.upper()}\n"
                    f"Amount: {amount} BTC\n"
                    f"Order ID: {order.order_id}\n"
                    f"Position: {self.position}\n"
                    f"Entry Price: ¥{self.entry_price:,.0f}",
                    "success",
                )
                return True
            else:
                logger.error("Order placement returned None")
                self._send_notification(
                    "⚠️ Trade Failed",
                    f"Order placement failed for {side.upper()} {amount} BTC",
                    "warning",
                )
                return False
                
        except InsufficientFundsError as e:
            # Handle insufficient funds gracefully
            error_msg = str(e)
            logger.warning(f"Trade failed due to insufficient funds: {error_msg}")
            self._send_notification(
                "⚠️ Insufficient Funds",
                f"Side: {side.upper()}\n"
                f"Amount: {amount} BTC\n"
                f"Error: {error_msg}\n"
                f"Trading will continue",
                "warning",
            )
            return False
        except MinimumSizeError as e:
            # Handle minimum size errors gracefully
            error_msg = str(e)
            logger.warning(f"Trade failed due to minimum size requirements: {error_msg}")
            self._send_notification(
                "⚠️ Order Too Small",
                f"Side: {side.upper()}\n"
                f"Amount: {amount} BTC\n"
                f"Error: {error_msg}\n"
                f"Trading will continue",
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
                    success = self._execute_trade("buy", self.config["min_trade_amount"])
                    if success:
                        self._send_notification(
                            "📈 Trade Executed",
                            f"BUY {self.config['min_trade_amount']} BTC\n"
                            f"Price: ¥{current_price:,.0f}\n"
                            f"New Position: {position_str}",
                            "success",
                        )
                elif action == ACTION_SELL:
                    success = self._execute_trade("sell", self.config["min_trade_amount"])
                    if success:
                        self._send_notification(
                            "📈 Trade Executed",
                            f"SELL {self.config['min_trade_amount']} BTC\n"
                            f"Price: ¥{current_price:,.0f}\n"
                            f"New Position: {position_str}",
                            "success",
                        )
            
            # Bug #29 fix: Always sync realized_pnl, even when trade_pnl is 0
            # (Opening positions have 0 trade_pnl but negative entry fees)
            old_total_pnl = self.total_pnl
            self.total_pnl = self.position_manager.realized_pnl
            pnl_change = self.total_pnl - old_total_pnl
            
            # Validate PnL if it changed
            if pnl_change != 0.0:
                # Validate PnL calculation (Reviewer B recommendation)
                if not np.isfinite(self.total_pnl):
                    logger = get_logger(__name__)
                    logger.error(
                        f"Invalid PnL calculation: {self.total_pnl}. Reverting to previous value."
                    )
                    self.total_pnl = old_total_pnl
                    pnl_change = 0.0
                
                # Sanity check: PnL change shouldn't exceed 10x estimated portfolio value
                # Estimate portfolio as 1M JPY base + total accumulated PnL
                estimated_portfolio = 1_000_000.0 + old_total_pnl
                if abs(pnl_change) > estimated_portfolio * 10:
                    logger = get_logger(__name__)
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
            logger = get_logger(__name__)
            logger.warning("Using legacy position management (PositionManager not available)")
            old_position = self.position

            # Handle position changes based on action and current position
            if action == ACTION_BUY:
                if self.position <= 0:  # Enter long position or reverse from short
                    self.position = 1
                    self.entry_price = current_price
                    self.trades_count += 1
                    self.daily_trades += 1
                    success = self._execute_trade("buy", self.config["min_trade_amount"])
                    if success:
                        self._send_notification(
                            "📈 Trade Executed",
                            f"BUY {self.config['min_trade_amount']} BTC\n"
                            f"Price: ¥{current_price:,.0f}\n"
                            f"New Position: long",
                            "success",
                        )

            elif action == ACTION_SELL:
                if self.position >= 0:  # Enter short position or reverse from long
                    self.position = -1
                    self.entry_price = current_price
                    self.trades_count += 1
                    self.daily_trades += 1
                    success = self._execute_trade("sell", self.config["min_trade_amount"])
                    if success:
                        self._send_notification(
                            "📈 Trade Executed",
                            f"SELL {self.config['min_trade_amount']} BTC\n"
                            f"Price: ¥{current_price:,.0f}\n"
                            f"New Position: short",
                            "success",
                        )

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

    def _setup_archive_config(self) -> None:
        """Setup archive configuration."""
        self.archive_counter = 0
        self.archive_interval = 1000
        self.archive_dir = Path("data/archives")
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def _setup_model_options(self, model_path: Union[str, Path, LiveTradingOptions], disable_risk_limits: bool) -> None:
        """Setup model options."""
        if isinstance(model_path, LiveTradingOptions):
            self.options = model_path
            self.model_path = Path(self.options.model_path).expanduser()
        else:
            self.model_path = Path(model_path).expanduser()
            self.options = LiveTradingOptions(
                model_path=self.model_path,
                algorithm="sac" if "sac" in str(self.model_path).lower() else "ppo",
                venue="coincheck",
                disable_risk_limits=disable_risk_limits,
                dry_run=False,
            )

    def _validate_model_path(self) -> None:
        """Validate model path exists."""
        if not self.model_path.exists():
            raise ValueError(f"Model file not found: {self.model_path}")

    def _setup_config(self, config: Optional[Dict[str, Any]]) -> None:
        """Setup configuration."""
        logger = get_logger(__name__)
        self.ztb_config = ZTBConfig()
        logger.debug("ZTBConfig initialized")
        self.config = config or self._get_default_config()

        # Validate configuration for safety
        self._validate_config(self.config)

        self.dry_run = self.options.dry_run
        self.disable_risk_limits = self.options.disable_risk_limits

    def _init_core_components(self) -> None:
        """Initialize core components."""
        logger = get_logger(__name__)
        self.algorithm = self.options.algorithm
        self.ACTION_NAMES = ACTION_NAMES
        self.price_history: deque[float] = deque(maxlen=100)
        self.expected_features = 64
        self.feature_names = None
        self.schema_available = False

        # Initialize essential attributes
        self.price_cache = TTLCache(ttl_seconds=30.0)
        self._last_valid_price = 0.0
        self.total_pnl = 0.0
        self.position = 0
        self.entry_price = 0.0
        self.trades_count = 0
        self.daily_start_pnl = 0.0
        self.daily_trades = 0
        self.notifier = None
        self._current_step = 0
        self._cleanup_counter = 0
        self._cleanup_interval = 100

    def _init_trading_components(self) -> None:
        """Initialize trading components."""
        # Initialize exchange adapter
        self._init_exchange_adapter()

        # Initialize position manager
        self._init_position_manager()

        # Initialize action mask provider
        self._init_action_mask_provider()

        # Initialize trading components
        self.trading_loop = TradingLoop(self)
        self.feature_computation = FeatureComputation(self)
        self.action_prediction = ActionPrediction(self)

    def _init_monitoring_components(self) -> None:
        """Initialize monitoring components."""
        self.health_monitoring = HealthMonitoring(self)
        self.model_loading = ModelLoading(self)

        # Load model
        self.model = self.model_loading.load_model()

    def _init_exchange_adapter(self) -> None:
        """Initialize exchange adapter."""
        logger = get_logger(__name__)
        venue = self.options.venue.lower()
        if venue == "coincheck":
            self.api_key, self.api_secret = get_coincheck_credentials()
            self.base_url = "https://coincheck.com"
            adapter_name = "coincheck"
        else:
            raise ValueError(f"Unsupported venue: {venue}")

        try:
            broker_registry = get_broker_registry()
            self.exchange_adapter = broker_registry.get_broker(
                adapter_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                dry_run=self.dry_run,
            )
            logger.info(f"{venue.upper()} adapter initialized")
        except Exception as e:
            logger.error(f"Failed to initialize {venue.upper()} adapter: {e}")
            self.exchange_adapter = None
            # Don't re-raise - allow initialization to continue with None adapter

    def _init_position_manager(self) -> None:
        """Initialize position manager."""
        logger = get_logger(__name__)
        if position_manager_available:
            try:
                position_config = self._create_position_config()
                self.position_manager = PositionManager(
                    config=position_config,
                    get_price_callback=self._get_current_price_callback,
                )
                logger.info(f"PositionManager initialized (position_size={position_config.max_position_size})")
            except Exception as e:
                logger.warning(f"Failed to initialize PositionManager: {e}")
                self.position_manager = None
        else:
            self.position_manager = None
            logger.warning("PositionManager not available, using legacy position logic")

    def _create_position_config(self) -> Any:
        """Create position configuration."""
        class LivePositionConfig:
            """Configuration for PositionManager in live trading."""
            def __init__(self, config_dict: Dict[str, Any]) -> None:
                self.allow_reverse = config_dict.get("allow_reverse", False)
                self.transaction_cost = config_dict.get("transaction_cost", 0.001)
                self.max_position_size = config_dict.get(
                    "max_position_size",
                    config_dict.get("min_trade_amount", 0.001)
                )
                self.enforce_reverse_cooldown = config_dict.get("enforce_reverse_cooldown", False)
                self.initial_portfolio_value = config_dict.get("initial_portfolio_value", 200000.0)

        return LivePositionConfig(self.config)

    def _get_current_price_callback(self) -> float:
        """Get current price callback for position manager."""
        return self._last_valid_price if self._last_valid_price > 0 else 5000000.0

    def _init_action_mask_provider(self) -> None:
        """Initialize action mask provider."""
        mask_config = ActionMaskConfig(
            min_holding_period=self.config.get("min_holding_period", 5),
            enable_forced_close=True,
            max_position_age=self.config.get("max_position_age", 1000)
        )
        self.mask_provider = ActionMaskProvider(mask_config)
        self._is_maskable_ppo = False

    async def run_trading_loop(self, duration_hours: float) -> None:
        """
        Run the main trading loop for live trading.
        
        Args:
            duration_hours: Duration to run trading in hours
        """
        logger = get_logger(__name__)
        
        # Display initial balance
        try:
            initial_balance = await self.get_account_balance()
            logger.info("💰 Initial account balance:")
            for currency, amount in initial_balance.items():
                logger.info(f"   {currency}: {amount:,.2f}")
        except Exception as e:
            logger.warning(f"Could not fetch initial balance: {e}")
        
        # Run trading loop
        self.trading_loop.run_trading_loop(duration_hours)
        
        # Display final balance
        try:
            final_balance = await self.get_account_balance()
            logger.info("💰 Final account balance:")
            for currency, amount in final_balance.items():
                logger.info(f"   {currency}: {amount:,.2f}")
        except Exception as e:
            logger.warning(f"Could not fetch final balance: {e}")