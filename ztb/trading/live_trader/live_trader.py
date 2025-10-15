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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

# Add project root to path
from ztb.utils.path_utils import get_project_root
project_root = get_project_root()
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO, SAC

from ztb.trading.live.action_mask_provider import (
    ActionMaskConfig,
    ActionMaskProvider,
)
from ztb.trading.live_trader.config import LiveTradingOptions, prometheus_available
from ztb.trading.live.registry.broker_registry import get_broker_registry
from ztb.utils.logging_utils import get_logger

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
    normalize_action,
)


class LiveTrader:
    """
    Live trading bot for BTC/JPY using trained PPO model.

    If COINCHECK_API_KEY and COINCHECK_API_SECRET are not set, the bot runs in demo mode and does not execute real trades.
    """

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
        if isinstance(model_path, LiveTradingOptions):
            options = model_path
            self.model_path = Path(options.model_path).expanduser()
            disable_risk_limits = options.disable_risk_limits
            dry_run = options.dry_run
        else:
            self.model_path = Path(model_path).expanduser()
            options = LiveTradingOptions(
                model_path=self.model_path,
                algorithm="ppo",  # default
                venue="coincheck",  # default
                disable_risk_limits=disable_risk_limits,
                dry_run=dry_run,
            )

        self.options = options
        if not self.model_path.exists():
            raise ValueError(f"Model file not found: {self.model_path}")
        self.ztb_config = ZTBConfig()
        logger.debug("ZTBConfig initialized")
        self.config = config or self._get_default_config()
        logger.debug(f"Trading config loaded: {self.config}")
        self.disable_risk_limits = options.disable_risk_limits
        self.dry_run = options.dry_run
        self.notifier: Optional[DiscordNotifier] = None

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

        # Exchange API settings based on venue
        venue = self.options.venue.lower()
        if venue == "coincheck":
            self.api_key = os.getenv("COINCHECK_API_KEY", "").strip()
            self.api_secret = os.getenv("COINCHECK_API_SECRET", "").strip()
            self.base_url = "https://coincheck.com"
            adapter_name = "coincheck"
        elif venue == "bitflyer":
            self.api_key = os.getenv("BITFLYER_API_KEY", "").strip()
            self.api_secret = os.getenv("BITFLYER_API_SECRET", "").strip()
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

        # Load and validate model (after coincheck adapter initialization)
        self.model = self._load_model()

        # Initialize PositionManager (Bug #25, #26 fix) - moved after model loading
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

        # Initialize utility components
        # Price cache for API response caching (TTL: 30 seconds)
        self.price_cache = TTLCache(ttl_seconds=30.0)
        
        # Rate limiter for API calls (1 request per second, burst limit 5)
        rate_limit_config = RateLimitConfig(
            requests_per_second=1.0,
            burst_limit=5,
            window_seconds=1.0
        )
        self.rate_limiter = TokenBucketRateLimiter(rate_limit_config)
        
        # Circuit breaker for API protection (5 failures -> open, 60s recovery)
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=3,
            timeout=10.0
        )
        self.api_circuit_breaker = CircuitBreaker("coincheck_api", circuit_config)
        
        # Update price history if adapter is available
        if self.exchange_adapter:
            self._update_price_history()
        else:
            logger.info("Price history update deferred until adapter is available")
        
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
            f"Model: {model_path}\nStrategy: Sell-biased\nMode: {'DEMO' if self.demo_mode else 'LIVE'}\nFeatures: {feature_info}\nTrading Mode: Normal (1M timeframe)\nUtils: RateLimiter+CircuitBreaker+Cache enabled",
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

    async def run_trading_loop(self, duration_hours: float) -> None:
        """Run the main trading loop for live trading."""
        from datetime import datetime, timedelta
        import time

        logger = get_logger(__name__)
        logger.info(f"🚀 Starting live trading loop for {duration_hours} hours")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        iteration_count = 0
        
        logger.debug("Entering trading loop")
        while datetime.now() < end_time:
            iteration_count += 1
            logger.debug(f"Starting iteration {iteration_count}")
            
            with PerformanceMonitor(f"trading_iteration_{iteration_count}"):
                try:
                    # Get current price
                    try:
                        current_price = await self._get_current_price()
                        logger.info(f"📈 Price update #{iteration_count}: ¥{current_price:,.0f}")
                    except Exception as e:
                        logger.error(f"Failed to get current price: {e}")
                        if self._last_valid_price > 0:
                            current_price = self._last_valid_price
                            logger.warning(f"Using last valid price: ¥{current_price:,.0f}")
                        else:
                            logger.error("No valid price available, skipping iteration")
                            continue
                    
                    # Update price history
                    self._update_price_history()
                    
                    # Compute features for prediction
                    logger.debug("Computing features...")
                    try:
                        features = self._compute_features()
                        logger.debug(f"Features computed: {len(features)} features")
                    except Exception as e:
                        logger.error(f"Failed to compute features: {e}")
                        logger.warning("Using zero features as fallback")
                        features = np.zeros(64, dtype=np.float32)
                    
                    # Predict action
                    logger.debug("Predicting action...")
                    try:
                        action = self._predict_action(features)
                        action_name = ACTION_NAMES.get(action, f"UNKNOWN({action})")
                        logger.debug(f"Predicted action: {action_name}")
                    except Exception as e:
                        logger.error(f"Failed to predict action: {e}")
                        action = ACTION_HOLD
                        action_name = "HOLD (fallback)"
                    
                    # Validate position before executing action
                    if not (-1 <= self.position <= 1):
                        logger.error(f"Invalid position detected: {self.position}, resetting to 0")
                        self.position = 0.0
                    
                    # Execute action
                    logger.debug("Executing action...")
                    pnl = self._execute_action(action)
                    logger.debug(f"Action executed, PnL: {pnl}")
                    
                    # Send periodic notification (every 10 iterations or significant events)
                    if iteration_count % 10 == 0 or action != ACTION_HOLD:
                        self._send_notification(
                            f"📊 Trading Update #{iteration_count}",
                            f"Price: ¥{current_price:,.0f}\nAction: {action_name}\nPnL: ¥{pnl:,.2f}\nPosition: {self.position:.4f} BTC",
                            "info" if action == ACTION_HOLD else "success"
                        )
                        logger.debug("Notification sent for iteration")
                    
                    # Periodic cleanup
                    self._periodic_cleanup()
                    
                except Exception as e:
                    logger.error(f"❌ Error in trading loop iteration {iteration_count}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    print(f"Traceback: {traceback.format_exc()}")
                    self._send_notification("⚠️ Trading Error", f"Error in iteration {iteration_count}: {e}", "error")
            
            # Wait before next iteration (1 minute)
            time.sleep(60)
        
        # Final report
        total_pnl = self.total_pnl
        trades_count = self.trades_count
        
        logger.info(f"🏁 Trading loop completed after {duration_hours} hours")
        logger.info(f"   Total PnL: ¥{total_pnl:,.2f}")
        logger.info(f"   Total trades: {trades_count}")
        
        self._send_notification(
            "🏁 Trading Session Complete",
            f"Duration: {duration_hours} hours\nTotal PnL: ¥{total_pnl:,.2f}\nTrades: {trades_count}\nFinal Position: {self.position:.4f} BTC",
            "success"
        )

    async def _get_current_price(self) -> float:
        """Get current BTC/JPY price from exchange with rate limiting, circuit breaker, and caching."""
        # Check cache first
        cache_key = "btc_jpy_price"
        cached_price = self.price_cache.get(cache_key)
        if cached_price is not None:
            logger.debug(f"Using cached price: ¥{cached_price:,.0f}")
            return cached_price

        if self.exchange_adapter:
            try:
                # Apply rate limiting
                if not await self.rate_limiter.acquire():
                    logger.warning("Rate limit exceeded, using last valid price")
                    if self._last_valid_price > 0:
                        return self._last_valid_price
                    else:
                        raise ValueError("Rate limited and no cached price available")

                # Apply circuit breaker
                price = await self.api_circuit_breaker.call(
                    self.exchange_adapter.get_current_price, "btc_jpy"
                )
                self._last_valid_price = price
                
                # Cache the result
                self.price_cache.set(cache_key, price)
                
                return price

            except Exception as e:
                logger.warning(f"Failed to get price from adapter: {e}")
                if self._last_valid_price > 0:
                    logger.info(f"Using cached price: ¥{self._last_valid_price:,.0f}")
                    return self._last_valid_price
                else:
                    raise ValueError("No valid price available")
        else:
            raise ValueError("Exchange adapter not available")

    def _compute_features(self) -> NDArray[np.float32]:
        """Compute features for model prediction."""
        try:
            # Use cached price history
            prices = list(self.price_history)
            
            if len(prices) < 10:
                logger.warning(f"Insufficient price history: {len(prices)} points")
                # Pad with current price if needed
                current_price = self._last_valid_price if self._last_valid_price > 0 else 5000000.0
                while len(prices) < 10:
                    prices.append(current_price)
            
            # Validate prices are reasonable
            prices = [max(1000.0, min(10000000.0, p)) for p in prices]  # Clamp to reasonable range
            
            # Compute basic features (simplified)
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
                rsi = self._compute_rsi(prices[-14:])
                features.append(rsi / 100.0 - 0.5)  # Normalize around 0
            else:
                features.append(0.0)
            
            # Pad to expected feature count
            expected_features = getattr(self, 'expected_features', 64) or 64
            while len(features) < expected_features:
                features.append(0.0)
            
            features = features[:expected_features]
            
            # Validate features are finite
            features = [0.0 if not np.isfinite(f) else f for f in features]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error in _compute_features: {e}")
            # Return zero features as safe fallback
            expected_features = getattr(self, 'expected_features', 64) or 64
            return np.zeros(expected_features, dtype=np.float32)

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
        logger = get_logger(__name__)
        try:
            # Handle different observation spaces for different algorithms
            if hasattr(self, 'algorithm') and self.algorithm == 'sac':
                # SAC expects 5 features, take first 5 or pad if needed
                if len(features) >= 5:
                    obs_features = features[:5]
                else:
                    obs_features = np.pad(features, (0, 5 - len(features)), 'constant')
                logger.debug(f"Using first 5 features for SAC: {obs_features}")
            else:
                # PPO uses all features
                obs_features = features
            
            # Reshape for model input
            obs = obs_features.reshape(1, -1)
            
            if self._is_maskable_ppo:
                # Update mask provider state
                self.mask_provider.update_state(
                    current_position=self.position,
                    position_entry_step=self._position_entry_step,
                    current_step=self._current_step,
                    forced_close_reason=None
                )
                # Use action masking
                action_masks = self.mask_provider.get_action_mask()
                action, _ = self.model.predict(obs, action_masks=action_masks)
            else:
                # Standard prediction
                action, _ = self.model.predict(obs)
            
            logger.debug(f"Model prediction result: {action}, type: {type(action)}, shape: {getattr(action, 'shape', 'no shape')}")
            
            # Handle different action formats and spaces
            if hasattr(self, 'is_continuous_action') and self.is_continuous_action:
                # Continuous action space - discretize to [0,1,2]
                if isinstance(action, (int, np.integer)):
                    action_val = float(action)
                elif isinstance(action, (float, np.floating)):
                    action_val = float(action)
                elif isinstance(action, np.ndarray):
                    if action.ndim == 0:
                        action_val = float(action.item())
                    elif action.ndim == 1 and len(action) == 1:
                        action_val = float(action[0])
                    elif action.ndim == 2 and action.shape == (1, 1):
                        # SAC often returns [[value]] format
                        action_val = float(action[0][0])
                        logger.debug(f"SAC action format [[{action_val}]] detected")
                    else:
                        logger.warning(f"Unexpected continuous action format: {action}")
                        action_val = 0.0
                else:
                    logger.warning(f"Unknown continuous action type: {type(action)}, value: {action}")
                    action_val = 0.0
                
                logger.debug(f"Continuous action value: {action_val}")
                logger.info(f"SAC model output: {action_val:.4f} (raw continuous action)")
                
                # Discretize: negative = SELL, zero = HOLD, positive = BUY
                # SAC continuous action mapping: negative values = SELL intensity, 0 = HOLD, positive = BUY intensity
                # Use the same threshold as the environment
                from ztb.trading.environment.constants import CONTINUOUS_TO_DISCRETE_THRESHOLD
                threshold = CONTINUOUS_TO_DISCRETE_THRESHOLD
                
                if action_val > threshold:
                    final_action = ACTION_BUY   # 1
                    logger.info(f"SAC action discretized: {action_val:.4f} -> BUY (1)")
                elif action_val < -threshold:
                    final_action = ACTION_SELL  # 2
                    logger.info(f"SAC action discretized: {action_val:.4f} -> SELL (2)")
                else:
                    final_action = ACTION_HOLD  # 0
                    logger.info(f"SAC action discretized: {action_val:.4f} -> HOLD (0)")
                    
            else:
                # Discrete action space
                if isinstance(action, (int, np.integer)):
                    final_action = int(action)
                elif isinstance(action, (float, np.floating)):
                    final_action = int(action)
                elif isinstance(action, np.ndarray):
                    if action.ndim == 0:
                        final_action = int(action.item())
                    elif action.ndim == 1:
                        if len(action) == 1:
                            final_action = int(action[0])
                        else:
                            # Probability distribution
                            logger.debug(f"Treating as probability distribution: {action}")
                            final_action = int(np.argmax(action))
                    else:
                        logger.debug(f"Multi-dimensional action array, flattening: {action}")
                        final_action = int(np.argmax(action.flatten()))
                else:
                    logger.warning(f"Unknown discrete action type: {type(action)}, value: {action}")
                    try:
                        final_action = int(action)
                    except (ValueError, TypeError):
                        logger.error(f"Cannot convert action {action} to int, using HOLD")
                        final_action = ACTION_HOLD
            
            logger.debug(f"Converted action: {final_action}")
            
            # Clamp action to valid range [0, 1, 2]
            if final_action < 0:
                logger.warning(f"Action {final_action} is negative, clamping to 0")
                final_action = 0
            elif final_action > 2:
                logger.warning(f"Action {final_action} is > 2, clamping to 2")
                final_action = 2
            
            # Additional validation (should be 0, 1, or 2 now)
            if final_action not in [ACTION_HOLD, ACTION_BUY, ACTION_SELL]:
                logger.error(f"Action {final_action} still invalid after clamping, using HOLD")
                final_action = ACTION_HOLD
            
            # Legacy support: normalize old ACTION_SELL=2 to new ACTION_SELL=-1
            final_action = normalize_action(final_action)
            
            action = final_action
            logger.debug(f"Final validated action: {action}")
            
            self._current_step += 1
            
            return action
            
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Failed to predict action: {e}")
            return ACTION_HOLD  # Safe fallback

    def _execute_action(self, action: int) -> float:
        """Execute trading action and return PnL."""
        logger = get_logger(__name__)
        
        # Validate action
        if action not in [ACTION_HOLD, ACTION_BUY, ACTION_SELL]:
            logger.error(f"Invalid action {action} passed to _execute_action, defaulting to HOLD")
            action = ACTION_HOLD
        
        if not self.position_manager:
            logger.warning("PositionManager not available, skipping action execution")
            return 0.0
            
        try:
            # Debug logging
            logger.debug(f"Executing action {action}, current_step={self._current_step}")
            if self.mask_provider:
                logger.debug(f"mask_provider.config.min_holding_period={self.mask_provider.config.min_holding_period}")
            else:
                logger.warning("mask_provider is None!")
            
            # Execute action through PositionManager
            pnl = self.position_manager.execute_action(
                action=action,
                current_step=self._current_step,
                min_holding_period=self.mask_provider.config.min_holding_period if self.mask_provider else 0
            )
            
            # Update local state
            self.position = self.position_manager.position
            self.entry_price = self.position_manager.entry_price
            self.total_pnl = self.position_manager.total_pnl
            self.trades_count = self.position_manager.trades_count
            
            # Update position entry step for masking
            if action != ACTION_HOLD:
                self._position_entry_step = self._current_step
            
            return pnl
            
        except Exception as e:
            logger.error(f"Failed to execute action {action}: {e}")
            return 0.0

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for monitoring."""
        return {
            "status": "healthy",
            "position": self.position,
            "total_pnl": self.total_pnl,
            "trades_count": self.trades_count,
            "last_price": self._last_valid_price,
            "model_loaded": self.model is not None,
            "adapter_available": self.exchange_adapter is not None,
        }

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
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

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
                return [asyncio.run(self._get_current_price())] * 14

            trades = data.get("data", [])
            if not trades:
                logger = get_logger(__name__)
                logger.warning("No trade data received from Coincheck")
                return [asyncio.run(self._get_current_price())] * 14

            # Extract prices (most recent first, we want oldest first for calculations)
            prices = []
            for trade in trades:
                if isinstance(trade, dict) and "rate" in trade:
                    prices.append(float(trade["rate"]))

            if not prices:
                logger = get_logger(__name__)
                logger.warning("No valid price data in trades")
                return [asyncio.run(self._get_current_price())] * 14

            # Reverse to get chronological order (oldest first)
            prices.reverse()
            logger = get_logger(__name__)
            logger.info(f"Successfully fetched {len(prices)} historical prices")
            return prices

        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to get historical prices: {e}, using fallback")
            current_price = asyncio.run(self._get_current_price())
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
                    current_price = asyncio.run(self._get_current_price())
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

    def _execute_trade(self, side: str, amount: float) -> bool:
        """Execute trade on Coincheck with enhanced error handling and notifications."""
        if self.demo_mode:
            logger = get_logger(__name__)
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

        # Enhanced error notification for live trading
        try:
            # TODO: Implement actual Coincheck API trading calls
            logger = get_logger(__name__)
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
            # Still send trade info notification even though execution is not implemented
            self._send_notification(
                "📈 Live Trade Info",
                f"Side: {side.upper()}\n"
                f"Amount: {amount} BTC\n"
                f"Position: {self.position}\n"
                f"Entry Price: ¥{self.entry_price:,.0f}",
                "info",
            )
            return False
        except Exception as e:
            # Critical error notification
            error_msg = f"CRITICAL: Trade execution failed - {str(e)}"
            logger = get_logger(__name__)
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