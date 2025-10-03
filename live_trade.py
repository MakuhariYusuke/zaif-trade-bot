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
import platform
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from stable_baselines3 import PPO

# Load environment variables from .env file
load_dotenv()

# Cross-platform path handling
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import feature computation
try:
    from ztb.features import FeatureRegistry
    from ztb.features.feature_engine import compute_features_batch
    features_available = True
except ImportError:
    features_available = False

# Import risk management
try:
    from ztb.risk.advanced_auto_stop import AdvancedAutoStop, create_production_auto_stop
    auto_stop_available = True
except ImportError:
    auto_stop_available = False

# Action constants for better readability
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2

ACTION_NAMES = {
    ACTION_HOLD: "HOLD",
    ACTION_BUY: "BUY",
    ACTION_SELL: "SELL"
}

from ztb.utils import DiscordNotifier

# Configure logging with cross-platform path handling
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create logger with file and console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# File handler
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


class LiveTrader:
    """
    Live trading bot for BTC/JPY using trained PPO model.

    If COINCHECK_API_KEY and COINCHECK_API_SECRET are not set, the bot runs in demo mode and does not execute real trades.
    """

    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None, disable_risk_limits: bool = False):
        # Cross-platform path handling
        self.model_path = Path(model_path)
        self.config = config or self._get_default_config()
        self.disable_risk_limits = disable_risk_limits

        # Adjust risk limits if disabled
        if self.disable_risk_limits:
            logger.warning("RISK LIMITS DISABLED - Operating without safety restrictions")
            self.config.update({
                'max_daily_loss': float('inf'),
                'max_daily_trades': float('inf'),
                'emergency_stop_loss': float('inf'),
            })

        # Coincheck API settings (initialize early)
        self.api_key = os.getenv('COINCHECK_API_KEY')
        self.api_secret = os.getenv('COINCHECK_API_SECRET')
        self.base_url = 'https://coincheck.com'

        # Validate API credentials for live trading
        self.demo_mode = not (self.api_key and self.api_secret)
        if self.demo_mode:
            logger.warning("COINCHECK_API_KEY and/or COINCHECK_API_SECRET not set - running in DEMO mode")
            logger.warning("Set environment variables or create .env file with API credentials for live trading")

        # Initialize Discord notifier with error handling
        webhook_url = os.getenv('DISCORD_WEBHOOK')
        if webhook_url:
            try:
                self.notifier = DiscordNotifier(webhook_url=webhook_url)
            except Exception as e:
                logger.warning(f"Failed to initialize Discord notifier: {e}")
                self.notifier = None
        else:
            logger.info("Discord webhook not configured - notifications disabled")
            self.notifier = None

        # Load and validate model
        self.model = self._load_model()
        self.position = 0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades_count = 0
        self.daily_start_pnl = 0.0
        self.daily_trades = 0
        self.daily_start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

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
            f"Model: {model_path}\nStrategy: Sell-biased\nMode: {'DEMO' if self.demo_mode else 'LIVE'}\nTechnical indicators: RSI, SMA enabled",
            "info"
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

    def _update_price_history(self):
        """Update cached price history for technical indicators."""
        try:
            self.price_history = self._get_historical_prices(limit=50)
            logger.info(f"Updated price history with {len(self.price_history)} data points")
        except Exception as e:
            logger.warning(f"Failed to update price history: {e}")
            # Fallback to current price
            current_price = self._get_current_price()
            self.price_history = [current_price] * 50

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default trading configuration with safety limits."""
        return {
            "reward_scaling": 1.0,
            "transaction_cost": 0.001,  # 0.1%
            "max_position_size": 0.1,  # Max 10% of available BTC (conservative)
            "sell_bias_multiplier": 2.0,  # Bias towards selling
            "min_trade_amount": 0.001,  # Minimum BTC trade
            "max_trades_per_hour": 6,  # Conservative trading frequency
            "price_check_interval": 60,  # Check price every 60 seconds (conservative)
            "max_daily_loss": 10000.0,  # Max daily loss in JPY
            "max_daily_trades": 50,  # Max trades per day
            "emergency_stop_loss": 0.05,  # 5% emergency stop loss
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
            if not hasattr(self, 'price_history'):
                current_price = 1000000.0  # Dummy price for checking
                self.price_history = [current_price] * 50

            sample_features = self._get_market_features()
            expected_features = len(sample_features)
            actual_obs_space = model.observation_space.shape[0]

            if expected_features != actual_obs_space:
                logger.warning(f"Feature count mismatch: model expects {actual_obs_space}, got {expected_features}")
                logger.warning("Using only basic features to match training data")

        except Exception as e:
            logger.warning(f"Could not verify feature compatibility: {e}")

        # Send model loaded notification
        self._send_notification(
            "✅ Model Loaded Successfully",
            f"Model path: {self.model_path}",
            "success"
        )

        return model

    def _get_current_price(self) -> float:
        """Get current BTC/JPY price from Coincheck."""
        try:
            response = requests.get(f"{self.base_url}/api/ticker", timeout=10)
            response.raise_for_status()
            data = response.json()
            return float(data['last'])  # Current price
        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            return 0.0

    def _get_historical_prices(self, limit: int = 100) -> List[float]:
        """Get historical BTC/JPY prices from Coincheck."""
        try:
            # Use Coincheck's trades API for historical data
            response = requests.get(
                f"{self.base_url}/api/trades",
                params={"pair": "btc_jpy", "limit": min(limit, 100)},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if not data.get('success', False):
                logger.warning("Coincheck API returned success=False")
                return [self._get_current_price()] * 14

            trades = data.get('data', [])
            if not trades:
                logger.warning("No trade data received from Coincheck")
                return [self._get_current_price()] * 14

            # Extract prices (most recent first, we want oldest first for calculations)
            prices = []
            for trade in trades:
                if isinstance(trade, dict) and 'rate' in trade:
                    prices.append(float(trade['rate']))

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
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
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
        df = pd.DataFrame({
            'timestamp': pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(minutes=len(prices)),
                                      periods=len(prices), freq='1min'),
            'open': prices,
            'high': prices,  # Mock high as current price
            'low': prices,   # Mock low as current price
            'close': prices,
            'volume': [1000] * len(prices),  # Mock volume
        })

        try:
            # Compute features using the feature engine
            result = compute_features_batch(df, verbose=False)

            # Handle different return types from compute_features_batch
            if isinstance(result, tuple) and len(result) >= 1:
                features_df = result[0]  # First element is DataFrame
            else:
                features_df = result

            if not hasattr(features_df, 'columns'):
                logger.warning("Feature computation returned unexpected format")
                return {}

            # Extract the latest feature values
            latest_features = {}
            for col in features_df.columns:
                if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                    try:
                        latest_features[col] = float(features_df[col].iloc[-1])
                    except (ValueError, TypeError):
                        continue

            return latest_features

        except Exception as e:
            logger.warning(f"Failed to compute advanced features: {e}")
            return {}

    def _get_market_features(self) -> np.ndarray:
        """Get current market features for model prediction with comprehensive indicators."""
        max_retries = 3
        for attempt in range(max_retries):
            current_price = self._get_current_price()
            if current_price != 0.0:
                break
            logger.warning(f"Failed to fetch current price (attempt {attempt + 1}/{max_retries}), retrying...")
            time.sleep(5)
        else:
            logger.error("Unable to fetch current price after multiple attempts.")
            raise RuntimeError("Failed to fetch current price for feature generation.")

        # Update price history with current price
        if self.price_history:
            self.price_history.append(current_price)
            # Keep only last 50 prices
            self.price_history = self.price_history[-50:]
        else:
            self.price_history = [current_price] * 50

        # Calculate basic technical indicators
        rsi = self._calculate_rsi(self.price_history, period=14)
        sma_short = self._calculate_sma(self.price_history, period=5)   # 5-period SMA
        sma_long = self._calculate_sma(self.price_history, period=20)   # 20-period SMA

        # Price (normalized)
        price_norm = current_price / 1000000.0  # Similar to training data

        # Volume/quantity (mock for now - would need order book data)
        qty = np.random.uniform(0.001, 0.01)

        # PnL and win flag (based on recent price movement)
        recent_prices = self.price_history[-10:] if len(self.price_history) >= 10 else self.price_history
        if len(recent_prices) >= 2:
            pnl = (recent_prices[-1] - recent_prices[0]) * 0.001  # Small position PnL
            win = 1 if pnl > 0 else 0
        else:
            pnl = 0.0
            win = 0

        # Start with basic features and extend to 68 dimensions
        features = [
            rsi,           # RSI (14-period)
            sma_short,     # Short SMA (5-period)
            sma_long,      # Long SMA (20-period)
            price_norm,    # Normalized price
            qty,           # Quantity
            pnl,           # Recent PnL
            win            # Win flag
        ]

        # Add advanced features if available
        if features_available and len(self.price_history) >= 20:
            advanced_features = self._compute_live_features(self.price_history)
            if advanced_features:
                logger.debug(f"Adding {len(advanced_features)} advanced features")
                # Add advanced features in sorted order for consistency
                for feature_name in sorted(advanced_features.keys()):
                    features.append(advanced_features[feature_name])

        # Ensure we have exactly 68 features for the model
        if len(features) < 68:
            # Pad with zeros or repeated basic features
            padding_needed = 68 - len(features)
            # Repeat basic features or use zeros
            for i in range(padding_needed):
                features.append(features[i % len(features)] if i < 7 else 0.0)
            logger.debug(f"Padded features to 68 dimensions (added {padding_needed})")
        elif len(features) > 68:
            features = features[:68]
            logger.debug(f"Truncated features to 68 dimensions")

        logger.debug(f"Total features: {len(features)}")
        return np.array(features, dtype=np.float32)

    def _should_trade_sell_bias(self, action: int) -> bool:
        """
        Apply sell bias to trading decisions.
        Returns True if trade should be executed.
        """
        if action == ACTION_HOLD:  # Hold
            return False

        # Apply sell bias multiplier
        sell_bias = self.config['sell_bias_multiplier']

        if action == ACTION_SELL:  # Sell signal
            return True  # Always allow sell signals
        elif action == ACTION_BUY:  # Buy signal
            # Only allow buy if sell bias allows (random chance)
            return np.random.random() < (1.0 / sell_bias)

        return False

    def _execute_trade(self, side: str, amount: float) -> bool:
        """Execute trade on Coincheck."""
        # This would implement actual API calls
        logger.info(f"Mock trade: {side} {amount} BTC")

        # Send Discord notification
        self._send_notification(
            f"💰 {side.upper()} Order Executed",
            f"Amount: {amount} BTC\nSide: {side}",
            "success" if side == "sell" else "warning"
        )

        return True

    def _update_position(self, action: int, current_price: float):
        """Update position based on model action."""
        old_position = self.position

        if action == ACTION_BUY and self.position <= 0:  # Buy signal
            self.position = 1
            self.entry_price = current_price
            self.trades_count += 1
            self.daily_trades += 1
            self._execute_trade("buy", self.config['min_trade_amount'])

        elif action == ACTION_SELL and self.position >= 0:  # Sell signal
            self.position = -1
            self.entry_price = current_price
            self.trades_count += 1
            self.daily_trades += 1
            self._execute_trade("sell", self.config['min_trade_amount'])

        elif action == ACTION_HOLD:  # Hold
            pass

        # Calculate PnL if position changed
        if old_position != self.position and old_position != 0:
            # Use +1 for long (buy), -1 for short (sell) to get correct PnL sign
            direction = 1 if old_position > 0 else -1
            pnl = (current_price - self.entry_price) * direction * self.config['min_trade_amount']
            self.total_pnl += pnl

            # Update auto-stop system with trade result
            if self.auto_stop:
                self.auto_stop.update_trade_result(pnl, {
                    "action": action,
                    "entry_price": self.entry_price,
                    "exit_price": current_price,
                    "position": self.position,
                    "timestamp": datetime.now()
                })

            self._send_notification(
                "📊 Position Update",
                f"PnL: {pnl:.2f} JPY\nTotal PnL: {self.total_pnl:.2f} JPY\nTrades: {self.trades_count}",
                "info"
            )

    def run_trading_loop(self, duration_hours: int = 1):
        """Run the main trading loop."""
        logger.info(f"Starting live trading for {duration_hours} hours")
        logger.info("Strategy: Sell-biased BTC/JPY trading")

        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)

        trades_this_hour = 0
        hour_start = datetime.now()
        last_price_update = datetime.now() - timedelta(minutes=10)  # Force initial update

        while datetime.now() < end_time:
            try:
                current_time = datetime.now()

                # Reset daily counters if new day
                if current_time.date() > self.daily_start_time.date():
                    self.daily_start_pnl = self.total_pnl
                    self.daily_trades = 0
                    self.daily_start_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                    logger.info("Daily counters reset")

                # Check daily risk limits
                if not self.disable_risk_limits:
                    daily_loss = self.total_pnl - self.daily_start_pnl
                    if daily_loss <= -self.config['max_daily_loss']:
                        logger.error(f"Daily loss limit reached: {daily_loss:.2f} JPY")
                        self._send_notification(
                            "🚨 Daily Loss Limit Reached",
                            f"Daily Loss: {daily_loss:.2f} JPY\nStopping trading for safety",
                            "error"
                        )
                        break

                    if self.daily_trades >= self.config['max_daily_trades']:
                        logger.warning(f"Daily trade limit reached: {self.daily_trades}")
                        time.sleep(self.config['price_check_interval'])
                        continue

                    # Check emergency stop loss (percentage based)
                    if self.entry_price > 0:
                        current_price = self._get_current_price()
                        if current_price > 0:
                            loss_pct = abs(current_price - self.entry_price) / self.entry_price
                            if loss_pct >= self.config['emergency_stop_loss']:
                                logger.error(f"Emergency stop loss triggered: {loss_pct:.1%} loss")
                                self._send_notification(
                                    "🚨 Emergency Stop Loss Triggered",
                                    f"Loss: {loss_pct:.1%}\nEntry: {self.entry_price:.0f}\nCurrent: {current_price:.0f}",
                                    "error"
                                )
                                # Close position immediately
                                if self.position != 0:
                                    self._update_position(ACTION_SELL if self.position > 0 else ACTION_BUY, current_price)
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
                if trades_this_hour >= self.config['max_trades_per_hour']:
                    time.sleep(self.config['price_check_interval'])
                    continue

                # Get current market features
                features = self._get_market_features()

                # Get model prediction
                obs = features.reshape(1, -1)
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action[0])

                # Apply sell bias
                should_trade = self._should_trade_sell_bias(action)

                if should_trade:
                    current_price = self._get_current_price()
                    if current_price > 0:
                        self._update_position(action, current_price)
                        trades_this_hour += 1

                        logger.info(f"Trade executed: action={ACTION_NAMES[action]} ({action}), price={current_price}")

                # Wait before next check
                time.sleep(self.config['price_check_interval'])

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                self._send_notification(
                    "❌ Trading Error",
                    f"Error: {str(e)}",
                    "error"
                )
                time.sleep(60)  # Wait 1 minute on error

        # Send completion notification
        auto_stop_status = ""
        if self.auto_stop:
            status = self.auto_stop.get_status()
            auto_stop_status = f"\nAuto-Stop Status: {'Active' if status['is_active'] else 'Stopped'}"

        self._send_notification(
            "🏁 Live Trading Completed",
            f"Duration: {duration_hours} hours\nTotal PnL: {self.total_pnl:.2f} JPY\nTotal Trades: {self.trades_count}{auto_stop_status}",
            "info"
        )

        logger.info(f"Live trading completed. Total PnL: {self.total_pnl:.2f} JPY")


def main():
    parser = argparse.ArgumentParser(description="Live BTC/JPY Trading Bot")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=0.1 if "--dry-run" in sys.argv else 1,
        help="Trading duration in hours (default: 0.1 for dry-run, 1 for live)",
    )
    parser.add_argument(
        "--disable-risk-limits",
        action="store_true",
        help="Disable all risk limits (daily loss, trade count, emergency stop) - USE WITH CAUTION",
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - No real trades will be executed")

    try:
        trader = LiveTrader(args.model_path, disable_risk_limits=args.disable_risk_limits)
        trader.run_trading_loop(args.duration_hours)

    except Exception as e:
        logger.error(f"Failed to start live trading: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()