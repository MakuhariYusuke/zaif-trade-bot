"""Trading loop implementation for live trading."""

from datetime import datetime, timedelta
import time
from typing import TYPE_CHECKING

import numpy as np
import requests

from ztb.trading.constants import ACTION_HOLD
from ztb.utils.logging_utils import get_logger
from ztb.utils.performance_utils import PerformanceMonitor

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader


class TradingLoop:
    """Handles the main trading loop execution for live trading."""

    def __init__(self, live_trader: 'LiveTrader'):
        """Initialize trading loop with reference to live trader."""
        self.live_trader = live_trader
        self.logger = get_logger(__name__)

    def run_trading_loop(self, duration_hours: float) -> None:
        """Run the main trading loop for live trading."""
        logger = self.logger
        logger.info(f"🚀 Starting live trading loop for {duration_hours} hours")

        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)

        iteration_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5

        logger.debug("Entering trading loop")
        while datetime.now() < end_time:
            iteration_count += 1
            logger.debug(f"Starting iteration {iteration_count}")

            with PerformanceMonitor(f"trading_iteration_{iteration_count}"):
                try:
                    # Get current price
                    try:
                        current_price = self.live_trader._get_current_price_sync()
                        
                        # Validate price data
                        if not isinstance(current_price, (int, float)) or not np.isfinite(current_price):
                            raise ValueError(f"Invalid price format: {current_price} (type: {type(current_price)})")
                        
                        # Check for reasonable price range (BTC/JPY should be between ¥1M and ¥100M)
                        if not (1000000 <= current_price <= 100000000):
                            logger.warning(f"Price outside expected range: ¥{current_price:,.0f} (expected: ¥1M-¥100M)")
                        
                        logger.info(f"📈 Price update #{iteration_count}: ¥{current_price:,.0f}")
                    except Exception as e:
                        logger.error(f"Failed to get current price: {e}")
                        # Log detailed traceback for debugging
                        import traceback
                        logger.debug(f"Price fetch traceback: {traceback.format_exc()}")
                        
                        # Handle specific error types
                        if isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
                            logger.warning(f"Network error during price fetch: {type(e).__name__}")
                            consecutive_errors += 1
                        elif isinstance(e, requests.exceptions.HTTPError):
                            logger.warning(f"HTTP error during price fetch: {e}")
                            consecutive_errors += 1
                        else:
                            logger.error(f"Unexpected error during price fetch: {type(e).__name__}")
                            consecutive_errors += 1
                        
                        if self.live_trader._last_valid_price > 0:
                            current_price = self.live_trader._last_valid_price
                            logger.warning(f"Using last valid price: ¥{current_price:,.0f} (iteration #{iteration_count})")
                            # Reset consecutive errors if we have fallback price
                            consecutive_errors = 0
                        else:
                            logger.critical("CRITICAL: No valid price available, terminating trading loop")
                            logger.critical(f"Failed iterations: {iteration_count}, Consecutive errors: {consecutive_errors}")
                            # Log detailed error context
                            import traceback
                            logger.error(f"Full error traceback: {traceback.format_exc()}")
                            self.live_trader._send_notification("🚨 CRITICAL: Trading Stopped", f"Unable to obtain price data after {consecutive_errors} consecutive failures. Manual intervention required.", "error")
                            break

                    # Reset consecutive error counter on successful price fetch
                    consecutive_errors = 0

                    # Update price history (skip in dry-run mode)
                    if not self.live_trader.dry_run:
                        try:
                            self.live_trader._update_price_history()
                        except Exception as e:
                            logger.warning(f"Failed to update price history: {e}")
                            # Continue with existing history

                    # Compute features for prediction
                    logger.debug("Computing features...")
                    try:
                        features = self.live_trader._compute_features()
                        logger.debug(f"Features computed: {len(features)} features")
                    except Exception as e:
                        logger.error(f"Failed to compute features: {e}")
                        logger.warning("Using zero features as fallback")
                        features = np.zeros(64, dtype=np.float32)

                    # Predict action
                    logger.debug("Predicting action...")
                    try:
                        action = self.live_trader._predict_action(features)
                        action_name = self.live_trader.ACTION_NAMES.get(action, f"UNKNOWN({action})")
                        logger.debug(f"Predicted action: {action_name}")
                    except Exception as e:
                        logger.error(f"Failed to predict action: {e}")
                        action = ACTION_HOLD
                        action_name = "HOLD (fallback)"

                    # Validate position before executing action
                    if not (-1 <= self.live_trader.position <= 1):
                        logger.error(f"Invalid position detected: {self.live_trader.position}, resetting to 0")
                        self.live_trader.position = 0.0

                    # Execute action
                    logger.debug("Executing action...")
                    try:
                        pnl = self.live_trader._execute_action(action)
                        logger.debug(f"Action executed, PnL: {pnl}")
                    except Exception as e:
                        logger.error(f"Failed to execute action: {e}")
                        pnl = 0.0
                        action_name = f"{action_name} (execution failed)"

                    # Send periodic notification (every 10 iterations or significant events)
                    if iteration_count % 10 == 0 or action != ACTION_HOLD:
                        self.live_trader._send_notification(
                            f"📊 Trading Update #{iteration_count}",
                            f"Price: ¥{current_price:,.0f}\nAction: {action_name}\nPnL: ¥{pnl:,.2f}\nPosition: {self.live_trader.position:.4f} BTC",
                            "info" if action == ACTION_HOLD else "success"
                        )
                        logger.debug("Notification sent for iteration")

                    # Periodic cleanup
                    try:
                        self.live_trader._periodic_cleanup()
                    except Exception as e:
                        logger.warning(f"Failed to perform periodic cleanup: {e}")

                except Exception as e:
                    logger.error(f"❌ Critical error in trading loop iteration {iteration_count}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    print(f"Traceback: {traceback.format_exc()}")
                    consecutive_errors += 1

                    self.live_trader._send_notification("⚠️ Trading Error", f"Critical error in iteration {iteration_count}: {e}", "error")

                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical(f"Too many consecutive errors ({consecutive_errors}), stopping trading loop")
                        self.live_trader._send_notification("🚨 CRITICAL: Trading Stopped", f"Too many consecutive errors ({consecutive_errors}). Manual intervention required.", "error")
                        break

            # Wait before next iteration (1 minute in live mode, 1 second in dry-run)
            wait_time = 1 if self.live_trader.dry_run else 60
            time.sleep(wait_time)

        # Final report with enhanced statistics
        total_pnl = self.live_trader.total_pnl
        trades_count = self.live_trader.trades_count
        final_position = self.live_trader.position
        total_iterations = iteration_count
        uptime_hours = (datetime.now() - start_time).total_seconds() / 3600

        logger.info(f"🏁 Trading loop completed after {duration_hours:.2f} hours")
        logger.info(f"   Total iterations: {total_iterations}")
        logger.info(f"   Total PnL: ¥{total_pnl:,.2f}")
        logger.info(f"   Total trades: {trades_count}")
        logger.info(f"   Final position: {final_position:.4f} BTC")
        logger.info(f"   Average iterations per hour: {total_iterations/uptime_hours:.1f}")
        logger.info(f"   Trading efficiency: {trades_count/total_iterations*100:.1f}% action rate")

        self.live_trader._send_notification(
            "🏁 Trading Session Complete",
            f"Duration: {duration_hours:.2f} hours ({total_iterations} iterations)\n"
            f"Total PnL: ¥{total_pnl:,.2f}\n"
            f"Trades: {trades_count} (avg: {trades_count/total_iterations*100:.1f}% action rate)\n"
            f"Final Position: {final_position:.4f} BTC\n"
            f"Performance: {total_iterations/uptime_hours:.1f} iter/hr",
            "success"
        )