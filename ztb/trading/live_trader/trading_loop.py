"""Trading loop implementation for live trading."""

import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from ztb.trading.constants import ACTION_HOLD
from ztb.utils.logging_utils import get_logger
from ztb.utils.performance_utils import PerformanceMonitor

if TYPE_CHECKING:
    from ztb.trading.live_trader.live_trader import LiveTrader


class TradingLoop:
    """Handles the main trading loop execution for live trading."""

    def __init__(self, live_trader: "LiveTrader"):
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
                        current_price = self.live_trader._get_current_price()
                        logger.info(
                            f"📈 Price update #{iteration_count}: ¥{current_price:,.0f}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to get current price: {e}")
                        if self.live_trader._last_valid_price > 0:
                            current_price = self.live_trader._last_valid_price
                            logger.warning(
                                f"Using last valid price: ¥{current_price:,.0f}"
                            )
                        else:
                            logger.error("No valid price available, skipping iteration")
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                logger.critical(
                                    f"Too many consecutive errors ({consecutive_errors}), stopping trading loop"
                                )
                                self.live_trader._send_notification(
                                    "🚨 CRITICAL: Trading Stopped",
                                    f"Too many consecutive errors ({consecutive_errors}). Manual intervention required.",
                                    "error",
                                )
                                break
                            time.sleep(60)
                            continue

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
                        action_name = self.live_trader.ACTION_NAMES.get(
                            action, f"UNKNOWN({action})"
                        )
                        logger.debug(f"Predicted action: {action_name}")
                    except Exception as e:
                        logger.error(f"Failed to predict action: {e}")
                        action = ACTION_HOLD
                        action_name = "HOLD (fallback)"

                    # Validate position before executing action
                    if not (-1 <= self.live_trader.position <= 1):
                        logger.error(
                            f"Invalid position detected: {self.live_trader.position}, resetting to 0"
                        )
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
                            "info" if action == ACTION_HOLD else "success",
                        )
                        logger.debug("Notification sent for iteration")

                    # Periodic cleanup
                    try:
                        self.live_trader._periodic_cleanup()
                    except Exception as e:
                        logger.warning(f"Failed to perform periodic cleanup: {e}")

                except Exception as e:
                    logger.error(
                        f"❌ Critical error in trading loop iteration {iteration_count}: {e}"
                    )
                    import traceback

                    logger.error(f"Traceback: {traceback.format_exc()}")
                    print(f"Traceback: {traceback.format_exc()}")
                    consecutive_errors += 1

                    self.live_trader._send_notification(
                        "⚠️ Trading Error",
                        f"Critical error in iteration {iteration_count}: {e}",
                        "error",
                    )

                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical(
                            f"Too many consecutive errors ({consecutive_errors}), stopping trading loop"
                        )
                        self.live_trader._send_notification(
                            "🚨 CRITICAL: Trading Stopped",
                            f"Too many consecutive errors ({consecutive_errors}). Manual intervention required.",
                            "error",
                        )
                        break

            # Wait before next iteration (1 minute in live mode, 1 second in dry-run)
            wait_time = 1 if self.live_trader.dry_run else 60
            time.sleep(wait_time)

        # Final report
        total_pnl = self.live_trader.total_pnl
        trades_count = self.live_trader.trades_count

        logger.info(f"🏁 Trading loop completed after {duration_hours} hours")
        logger.info(f"   Total PnL: ¥{total_pnl:,.2f}")
        logger.info(f"   Total trades: {trades_count}")

        self.live_trader._send_notification(
            "🏁 Trading Session Complete",
            f"Duration: {duration_hours} hours\nTotal PnL: ¥{total_pnl:,.2f}\nTrades: {trades_count}\nFinal Position: {self.live_trader.position:.4f} BTC",
            "success",
        )
