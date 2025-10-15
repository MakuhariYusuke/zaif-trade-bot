#!/usr/bin/env python3
"""
Configuration classes for live trading.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Optional health check endpoint
try:
    from flask import Flask, jsonify  # type: ignore[import-untyped]

    flask_available = True
except ImportError:
    flask_available = False

# Optional metrics collection
try:
    from prometheus_client import start_http_server  # type: ignore[import-untyped]

    prometheus_available = True
except ImportError:
    prometheus_available = False


@dataclass(frozen=True)
class LiveTradingOptions:
    """Runtime configuration for live trading."""

    model_path: Path
    algorithm: str
    venue: str
    duration_hours: float = 1.0
    disable_risk_limits: bool = False
    dry_run: bool = False
    log_level: str = "INFO"
    enable_metrics: bool = False
    metrics_port: int = 8000
    enable_health_check: bool = False
    health_port: int = 8080

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "LiveTradingOptions":
        """Create options from parsed CLI arguments and environment toggles."""
        log_level = args.log_level.upper()

        enable_metrics = (
            prometheus_available
            and os.getenv("ZTB_ENABLE_METRICS", "false").lower() == "true"
        )
        metrics_port = int(os.getenv("ZTB_METRICS_PORT", "8000"))

        enable_health = (
            flask_available
            and os.getenv("ZTB_ENABLE_HEALTH_CHECK", "false").lower() == "true"
        )
        health_port = int(os.getenv("ZTB_HEALTH_PORT", "8080"))

        return cls(
            model_path=Path(args.model_path).expanduser(),
            algorithm=args.algorithm,
            venue=args.venue,
            duration_hours=args.duration_hours,
            disable_risk_limits=args.disable_risk_limits,
            dry_run=args.dry_run,
            log_level=log_level,
            enable_metrics=enable_metrics,
            metrics_port=metrics_port,
            enable_health_check=enable_health,
            health_port=health_port,
        )


@dataclass
class MetricsServerHandle:
    """Handle for metrics server lifecycle."""

    port: int


@dataclass
class HealthServerHandle:
    """Handle for health-check HTTP server."""

    port: int
    thread: "threading.Thread"


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for live trading."""
    parser = argparse.ArgumentParser(description="Live BTC/JPY Trading Bot")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["ppo", "sac"],
        default="sac",
        help="Algorithm type (ppo or sac)",
    )
    parser.add_argument(
        "--venue",
        choices=["coincheck", "bitflyer"],
        default="coincheck",
        help="Trading venue (coincheck or bitflyer)",
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enable dry run mode - no real trades will be executed",
    )
    return parser


def _start_metrics_server(
    options: LiveTradingOptions, logger
) -> Optional[MetricsServerHandle]:
    """Bootstrap Prometheus metrics server if requested and available."""
    if not options.enable_metrics:
        return None
    if not prometheus_available:
        logger.warning(
            "Prometheus metrics requested but prometheus_client is not installed."
        )
        return None

    start_http_server(options.metrics_port)  # type: ignore[name-defined]
    logger.info("Prometheus metrics server started on port %s", options.metrics_port)
    return MetricsServerHandle(port=options.metrics_port)


def _start_health_server(
    options: LiveTradingOptions,
    trader_status_provider,
    logger,
) -> Optional[HealthServerHandle]:
    """Bootstrap Flask health endpoint if requested and available."""
    if not options.enable_health_check:
        return None
    if not flask_available:
        logger.warning(
            "Health check endpoint requested but Flask is not installed."
        )
        return None

    health_app = Flask(__name__)  # type: ignore[name-defined]

    @health_app.route("/health")  # type: ignore[misc]
    def health_check() -> Any:
        return jsonify(trader_status_provider())  # type: ignore[name-defined]

    import threading

    thread = threading.Thread(
        target=lambda: health_app.run(
            host="0.0.0.0", port=options.health_port, debug=False
        ),
        daemon=True,
    )
    thread.start()
    logger.info("Health check endpoint started on port %s", options.health_port)
    return HealthServerHandle(port=options.health_port, thread=thread)


class TradingConfig:
    """Configuration manager for trading parameters."""

    def __init__(self, ztb_config: Optional[Any] = None) -> None:
        """Initialize trading configuration.

        Args:
            ztb_config: ZTBConfig instance for environment-specific settings
        """
        self.ztb_config = ztb_config
        self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default trading configuration with safety limits."""
        return {
            "reward_scaling": self._get_float("ZTB_REWARD_SCALING", 1.0),
            "transaction_cost": self._get_float("ZTB_TRANSACTION_COST", 0.001),
            "max_position_size": self._get_float("ZTB_MAX_POSITION_SIZE", 0.1),
            "sell_bias_multiplier": self._get_float("ZTB_SELL_BIAS_MULTIPLIER", 2.0),
            "min_trade_amount": self._get_float("ZTB_MIN_TRADE_AMOUNT", 0.001),
            "max_trades_per_hour": self._get_int("ZTB_MAX_TRADES_PER_HOUR", 6),
            "price_check_interval": self._get_int("ZTB_PRICE_CHECK_INTERVAL", 60),
            "max_daily_loss": self._get_float("ZTB_MAX_DAILY_LOSS", 10000.0),
            "max_daily_trades": self._get_int("ZTB_MAX_DAILY_TRADES", 50),
            "emergency_stop_loss": self._get_float("ZTB_EMERGENCY_STOP_LOSS", 0.05),
            "price_history_length": self._get_int("ZTB_PRICE_HISTORY_LENGTH", 64),
            "rsi_neutral_value": self._get_float("ZTB_RSI_NEUTRAL_VALUE", 50.0),
            "rsi_period": self._get_int("ZTB_RSI_PERIOD", 14),
            "fallback_price": self._get_float("ZTB_FALLBACK_PRICE", 5000000.0),
            "price_min": self._get_int("ZTB_PRICE_MIN", 1000000),
            "price_max": self._get_int("ZTB_PRICE_MAX", 50000000),
            "price_change_threshold": self._get_float("ZTB_PRICE_CHANGE_THRESHOLD", 0.20),
            "continuous_to_discrete_threshold": self._get_float("ZTB_CONTINUOUS_TO_DISCRETE_THRESHOLD", 0.33),
        }

    def _get_float(self, key: str, default: float) -> float:
        """Get float value from config or environment."""
        if self.ztb_config:
            return self.ztb_config.get_float(key, default)
        return float(os.getenv(key, str(default)))

    def _get_int(self, key: str, default: int) -> int:
        """Get int value from config or environment."""
        if self.ztb_config:
            return self.ztb_config.get_int(key, default)
        return int(os.getenv(key, str(default)))

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration values."""
        self._config.update(updates)

    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config.copy()