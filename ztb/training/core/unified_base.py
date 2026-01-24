#!/usr/bin/env python3
"""
Base Classes for Unified Training and Analysis Systems

Common functionality shared between trainers and analyzers.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, cast

from ztb.io.json_io import read_json, write_json
from ztb.utils.logging_utils import get_logger, setup_logging
from ztb.utils.safety import safe_config_get


class UnifiedBase(ABC):
    """Base class for unified training and analysis systems."""

    def __init__(
        self, config_path: Optional[str] = None, version: Optional[str] = None
    ):
        """
        Initialize base unified system.

        Args:
            config_path: Path to configuration file
            version: Version identifier
        """
        self.config_path = Path(config_path) if config_path else None
        self.version = version
        self.config: Dict[str, Any] = {}
        self.logger = get_logger(self.__class__.__name__)

        # Setup logging if config specifies level
        if self.config_path and self.config_path.exists():
            try:
                temp_config = read_json(self.config_path)
                log_level = safe_config_get(temp_config, "logging_level", "INFO")
                # Convert string level to int if needed
                if isinstance(log_level, str):
                    log_level = getattr(logging, log_level.upper(), logging.INFO)
                setup_logging(level=log_level)
            except Exception:
                setup_logging(level=logging.INFO)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            config = read_json(config_path)
            self.logger.info(f"Configuration loaded from: {config_path}")
            return cast(Dict[str, Any], config)
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """Save configuration to file."""
        try:
            write_json(output_path, config, indent=2, ensure_ascii=False)
            self.logger.info(f"Configuration saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise

    def validate_config(self, config: Dict[str, Any], required_fields: list) -> bool:
        """Validate configuration has required fields."""
        missing = [field for field in required_fields if field not in config]
        if missing:
            self.logger.error(f"Missing required fields: {missing}")
            return False
        return True

    @abstractmethod
    def run(self) -> None:
        """Execute the main functionality."""
        pass


class ConfigMixin:
    """Mixin for configuration management."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config: Dict[str, Any] = {}
        self.logger = get_logger(self.__class__.__name__)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return safe_config_get(self.config, key, default)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.config.update(updates)
        self.logger.debug(f"Configuration updated: {list(updates.keys())}")


class LoggingMixin:
    """Mixin for enhanced logging."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)

    def log_operation_start(self, operation: str, **kwargs: Any) -> None:
        """Log operation start with context."""
        self.logger.info(f"Starting {operation}", extra=kwargs)

    def log_operation_end(
        self, operation: str, success: bool = True, **kwargs: Any
    ) -> None:
        """Log operation end with result."""
        status = "completed successfully" if success else "failed"
        self.logger.info(f"{operation.capitalize()} {status}", extra=kwargs)

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        """Log metrics in structured format."""
        for key, value in metrics.items():
            self.logger.info(f"{prefix}{key}: {value}")
