"""
Reward Function Configuration Manager

Handles loading, validation, and management of reward function configurations.
Separated from the main optimizer to follow Single Responsibility Principle.
"""

from pathlib import Path
from typing import Optional

from ztb.io.json_io import read_json_object, write_json
from ztb.utils.config_manager import ConfigManager
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)
ConfigObject = dict[str, object]


class RewardFunctionConfigManager(ConfigManager):
    """
    Manages reward function configurations.

    Extends ConfigManager to provide reward function specific configuration handling.
    """

    def __init__(self, config_dir: Optional[str] = None):
        super().__init__(config_dir)
        self.logger = get_logger(__name__)

    def load_base_config_from_file(self, config_file_path: str) -> ConfigObject:
        """
        Load base configuration from a JSON file.

        Args:
            config_file_path: Path to the configuration file

        Returns:
            Loaded configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            TypeError: If config file does not contain a JSON object
        """
        config_path = Path(config_file_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

        config = read_json_object(config_path)
        self.logger.info(f"Loaded configuration from {config_file_path}")
        return config

    def validate_config(self, config: ConfigObject) -> bool:
        """
        Validate configuration structure.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        required_keys = ["optimization", "backtest", "reward_function"]

        for key in required_keys:
            if key not in config:
                self.logger.error(f"Missing required config key: {key}")
                return False

        # Validate optimization config
        opt_config = config.get("optimization")
        if not isinstance(opt_config, dict):
            self.logger.error("optimization must be an object")
            return False
        if not isinstance(opt_config.get("max_trials"), int):
            self.logger.warning("max_trials should be an integer")

        # Validate backtest config
        bt_config = config.get("backtest")
        if not isinstance(bt_config, dict):
            self.logger.error("backtest must be an object")
            return False
        if not isinstance(bt_config.get("initial_balance"), (int, float)):
            self.logger.warning("initial_balance should be a number")

        return True

    def get_default_config(self) -> ConfigObject:
        """
        Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "optimization": {
                "max_trials": 100,
                "timeout_hours": 24.0,
                "n_jobs": 1,
                "early_stopping_rounds": 20,
                "optimization_method": "bayesian",
            },
            "backtest": {
                "initial_balance": 100000,
                "commission": 0.001,
                "slippage": 0.0005,
                "max_position_size": 1.0,
                "min_position_size": 0.01,
            },
            "reward_function": {
                "stage": "balanced_transition",
                "objectives": ["profit", "sharpe", "win_rate", "consistency"],
                "weights": {
                    "profit": 0.4,
                    "sharpe": 0.3,
                    "win_rate": 0.2,
                    "consistency": 0.1,
                },
            },
        }

    def merge_configs(
        self, base_config: ConfigObject, override_config: ConfigObject
    ) -> ConfigObject:
        """
        Merge two configurations with override taking precedence.

        Args:
            base_config: Base configuration
            override_config: Configuration to override with

        Returns:
            Merged configuration
        """
        merged = base_config.copy()

        def deep_merge(base: ConfigObject, override: ConfigObject) -> ConfigObject:
            result = base.copy()
            for key, value in override.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_merge(merged, override_config)

    def save_config_to_file(self, config: ConfigObject, file_path: str) -> None:
        """
        Save configuration to a JSON file.

        Args:
            config: Configuration to save
            file_path: Path to save the configuration
        """
        config_path = Path(file_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved configuration to {file_path}")
