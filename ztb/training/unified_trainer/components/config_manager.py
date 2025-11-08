"""
Training Configuration Manager - Handles training configuration management.

This module separates configuration-related logic from the main trainer class,
including config validation, environment setup, and parameter management.
"""

from typing import Any, Dict, Optional

from ztb.training.constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE_SAC,
    DEFAULT_TOTAL_TIMESTEPS_SAC,
    DEFAULT_TOTAL_TIMESTEPS_PPO,
)
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrainingConfigManager:
    """
    Manages training configuration and setup.

    This class handles:
    - Configuration validation and normalization
    - Environment-specific parameter setup
    - Algorithm-specific hyperparameter management
    - Training pipeline configuration
    """

    def __init__(self):
        """Initialize TrainingConfigManager."""
        self.logger = get_logger(__name__)

    def process_config(
        self,
        config: Any,
        global_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Process and validate training configuration.

        Args:
            config: Raw configuration input
            global_config: Global configuration object

        Returns:
            Processed configuration dictionary

        Raises:
            ValueError: If configuration is invalid
            TypeError: If config types are incorrect
        """
        try:
            # Handle ZaifTradeBotConfig
            if self._is_zaif_config(config):
                return self._process_zaif_config(config)
            else:
                return self._validate_config_dict(config)

        except Exception as e:
            self.logger.error(f"Failed to process training config: {e}")
            raise

    def _is_zaif_config(self, config: Any) -> bool:
        """Check if config is a ZaifTradeBotConfig object."""
        return (
            hasattr(config, "training") and config.training is not None
        )

    def _process_zaif_config(self, config: Any) -> Dict[str, Any]:
        """Process ZaifTradeBotConfig into training config dict."""
        try:
            from ztb.config.schema import ZaifTradeBotConfig

            if not isinstance(config, ZaifTradeBotConfig):
                raise TypeError(f"Expected ZaifTradeBotConfig, got {type(config)}")

            training_config = config.training
            config_dict = {
                "training": {
                    "algorithm": training_config.algorithm,
                    "total_timesteps": training_config.total_timesteps or (
                        DEFAULT_TOTAL_TIMESTEPS_SAC if training_config.algorithm == "sac"
                        else DEFAULT_TOTAL_TIMESTEPS_PPO
                    ),
                    "model_name": training_config.model_name,
                },
                "data_path": training_config.data_config.csv_path
                if training_config.data_config and training_config.data_config.csv_path
                else None,
                "data_config": training_config.data_config.dict()
                if training_config.data_config
                else {},
                "environment": training_config.environment.dict()
                if training_config.environment
                else {},
                "features": training_config.features.dict()
                if training_config.features
                else {},
            }

            # Add algorithm-specific hyperparameters
            if (
                training_config.algorithm == "sac"
                and training_config.sac_hyperparameters
            ):
                config_dict["sac_hyperparameters"] = training_config.sac_hyperparameters.dict()
            elif training_config.algorithm == "sac":
                # Use default SAC hyperparameters
                config_dict["sac_hyperparameters"] = {
                    "learning_rate": DEFAULT_LEARNING_RATE,
                    "batch_size": DEFAULT_BATCH_SIZE_SAC,
                }
            elif (
                training_config.algorithm == "ppo"
                and training_config.ppo_hyperparameters
            ):
                config_dict[
                    "ppo_hyperparameters"
                ] = training_config.ppo_hyperparameters.dict()

            return config_dict

        except Exception as e:
            self.logger.error(f"Failed to process Zaif config: {e}")
            raise

    def _validate_config_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary."""
        if not isinstance(config, dict):
            raise TypeError(f"Config must be dict, got {type(config)}")

        required_keys = ["training"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        training_config = config["training"]
        if not isinstance(training_config, dict):
            raise TypeError("training config must be dict")

        required_training_keys = ["algorithm", "total_timesteps"]
        for key in required_training_keys:
            if key not in training_config:
                raise ValueError(f"Missing required training config key: {key}")

        # Process advanced market regime configuration
        if "advanced_market_regime" in config:
            self._process_advanced_market_regime_config(config)

        return config

    def _process_advanced_market_regime_config(self, config: Dict[str, Any]) -> None:
        """Process advanced market regime configuration."""
        regime_config = config["advanced_market_regime"]

        # Validate required fields
        if not isinstance(regime_config, dict):
            raise TypeError("advanced_market_regime config must be dict")

        required_fields = ["enabled", "detector_type", "detection_window", "adaptation_frequency"]
        for field in required_fields:
            if field not in regime_config:
                raise ValueError(f"Missing required advanced_market_regime field: {field}")

        # Validate detector type
        valid_detector_types = ["basic", "advanced"]
        if regime_config["detector_type"] not in valid_detector_types:
            raise ValueError(f"Invalid detector_type. Must be one of: {valid_detector_types}")

        # Validate regime-specific parameters if present
        if "regime_specific_params" in regime_config:
            params = regime_config["regime_specific_params"]
            if not isinstance(params, dict):
                raise TypeError("regime_specific_params must be dict")

            # Validate that all required regimes are present for advanced detector
            if regime_config["detector_type"] == "advanced":
                required_regimes = [
                    "strong_bull_trend", "moderate_bull_trend", "weak_bull_trend",
                    "strong_bear_trend", "moderate_bear_trend", "weak_bear_trend",
                    "high_volatility_ranging", "moderate_volatility_ranging", "low_volatility_ranging",
                    "extreme_volatility", "consolidation", "breakout_setup", "breakdown_setup"
                ]

                for regime in required_regimes:
                    if regime not in params:
                        raise ValueError(f"Missing regime parameters for: {regime}")

        self.logger.info("Advanced market regime configuration validated successfully")

    def get_algorithm_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract algorithm-specific configuration.

        Args:
            config: Processed configuration

        Returns:
            Algorithm-specific config
        """
        algorithm = config["training"]["algorithm"]
        if algorithm == "sac" and "sac_hyperparameters" in config:
            return config["sac_hyperparameters"]
        elif algorithm == "ppo" and "ppo_hyperparameters" in config:
            return config["ppo_hyperparameters"]
        else:
            return {}

    def get_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract environment configuration.

        Args:
            config: Processed configuration

        Returns:
            Environment config
        """
        return config.get("environment", {})

    def get_data_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data configuration.

        Args:
            config: Processed configuration

        Returns:
            Data config
        """
        return config.get("data_config", {})

    def get_feature_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract feature configuration.

        Args:
            config: Processed configuration

        Returns:
            Feature config
        """
        return config.get("features", {})