"""
Configuration schemas using Pydantic for type safety and validation.

This module defines the configuration models for various components of the trading system.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ztb.trading.environment.constants import (
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_TRANSACTION_COST,
)

# Import path utilities for robust path handling
from ztb.utils.path_utils import ensure_dir, get_project_root


class DataConfig(BaseModel):
    """Data source configuration."""

    csv_path: Optional[str] = None
    use_real_data: bool = True
    data_path: Optional[str] = None
    random_start: bool = True
    min_samples: int = Field(
        default=10000, description="Minimum samples for evaluation"
    )


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""

    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)
    wave: Optional[Union[str, int]] = None
    harmful: bool = False


class FeaturesConfig(BaseModel):
    """Complete feature set configuration."""

    features: Dict[str, FeatureConfig] = Field(default_factory=dict)
    feature_set: str = "default"
    adaptive_feature_selection: Dict[str, Any] = Field(default_factory=dict)


class RewardSettings(BaseModel):
    """Reward function configuration."""

    use_simple_reward: bool = False
    reward_scale: float = 100.0
    trading_bonus: float = 0.01
    profit_bonuses: Dict[str, float] = Field(default_factory=dict)
    penalty_coefficients: Dict[str, float] = Field(default_factory=dict)
    entropy_bonus: float = 0.0
    reward_clip_value: float = 2.0


class EnvironmentConfig(BaseModel):
    """Trading environment configuration."""

    model_config = ConfigDict(extra="allow")
    
    initial_balance: float = DEFAULT_INITIAL_BALANCE
    transaction_cost: float = DEFAULT_TRANSACTION_COST
    max_position_size: float = 1.0
    enable_action_masking: bool = True
    use_continuous_actions: bool = True
    use_standardized_observations: bool = True
    curriculum_stage: str = "balanced_trading"
    continuous_to_discrete_threshold: float = 0.1
    feature_set: str = "default"
    csv_path: Optional[str] = None
    reward_settings: RewardSettings = Field(default_factory=RewardSettings)


class CurriculumLearningConfig(BaseModel):
    """Curriculum learning configuration."""

    model_config = ConfigDict(extra="allow")
    
    enabled: bool = False
    curriculum_stage: str = "pnl_focused"
    stage_progression: List[str] = Field(default_factory=list)
    stage_timesteps: List[int] = Field(default_factory=list)


class SACHyperparameters(BaseModel):
    """SAC algorithm hyperparameters."""

    learning_rate: float = 0.0003
    buffer_size: int = 50000
    learning_starts: int = 1000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    ent_coef: Union[str, float] = 0.01
    target_update_interval: int = 1
    target_entropy: Union[str, float] = -2.0


class PPOHyperparameters(BaseModel):
    """PPO algorithm hyperparameters."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class TrainingConfig(BaseModel):
    """Training configuration."""

    model_config = ConfigDict(extra="allow")

    model_name: str
    algorithm: str = Field(..., pattern="^(sac|ppo)$")
    total_timesteps: int = 10000
    data_config: DataConfig = Field(default_factory=DataConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    curriculum_learning: Optional[CurriculumLearningConfig] = Field(default=None, description="Curriculum learning configuration")
    sac_hyperparameters: Optional[SACHyperparameters] = None
    ppo_hyperparameters: Optional[PPOHyperparameters] = None

    @model_validator(mode="after")
    def validate_algorithm_params(self):
        algorithm = self.algorithm
        if algorithm == "sac" and not self.sac_hyperparameters:
            self.sac_hyperparameters = SACHyperparameters()
        elif algorithm == "ppo" and not self.ppo_hyperparameters:
            self.ppo_hyperparameters = PPOHyperparameters()
        return self


class EvaluationThresholds(BaseModel):
    """Evaluation performance thresholds."""

    re_evaluate: float = 0.05
    monitor: float = 0.01


class EvaluationConfig(BaseModel):
    """Evaluation and backtesting configuration."""

    thresholds: EvaluationThresholds = Field(default_factory=EvaluationThresholds)
    min_samples: int = 10000
    risk_metrics: List[str] = Field(
        default_factory=lambda: ["sharpe", "sortino", "max_drawdown"]
    )
    performance_metrics: List[str] = Field(
        default_factory=lambda: ["total_return", "win_rate", "profit_factor"]
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None


class DeploymentConfig(BaseModel):
    """Production deployment configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    model_path: str = Field(
        default_factory=lambda: str(get_project_root() / "models" / "production")
    )
    feature_scaler_path: Optional[str] = Field(
        default_factory=lambda: str(get_project_root() / "models" / "scalers")
    )
    enable_monitoring: bool = True
    log_level: str = "INFO"

    @field_validator("model_path", "feature_scaler_path")
    @classmethod
    def validate_paths(cls, v):
        """Validate and resolve paths relative to project root."""
        if v and not Path(v).is_absolute():
            return str(get_project_root() / v)
        return v


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint management."""

    model_config = ConfigDict(extra="ignore")

    async_save: bool = Field(
        default=True, description="Enable asynchronous checkpoint saving"
    )
    compress: str = Field(
        default="zstd", description="Compression algorithm (none, zstd, lz4)"
    )
    max_pending: int = Field(default=1, description="Maximum pending checkpoints")
    retention: int = Field(default=5, description="Number of checkpoints to retain")
    interval_steps: int = Field(
        default=10000, description="Checkpoint interval in steps"
    )
    light_mode: bool = Field(default=False, description="Use light checkpoint mode")


class StreamingConfig(BaseModel):
    """Configuration for data streaming."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(default=False, description="Enable streaming mode")
    batch_size: int = Field(default=64, description="Streaming batch size")
    buffer_policy: str = Field(
        default="drop_oldest", description="Buffer overflow policy"
    )
    prefetch_factor: int = Field(default=2, description="Prefetch factor for streaming")


class EvalConfig(BaseModel):
    """Configuration for evaluation parameters."""

    model_config = ConfigDict(extra="ignore")

    dsr_trials: int = Field(default=1000, description="Number of DSR trials")
    bootstrap_resamples: int = Field(
        default=1000, description="Number of bootstrap resamples"
    )
    bootstrap_block: Optional[int] = Field(
        default=None, description="Bootstrap block size"
    )
    bootstrap_overlap: Optional[int] = Field(
        default=None, description="Bootstrap overlap"
    )
    eval_freq: int = Field(default=50000, description="Evaluation frequency in steps")
    benchmark_strategies: List[str] = Field(
        default=["sma", "buy_hold"], description="Benchmark strategies"
    )


class VenuePrecisionConfig(BaseModel):
    """Configuration for venue-specific precision policies."""

    model_config = ConfigDict(extra="ignore")

    price_tick: float = Field(default=0.01, description="Minimum price increment")
    quantity_step: float = Field(
        default=0.0001, description="Minimum quantity increment"
    )
    min_quantity: Optional[float] = Field(
        default=None, description="Minimum order quantity"
    )
    max_quantity: Optional[float] = Field(
        default=None, description="Maximum order quantity"
    )
    min_price: Optional[float] = Field(default=None, description="Minimum order price")
    max_price: Optional[float] = Field(default=None, description="Maximum order price")


class RiskProfileConfig(BaseModel):
    """Configuration for risk management profiles."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Profile name")
    max_position_size: float = Field(
        default=0.1, description="Maximum position size as fraction of portfolio"
    )
    max_daily_loss: float = Field(
        default=0.05, description="Maximum daily loss as fraction of portfolio"
    )
    stop_loss_pct: float = Field(default=0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.04, description="Take profit percentage")
    max_open_positions: int = Field(
        default=5, description="Maximum number of open positions"
    )
    risk_per_trade: float = Field(
        default=0.01, description="Risk per trade as fraction of portfolio"
    )
    max_leverage: float = Field(default=1.0, description="Maximum leverage")
    cooldown_period: int = Field(
        default=60, description="Cooldown period in seconds after loss"
    )


class GlobalConfig(BaseModel):
    """Global configuration container."""

    model_config = ConfigDict(extra="allow")

    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)

    # Additional global settings
    experiment_name: Optional[str] = Field(default=None, description="Experiment name")
    log_level: str = Field(default="INFO", description="Logging level")
    output_dir: str = Field(
        default_factory=lambda: str(get_project_root() / "results"),
        description="Output directory",
    )

    # Venue precision policies
    venue_precision: Dict[str, Dict[str, VenuePrecisionConfig]] = Field(
        default_factory=dict, description="Venue-specific precision policies"
    )

    # Risk profile presets
    risk_profiles: Dict[str, RiskProfileConfig] = Field(
        default_factory=lambda: {
            "conservative": RiskProfileConfig(
                name="conservative",
                max_position_size=0.05,
                max_daily_loss=0.02,
                stop_loss_pct=0.01,
                take_profit_pct=0.02,
                max_open_positions=3,
                risk_per_trade=0.005,
                max_leverage=1.0,
                cooldown_period=120,
            ),
            "moderate": RiskProfileConfig(
                name="moderate",
                max_position_size=0.1,
                max_daily_loss=0.05,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                max_open_positions=5,
                risk_per_trade=0.01,
                max_leverage=1.0,
                cooldown_period=60,
            ),
            "aggressive": RiskProfileConfig(
                name="aggressive",
                max_position_size=0.2,
                max_daily_loss=0.1,
                stop_loss_pct=0.05,
                take_profit_pct=0.1,
                max_open_positions=10,
                risk_per_trade=0.02,
                max_leverage=2.0,
                cooldown_period=30,
            ),
        },
        description="Risk profile presets for different trading strategies",
    )


class ZaifTradeBotConfig(BaseModel):
    """
    Unified configuration schema for Zaif Trade Bot.

    This is the root configuration model that encompasses all aspects
    of the trading bot system.
    """

    version: str = "1.0"
    training: Optional[TrainingConfig] = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    deployment: Optional[DeploymentConfig] = None
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)

    @field_validator("version")
    @classmethod
    def validate_version(cls, v):
        """Validate configuration version."""
        if not v.startswith("1."):
            raise ValueError(f"Unsupported configuration version: {v}")
        return v

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="allow",
    )


class ConfigLoader:
    """
    Configuration loader with environment variable support.

    Supports loading from YAML/JSON files with environment variable
    overrides and validation. Uses path utilities for robust path handling.
    """

    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> ZaifTradeBotConfig:
        """
        Load configuration from YAML or JSON file.

        Args:
            file_path: Path to configuration file (relative to project root if not absolute)

        Returns:
            Validated configuration object
        """
        import json

        import yaml

        # Resolve path relative to project root if not absolute
        path = Path(file_path)
        if not path.is_absolute():
            path = get_project_root() / path

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        return ZaifTradeBotConfig(**data)

    @staticmethod
    def load_from_env(prefix: str = "ZTB_") -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.

        Args:
            prefix: Environment variable prefix

        Returns:
            Dictionary of configuration overrides
        """
        overrides = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(prefix) :].lower()
                keys = config_key.split("_")
                current = overrides
                for k in keys[:-1]:
                    current = current.setdefault(k, {})
                current[keys[-1]] = value
        return overrides

    @classmethod
    def load_config(
        cls,
        file_path: Union[str, Path, None] = None,
        env_prefix: str = "ZTB_",
    ) -> ZaifTradeBotConfig:
        """
        Load configuration from file with environment overrides.

        Args:
            file_path: Path to configuration file (relative to project root if not absolute).
                      If None, tries to find config files in standard locations.
            env_prefix: Environment variable prefix

        Returns:
            Validated configuration with environment overrides
        """
        # If no file path provided, try to find config files in standard locations
        if file_path is None:
            file_path = cls._find_config_file()

        config = cls.load_from_file(file_path)
        env_overrides = cls.load_from_env(env_prefix)

        if env_overrides:
            # Apply environment overrides
            config_dict = config.dict()
            cls._deep_update(config_dict, env_overrides)
            config = ZaifTradeBotConfig(**config_dict)

        return config

    @classmethod
    def _find_config_file(cls) -> Path:
        """
        Find configuration file in standard locations.

        Searches for config files in the following order:
        1. config/trading.yaml
        2. config/config.yaml
        3. config/default.yaml
        4. ztb_config.yaml (in project root)

        Returns:
            Path to the first found configuration file

        Raises:
            FileNotFoundError: If no configuration file is found
        """
        project_root = get_project_root()
        search_paths = [
            project_root / "config" / "trading.yaml",
            project_root / "config" / "config.yaml",
            project_root / "config" / "default.yaml",
            project_root / "ztb_config.yaml",
        ]

        for config_path in search_paths:
            if config_path.exists():
                return config_path

        raise FileNotFoundError(
            f"No configuration file found in standard locations: {[str(p) for p in search_paths]}"
        )

    @classmethod
    def save_config(
        cls,
        config: ZaifTradeBotConfig,
        file_path: Union[str, Path],
        format: str = "yaml",
    ) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration object to save
            file_path: Path to save configuration file (relative to project root if not absolute)
            format: File format ('yaml' or 'json')
        """
        import json

        import yaml

        # Resolve path relative to project root if not absolute
        path = Path(file_path)
        if not path.is_absolute():
            path = get_project_root() / path

        # Ensure parent directory exists
        ensure_dir(path.parent)

        # Convert config to dict
        config_dict = config.dict()

        # Save to file
        with open(path, "w", encoding="utf-8") as f:
            if format.lower() == "yaml":
                yaml.safe_dump(
                    config_dict, f, default_flow_style=False, sort_keys=False
                )
            elif format.lower() == "json":
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def create_default_config(
        cls, file_path: Union[str, Path] = "config/default.yaml"
    ) -> ZaifTradeBotConfig:
        """
        Create and save a default configuration file.

        Args:
            file_path: Path to save the default configuration

        Returns:
            Default configuration object
        """
        # Create default config with all default values
        config = ZaifTradeBotConfig()
        cls.save_config(config, file_path)
        return config

    @staticmethod
    def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if (
                isinstance(value, dict)
                and key in base_dict
                and isinstance(base_dict[key], dict)
            ):
                ConfigLoader._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Type aliases for backward compatibility
GlobalConfig = ZaifTradeBotConfig
__all__ = [
    "TrainingConfig",
    "CheckpointConfig",
    "StreamingConfig",
    "EvalConfig",
    "VenuePrecisionConfig",
    "RiskProfileConfig",
    "GlobalConfig",
    # Unified configuration system
    "DataConfig",
    "FeatureConfig",
    "FeaturesConfig",
    "RewardSettings",
    "EnvironmentConfig",
    "SACHyperparameters",
    "PPOHyperparameters",
    "TrainingConfig",
    "EvaluationThresholds",
    "EvaluationConfig",
    "LoggingConfig",
    "DeploymentConfig",
    "ZaifTradeBotConfig",
    "ConfigLoader",
]
