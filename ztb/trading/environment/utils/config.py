"""
Environment Configuration - EnvironmentConfig and RewardSettings.

This module contains configuration classes for the Heavy Trading Environment.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Union

from ztb.features.feature_set_manager import get_feature_set
from ztb.trading.constants import SAC_CONTINUOUS_THRESHOLD, SAC_CONTINUOUS_THRESHOLD_NEG
from ztb.training.config.ppo_config import (
    DEFAULT_MAX_CONSECUTIVE_TRADES,
    DEFAULT_MIN_HOLDING_PERIOD,
    DEFAULT_REWARD_CLIP_VALUE,
    DEFAULT_REWARD_CONSECUTIVE_TRADE_PENALTY,
    DEFAULT_REWARD_INVENTORY_PENALTY_SCALE,
    DEFAULT_REWARD_INVENTORY_WINDOW,
    DEFAULT_REWARD_MAX_CONSECUTIVE_TRADES,
    DEFAULT_REWARD_POSITION_PENALTY_EXPONENT,
    DEFAULT_REWARD_POSITION_PENALTY_SCALE,
    DEFAULT_REWARD_POSITION_SOFT_CAP,
    DEFAULT_REWARD_SCALING,
    DEFAULT_REWARD_SHARPE_BONUS_SCALE,
    DEFAULT_REWARD_TRADE_COOLDOWN_PENALTY,
    DEFAULT_REWARD_TRADE_COOLDOWN_STEPS,
    DEFAULT_REWARD_TRADE_FREQUENCY_HALFLIFE,
    DEFAULT_REWARD_TRADE_FREQUENCY_PENALTY,
    DEFAULT_REWARD_VOLATILITY_PENALTY_SCALE,
    DEFAULT_REWARD_VOLATILITY_WINDOW,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_STOP_LOSS_THRESHOLD,
)
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class RewardSettings:
    """Type-safe reward settings configuration."""

    use_simple_reward: bool = False
    reward_scale: float = 100.0
    trading_bonus: float = 0.01
    profit_bonuses: Dict[str, float] = dataclasses.field(default_factory=dict)
    penalty_coefficients: Dict[str, float] = dataclasses.field(default_factory=dict)
    entropy_bonus: float = 0.0
    custom_reward_params: Dict[str, float] = dataclasses.field(default_factory=dict)
    balance_penalty: float = 0.1
    balance_penalty_tolerance: float = 0.05
    profit_weight: float = 1.0
    risk_weight: float = 0.5
    consistency_weight: float = 0.2
    ultra_profit_multiplier: float = 2.0
    ultra_risk_multiplier: float = 0.5
    position_soft_cap: float = 0.5
    position_penalty_scale: float = 0.1
    position_penalty_exponent: float = 2.0
    inventory_window: int = 10
    inventory_penalty_scale: float = 0.01
    trade_frequency_penalty: float = 0.001
    trade_frequency_halflife: float = 100.0
    trade_cooldown_steps: int = 5
    trade_cooldown_penalty: float = 0.01
    max_consecutive_trades: int = 3
    consecutive_trade_penalty: float = 0.05
    volatility_window: int = 20
    volatility_penalty_scale: float = 0.01
    sharpe_bonus_scale: float = 0.01
    sortino_bonus_scale: float = 0.01
    calmar_bonus_scale: float = 0.005
    reward_clip_value: float = DEFAULT_REWARD_CLIP_VALUE
    reward_clip_min: float = -80.0
    reward_clip_max: float = 80.0
    profit_bonus_multipliers: List[float] = dataclasses.field(
        default_factory=lambda: [1.0, 1.5, 2.0]
    )
    enable_forced_diversity: bool = False
    curriculum_stage: str = "simple"


@dataclasses.dataclass
class EnvironmentConfig:
    """Configuration for HeavyTradingEnv with proper typing."""

    # Core settings
    reward_scaling: float = DEFAULT_REWARD_SCALING
    transaction_cost: float = 0.0
    max_position_size: float = 1.0
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    timeframe: str = "1m"
    feature_set: str = "full"
    feature_names: Optional[
        List[str]
    ] = None  # Explicit feature list (overrides feature_set)
    correlation_reduction: bool = True
    curriculum_stage: str = "forced_balance"
    feature_storage_dtype: str = "float16"
    precision_columns: List[str] = dataclasses.field(
        default_factory=lambda: ["close", "open", "high", "low", "volume"]
    )
    exchange: str = "coincheck"
    stop_loss_threshold: float = DEFAULT_STOP_LOSS_THRESHOLD
    max_consecutive_trades: int = DEFAULT_MAX_CONSECUTIVE_TRADES
    min_holding_period: int = DEFAULT_MIN_HOLDING_PERIOD

    # Reward settings
    reward_position_soft_cap: float = DEFAULT_REWARD_POSITION_SOFT_CAP
    reward_position_penalty_scale: float = DEFAULT_REWARD_POSITION_PENALTY_SCALE
    reward_position_penalty_exponent: float = DEFAULT_REWARD_POSITION_PENALTY_EXPONENT
    reward_inventory_window: int = DEFAULT_REWARD_INVENTORY_WINDOW
    reward_inventory_penalty_scale: float = DEFAULT_REWARD_INVENTORY_PENALTY_SCALE
    reward_trade_frequency_penalty: float = DEFAULT_REWARD_TRADE_FREQUENCY_PENALTY
    reward_trade_frequency_halflife: float = DEFAULT_REWARD_TRADE_FREQUENCY_HALFLIFE
    reward_trade_cooldown_steps: int = DEFAULT_REWARD_TRADE_COOLDOWN_STEPS
    reward_trade_cooldown_penalty: float = DEFAULT_REWARD_TRADE_COOLDOWN_PENALTY
    reward_max_consecutive_trades: int = DEFAULT_REWARD_MAX_CONSECUTIVE_TRADES
    reward_consecutive_trade_penalty: float = DEFAULT_REWARD_CONSECUTIVE_TRADE_PENALTY
    reward_volatility_window: int = DEFAULT_REWARD_VOLATILITY_WINDOW
    reward_volatility_penalty_scale: float = DEFAULT_REWARD_VOLATILITY_PENALTY_SCALE
    reward_sharpe_bonus_scale: float = DEFAULT_REWARD_SHARPE_BONUS_SCALE
    reward_clip_value: float = DEFAULT_REWARD_CLIP_VALUE
    enable_forced_diversity: bool = False

    # 🔧 CRITICAL FIX: 現実的な資金設定
    # Bitcoin価格 ≈ 18,000,000円 を考慮した設定
    # - 訓練用: 200,000円 (0.01 BTC程度購入可能、実口座の10-20倍で学習)
    # - 実取引用: 少額対応可能 (1 mBTC = 0.001 BTC ≈ 18,000円)
    # - 旧デフォルト: 1,000,000円では max_position_size=1.0 (1800万円) で取引不可能だった
    initial_portfolio_value: float = 200_000.0

    reward_profit_bonus_multipliers: List[float] = dataclasses.field(
        default_factory=lambda: [1.0, 1.0, 0.8]
    )
    reward_settings: Optional[
        Dict[str, Union[int, float, bool, str, List[Union[int, float, bool, str]]]]
    ] = None

    # Memory and performance settings
    memory_logging_enabled: bool = False
    memory_log_interval_steps: Optional[int] = None
    max_action_history: int = 512

    # Observation normalization
    use_standardized_observations: bool = (
        True  # Use StandardScaler for feature normalization
    )

    # Action space configuration
    use_continuous_actions: bool = False  # True for SAC, False for PPO
    action_space_type: Optional[str] = None
    target_feature_count: Optional[
        int
    ] = None  # Desired observation feature count when reducing correlations
    enable_action_masking: bool = False  # Only for discrete actions (PPO)
    continuous_to_discrete_threshold: float = (
        SAC_CONTINUOUS_THRESHOLD  # Threshold for SAC continuous→discrete conversion
    )
    continuous_to_discrete_threshold_neg: float = (
        SAC_CONTINUOUS_THRESHOLD_NEG  # Negative threshold for SELL conversion
    )

    # Signal guidance settings (v436 enhancement)
    signal_guidance_enabled: bool = False
    signal_guidance_mode: str = "partial"
    signal_bonus_weight: float = 0.1
    signal_penalty_weight: float = 0.05
    signal_weight: float = 1.0
    guidance_decay: float = 0.95

    # Market regime detection and adaptation settings (v443 enhancement)
    market_regime: Optional[Dict[str, Any]] = None
    advanced_market_regime: Optional[Dict[str, Any]] = None
    dynamic_reward_shaping: Optional[Dict[str, Any]] = None

    # Adaptive feature selection settings
    adaptive_feature_selection: Optional[Dict[str, Any]] = None
    allow_reverse: bool = True  # If False, SELL from Long/BUY from Short only closes position (no immediate reverse)
    enforce_reverse_cooldown: bool = (
        False  # If True, min_holding_period also applies to reversal entries
    )
    random_start: bool = (
        False  # If True, episodes start at random positions in the data
    )

    @classmethod
    def from_dict(
        cls,
        config_dict: Optional[
            Dict[str, Union[int, float, bool, str, List[Union[int, float, bool, str]]]]
        ] = None,
    ) -> "EnvironmentConfig":
        """Create config from dictionary, with defaults for missing values."""
        if config_dict is None:
            return cls()

        # Diagnostic logging: show whether the incoming dict explicitly contains use_continuous_actions
        try:
            logger.info(
                "EnvironmentConfig.from_dict received config type=%s, contains_use_continuous=%s",
                type(config_dict),
                (
                    "YES"
                    if (
                        isinstance(config_dict, dict)
                        and "use_continuous_actions" in config_dict
                    )
                    else "NO"
                ),
            )
            if isinstance(config_dict, dict):
                logger.info(
                    "EnvironmentConfig.from_dict preview use_continuous_actions=%s",
                    config_dict.get("use_continuous_actions", "NOT_PRESENT"),
                )
        except Exception:
            logger.exception("Failed to log EnvironmentConfig.from_dict diagnostic")

        # Convert dictionary to config, handling type conversions
        config_kwargs = {}
        known_fields = {field.name for field in dataclasses.fields(cls)}
        for field in dataclasses.fields(cls):
            if field.name in config_dict:
                value = config_dict[field.name]
                # Basic type conversion for common cases
                if field.name in [
                    "enable_forced_diversity",
                    "allow_reverse",
                    "enforce_reverse_cooldown",
                    "random_start",
                    "use_continuous_actions",
                    "enable_action_masking",
                    "use_standardized_observations",
                    "correlation_reduction",
                    "signal_guidance_enabled",
                ] and not isinstance(value, bool):
                    value = cls._as_bool(value)  # type: ignore[arg-type]
                # Basic type conversion for common cases
                if field.name in [
                    "enable_forced_diversity",
                    "allow_reverse",
                    "enforce_reverse_cooldown",
                    "random_start",
                    "use_continuous_actions",
                    "enable_action_masking",
                    "use_standardized_observations",
                    "correlation_reduction",
                    "signal_guidance_enabled",
                ] and not isinstance(value, bool):
                    value = cls._as_bool(value)  # type: ignore[arg-type]
                elif field.name in [
                    "max_consecutive_trades",
                    "min_holding_period",
                    "reward_inventory_window",
                    "reward_trade_cooldown_steps",
                    "reward_max_consecutive_trades",
                    "reward_volatility_window",
                    "target_feature_count",
                ] and isinstance(value, (float, str)):
                    try:
                        value = int(float(value))
                    except (ValueError, TypeError):
                        pass  # Keep original value
                elif field.name in [
                    "reward_scaling",
                    "transaction_cost",
                    "max_position_size",
                    "risk_free_rate",
                    "stop_loss_threshold",
                    "reward_position_soft_cap",
                    "reward_position_penalty_scale",
                    "reward_position_penalty_exponent",
                    "reward_inventory_penalty_scale",
                    "reward_trade_frequency_penalty",
                    "reward_trade_frequency_halflife",
                    "reward_trade_cooldown_penalty",
                    "reward_consecutive_trade_penalty",
                    "reward_volatility_penalty_scale",
                    "reward_sharpe_bonus_scale",
                    "reward_clip_value",
                    "initial_portfolio_value",
                    "continuous_to_discrete_threshold",
                    "continuous_to_discrete_threshold_neg",
                    "signal_bonus_weight",
                    "signal_penalty_weight",
                    "signal_weight",
                    "guidance_decay",
                ] and isinstance(value, str):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        pass  # Keep original value
                config_kwargs[field.name] = value
            # Field will use default if not in config_dict

        # Also process any known fields that might be in nested structures
        for key, value in config_dict.items():
            if key in known_fields and key not in config_kwargs:
                # Basic type conversion for known fields
                if key in [
                    "enable_forced_diversity",
                    "allow_reverse",
                    "enforce_reverse_cooldown",
                    "random_start",
                    "use_continuous_actions",
                    "enable_action_masking",
                    "use_standardized_observations",
                    "correlation_reduction",
                    "signal_guidance_enabled",
                ] and not isinstance(value, bool):
                    value = cls._as_bool(value)  # type: ignore[arg-type]
                elif key in [
                    "max_consecutive_trades",
                    "min_holding_period",
                    "reward_inventory_window",
                    "reward_trade_cooldown_steps",
                    "reward_max_consecutive_trades",
                    "reward_volatility_window",
                    "target_feature_count",
                ] and isinstance(value, (float, str)):
                    try:
                        value = int(float(value))
                    except (ValueError, TypeError):
                        pass  # Keep original value
                elif key in [
                    "reward_scaling",
                    "transaction_cost",
                    "max_position_size",
                    "risk_free_rate",
                    "stop_loss_threshold",
                    "reward_position_soft_cap",
                    "reward_position_penalty_scale",
                    "reward_position_penalty_exponent",
                    "reward_inventory_penalty_scale",
                    "reward_trade_frequency_penalty",
                    "reward_trade_frequency_halflife",
                    "reward_trade_cooldown_penalty",
                    "reward_consecutive_trade_penalty",
                    "reward_volatility_penalty_scale",
                    "reward_sharpe_bonus_scale",
                    "reward_clip_value",
                    "initial_portfolio_value",
                    "continuous_to_discrete_threshold",
                    "continuous_to_discrete_threshold_neg",
                    "signal_bonus_weight",
                    "signal_penalty_weight",
                    "signal_weight",
                    "guidance_decay",
                ] and isinstance(value, str):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        pass  # Keep original value
                config_kwargs[key] = value

        # Handle nested training.environment section
        if "training" in config_dict and isinstance(config_dict["training"], dict):
            training_config = config_dict["training"]
            # Support both flattened (training.environment.<fields>) and
            # extra-nested (training.environment.config.<fields>) layouts.
            env_section = None
            if "environment" in training_config and isinstance(
                training_config["environment"], dict
            ):
                env_section = training_config["environment"]

            # If there's an inner 'config' dict (v4xx converter sometimes nests under .config)
            if (
                env_section
                and "config" in env_section
                and isinstance(env_section["config"], dict)
            ):
                # Merge inner config taking precedence over direct env_section keys
                inner = dict(env_section.get("config", {}))
                # shallow-merge so inner config overrides
                merged_env = dict(env_section)
                merged_env.update(inner)
                env_config = merged_env
            else:
                env_config = env_section

            if isinstance(env_config, dict):
                # Copy environment config values to top level for processing
                for key, value in env_config.items():
                    if (
                        key in known_fields and key not in config_kwargs
                    ):  # Don't override if already set
                        config_kwargs[key] = value
                    elif key == "signal_guidance" and isinstance(
                        value, dict
                    ):  # Special handling for signal_guidance
                        config_kwargs[key] = value
                    elif key == "market_regime" and isinstance(
                        value, dict
                    ):  # Special handling for market_regime
                        config_kwargs[key] = value
                    elif key == "dynamic_reward_shaping" and isinstance(
                        value, dict
                    ):  # Special handling for dynamic_reward_shaping
                        config_kwargs[key] = value

        # Handle field name mappings
        if (
            "initial_balance" in config_kwargs
            and "initial_portfolio_value" not in config_kwargs
        ):
            config_kwargs["initial_portfolio_value"] = float(
                config_kwargs["initial_balance"]
            )
        # Remove deprecated field names
        config_kwargs.pop("initial_balance", None)
        signal_guidance_config = None
        if "signal_guidance" in config_kwargs and isinstance(
            config_kwargs["signal_guidance"], dict
        ):
            signal_guidance_config = config_kwargs["signal_guidance"]
        elif "signal_guidance" in config_dict and isinstance(
            config_dict["signal_guidance"], dict
        ):
            signal_guidance_config = config_dict["signal_guidance"]

        if signal_guidance_config:
            enabled = cls._as_bool(signal_guidance_config.get("enabled", False))
            config_kwargs["signal_guidance_enabled"] = enabled
            config_kwargs["signal_guidance_mode"] = signal_guidance_config.get(
                "guidance_mode", "partial"
            )
            config_kwargs["signal_bonus_weight"] = float(
                signal_guidance_config.get("signal_bonus_weight", 0.1)
            )
            config_kwargs["signal_penalty_weight"] = float(
                signal_guidance_config.get("signal_penalty_weight", 0.05)
            )
            config_kwargs["guidance_decay"] = float(
                signal_guidance_config.get("guidance_decay", 0.95)
            )

        # Apply curated feature set if specified
        feature_set = config_dict.get("feature_set", "full")
        if feature_set == "curated" and "feature_names" not in config_dict:
            try:
                curated_features = get_feature_set("curated")
                config_kwargs["feature_names"] = curated_features
                logger.info(
                    f"Applied curated feature set with {len(curated_features)} features"
                )
            except Exception as e:
                logger.warning(f"Failed to load curated features: {e}")

        # Handle reward_settings
        if "reward_settings" in config_dict and isinstance(
            config_dict["reward_settings"], dict
        ):
            reward_settings_dict = config_dict["reward_settings"]
            try:
                config_kwargs["reward_settings"] = RewardSettings(
                    **reward_settings_dict
                )
            except TypeError:
                # Be tolerant of extra/unknown keys in configs (forward compatibility)
                rs = RewardSettings()
                for k, v in reward_settings_dict.items():
                    if hasattr(rs, k):
                        try:
                            setattr(rs, k, v)
                        except Exception:
                            # skip invalid assignments
                            pass
                    else:
                        # Store unknown keys in custom_reward_params for inspection
                        try:
                            if isinstance(v, (int, float)):
                                rs.custom_reward_params[k] = float(v)
                            else:
                                rs.custom_reward_params[k] = v
                        except Exception:
                            rs.custom_reward_params[k] = v
                config_kwargs["reward_settings"] = rs

        return cls(**config_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _as_bool(
        value: Union[bool, int, float, str, None], default: bool = False
    ) -> bool:
        """Convert various types to boolean."""
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        value_str = str(value).strip().lower()
        if value_str in {"true", "1", "yes", "y", "on"}:
            return True
        if value_str in {"false", "0", "no", "n", "off"}:
            return False
        return default

    def as_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary for compatibility."""
        return dataclasses.asdict(self)


__all__ = ["EnvironmentConfig", "RewardSettings"]
