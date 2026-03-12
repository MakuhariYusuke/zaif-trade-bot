"""
Environment Configuration - EnvironmentConfig and RewardSettings.

This module contains configuration classes for the Heavy Trading Environment.
"""

import dataclasses
from typing import Any

from ztb.trading.constants import SAC_CONTINUOUS_THRESHOLD, SAC_CONTINUOUS_THRESHOLD_NEG
from ztb.trading.environment.utils.domain_randomizer import DomainRandomizationConfig
from ztb.trading.environment.utils.exchange_profile import ExchangeProfile
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
    profit_bonuses: dict[str, float] = dataclasses.field(default_factory=dict)
    penalty_coefficients: dict[str, float] = dataclasses.field(default_factory=dict)
    entropy_bonus: float = 0.0
    # v453 Hybrid Strategy
    hybrid_config: dict[str, Any] | None = None
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
    # Additional multiplicative penalty/bonus scalars for specific actions
    hold_penalty_multiplier: float = 1.0
    trade_frequency_bonus: float = 0.0
    
    volatility_window: int = 20
    volatility_penalty_scale: float = 0.01
    sharpe_bonus_scale: float = 0.01
    sortino_bonus_scale: float = 0.01
    calmar_bonus_scale: float = 0.005
    reward_clip_value: float = DEFAULT_REWARD_CLIP_VALUE
    reward_clip_min: float = -80.0
    reward_clip_max: float = 80.0
    profit_bonus_multipliers: list[float] = dataclasses.field(
        default_factory=lambda: [1.0, 1.5, 2.0]
    )
    enable_forced_diversity: bool = False
    curriculum_stage: str = "simple"
    unrealized_loss_penalty_enabled: bool = False
    unrealized_loss_penalty_base: float = 1.1
    unrealized_loss_penalty_max_steps: int = 10
    asymmetric_reward_scaling: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "long_position_reward_multiplier": 1.0,
            "short_position_reward_multiplier": 1.0,
            "long_position_penalty_multiplier": 1.0,
            "short_position_penalty_multiplier": 1.0,
        }
    )

    # Accept dynamic shaping configuration (backwards compatibility for v440 tests)
    dynamic_reward_shaping: dict[str, Any] | None = None

    # Backwards-compatible extension point to hold unknown/experimental reward keys
    custom_reward_params: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Defaults for behavioral penalty lookbacks (used by BehavioralPenaltyCalculator)
    consistency_lookback: int = 50
    skewness_lookback: int = 10
    action_entropy_lookback: int = 10
    consistency_min_actions: int = 2

    # Additional attributes for backward compatibility
    action_balance_target: float = 0.5
    entropy_regularization: float = 0.0
    action_smoothing: float = 0.0
    consistency_penalty: float = 0.0
    redundant_trade_penalty: float = 0.0

    # Backwards compatibility alias - some configs refer to 'reward_scaling'
    reward_scaling: float | None = None

    # Backwards compatibility for v440-style long/short multipliers
    long_position_reward_multiplier: float = 1.0
    short_position_reward_multiplier: float = 1.0
    long_position_penalty_multiplier: float = 1.0
    short_position_penalty_multiplier: float = 1.0

    def __post_init__(self):
        # If legacy 'reward_scaling' provided, map it to 'reward_scale'
        try:
            if self.reward_scaling is not None:
                self.reward_scale = float(self.reward_scaling)
        except Exception:
            pass

        # If multipliers were provided as top-level fields, ensure the asymmetric map reflects them
        try:
            if isinstance(self.asymmetric_reward_scaling, dict):
                self.asymmetric_reward_scaling["long_position_reward_multiplier"] = float(
                    self.long_position_reward_multiplier
                )
                self.asymmetric_reward_scaling["short_position_reward_multiplier"] = float(
                    self.short_position_reward_multiplier
                )
                self.asymmetric_reward_scaling["long_position_penalty_multiplier"] = float(
                    self.long_position_penalty_multiplier
                )
                self.asymmetric_reward_scaling["short_position_penalty_multiplier"] = float(
                    self.short_position_penalty_multiplier
                )
        except Exception:
            pass

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RewardSettings":
        """Create RewardSettings from dictionary."""
        # Filter out keys that are not in the dataclass
        valid_keys = {field.name for field in dataclasses.fields(cls)}
        
        # Split into known and unknown parameters
        filtered_dict = {}
        custom_params = {}
        
        for k, v in config_dict.items():
            if k in valid_keys:
                filtered_dict[k] = v
            elif k != "custom_reward_params":
                custom_params[k] = v
                
        # If the input dict already had custom_reward_params, merge it
        if "custom_reward_params" in config_dict:
            custom_params.update(config_dict["custom_reward_params"])
            
        # Create instance with known parameters
        instance = cls(**filtered_dict)
        
        # set custom parameters
        if custom_params:
            instance.custom_reward_params = custom_params
            
        return instance

@dataclasses.dataclass
class EnvironmentConfig:
    """Configuration for HeavyTradingEnv with proper typing."""

    # Core settings
    # 386# FIX: reward_scaling default を PPO 値 6.0 → 中立値 1.0 に変更。
    # PPO/SAC 各 YAML で明示的に設定すること。
    reward_scaling: float = 1.0
    transaction_cost: float = 0.0
    commission: float = 0.0  # Alias for transaction_cost for backward compatibility
    slippage: float = 0.0  # Slippage cost
    exchange_profile: ExchangeProfile | None = None  # Exchange profile (fees, liquidity, etc.)
    execution_model: dict[str, Any] | bool | None = None  # Execution model config (Phase 3)
    risk_management: dict[str, Any] = dataclasses.field(default_factory=dict)
    domain_randomization: DomainRandomizationConfig | None = None  # Domain randomization config
    max_steps: int | None = None  # Maximum steps per episode
    train_end_index: int | None = None  # End index for training data subset
    debug_internal_state: bool = False  # Enable debug information in step output
    max_position_size: float = 1.0
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    timeframe: str = "1m"
    feature_set: str = "full"
    feature_names: list[str] | None = None  # Explicit feature list (overrides feature_set)
    scaler_mean: list[float] | None = None  # Optional schema-provided observation scaler
    scaler_std: list[float] | None = None  # Optional schema-provided observation scaler
    correlation_reduction: bool = True
    curriculum_stage: str | None = None  # set from training.curriculum_learning
    curriculum_learning: dict[str, Any] | None = None
    feature_storage_dtype: str = "float16"
    precision_columns: list[str] = dataclasses.field(
        default_factory=lambda: ["close", "open", "high", "low", "volume"]
    )
    exchange: str = "coincheck"
    stop_loss_threshold: float = DEFAULT_STOP_LOSS_THRESHOLD
    max_consecutive_trades: int = DEFAULT_MAX_CONSECUTIVE_TRADES
    min_holding_period: int = DEFAULT_MIN_HOLDING_PERIOD

    # v453 Hybrid Strategy
    hybrid_config: dict[str, Any] | None = None

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

    # Action bonuses and penalties
    action_bonuses: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "buy_action_bonus": 0.0,
            "sell_action_bonus": 0.0,
            "hold_action_bonus": 0.0,
        }
    )
    base_action_penalty: float = 0.015
    behavior_optimization: dict[str, Any] = dataclasses.field(default_factory=dict)
    logging: dict[str, Any] = dataclasses.field(default_factory=dict)

    # 🔧 CRITICAL FIX: 現実的な資金設定
    # Bitcoin価格 ≈ 18,000,000円 を考慮した設定
    # - 訓練用: 200,000円 (0.01 BTC程度購入可能、実口座の10-20倍で学習)
    # - 実取引用: 少額対応可能 (1 mBTC = 0.001 BTC ≈ 18,000円)
    # - 旧デフォルト: 1,000,000円では max_position_size=1.0 (1800万円) で取引不可能だった
    initial_portfolio_value: float = 200_000.0

    reward_profit_bonus_multipliers: list[float] = dataclasses.field(
        default_factory=lambda: [1.0, 1.0, 0.8]
    )
    reward_settings: RewardSettings | None = None

    # Memory and performance settings
    memory_logging_enabled: bool = False
    memory_log_interval_steps: int | None = None
    max_action_history: int = 512

    # Observation normalization
    use_standardized_observations: bool = (
        True  # Use StandardScaler for feature normalization
    )

    # Action space configuration
    use_continuous_actions: bool = False  # True for SAC, False for PPO
    action_space_type: str | None = None
    target_feature_count: int | None = None  # Desired observation feature count when reducing correlations
    enable_action_masking: bool = False  # Only for discrete actions (PPO)
    embed_action_masks: bool = False  # P7: Embed action_masks [3] into observation for SAC
    continuous_to_discrete_threshold: float = (
        SAC_CONTINUOUS_THRESHOLD  # Threshold for SAC continuous→discrete conversion
    )
    continuous_to_discrete_threshold_neg: float = (
        SAC_CONTINUOUS_THRESHOLD_NEG  # Negative threshold for SELL conversion
    )

    # Adaptive Thresholding (v449)
    adaptive_threshold_mode: bool = False
    threshold_volatility_multiplier: float = 1.0
    min_action_threshold: float = 0.001

    # Bankruptcy and Drawdown settings
    bankruptcy_threshold: float = 2000.0
    bankruptcy_penalty: float = 1000.0
    drawdown_penalty_threshold: float = 0.20  # 20% drawdown
    drawdown_penalty_factor: float = 0.1  # Penalty multiplier for drawdown
    max_action_threshold: float = 1.0

    # Signal guidance settings (v436 enhancement)
    signal_guidance_enabled: bool = False
    signal_guidance_mode: str = "partial"
    signal_bonus_weight: float = 0.1
    signal_penalty_weight: float = 0.05
    signal_weight: float = 1.0
    guidance_decay: float = 0.95

    # Market regime detection and adaptation settings (v443 enhancement)
    market_regime: dict[str, Any] | None = None
    advanced_market_regime: dict[str, Any] | None = None
    dynamic_reward_shaping: dict[str, Any] | None = None

    # Adaptive feature selection settings
    adaptive_feature_selection: dict[str, Any] | None = None
    allow_reverse: bool = True  # If False, SELL from Long/BUY from Short only closes position (no immediate reverse)
    enforce_reverse_cooldown: bool = (
        False  # If True, min_holding_period also applies to reversal entries
    )
    random_start: bool = (
        False  # If True, episodes start at random positions in the data
    )

    def __post_init__(self) -> None:
        """Initialize derived fields and backward compatibility."""
        # Sync commission alias
        if self.commission != 0.0 and self.transaction_cost == 0.0:
            self.transaction_cost = self.commission

        # Initialize exchange profile if not provided
        if self.exchange_profile is None:
            # Check if we should use ExchangeFeeModel (if exchange is specific and cost is 0)
            from ztb.utils.fee_model import ExchangeFeeModel, FeeModel, FixedFeeModel

            fee_model: FeeModel | None = None
            fee_rate = self.transaction_cost

            # If transaction cost is 0 (default) and we have a specific exchange, try to use its defaults
            if fee_rate == 0.0 and self.exchange in [
                "binance",
                "bitflyer",
                "coincheck",
            ]:
                try:
                    exch_model = ExchangeFeeModel()
                    exch_model.set_exchange(self.exchange)
                    fee_model = exch_model
                    # Update legacy cost for compatibility
                    fee_rate = exch_model.get_fee_rate("buy")
                    self.transaction_cost = fee_rate
                except Exception:
                    pass

            if fee_model is None:
                fee_model = FixedFeeModel(buy_fee_rate=fee_rate, sell_fee_rate=fee_rate)

            self.exchange_profile = ExchangeProfile(
                name=self.exchange,
                fee_model=fee_model,
                slippage_rate=self.slippage,
                maker_fee_rate=fee_rate,
                taker_fee_rate=fee_rate,
            )
        else:
            # If profile is provided, ensure legacy params are consistent
            if (
                self.transaction_cost == 0.0
                and self.exchange_profile.taker_fee_rate > 0
            ):
                self.transaction_cost = self.exchange_profile.taker_fee_rate

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "EnvironmentConfig":
        """
        Create EnvironmentConfig from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            EnvironmentConfig instance
        """
        import logging

        logger = logging.getLogger(__name__)

        logger.debug(
            f"EnvironmentConfig.from_dict called with config_dict keys: {list(config_dict.keys())}"
        )

        training_section = config_dict.get("training")
        if isinstance(training_section, dict):
            training_env = training_section.get("environment")
            if isinstance(training_env, dict):
                normalized_env = dict(training_env)
                nested_env_config = normalized_env.pop("config", None)
                if isinstance(nested_env_config, dict):
                    normalized_env = {**normalized_env, **nested_env_config}

                config_dict = {
                    **{k: v for k, v in config_dict.items() if k != "training"},
                    "environment": normalized_env,
                }

        # Create instance with defaults
        instance = cls()

        # First pass: collect individual action bonus keys at root level
        root_level_bonuses = {}
        for bonus_key in ["buy_action_bonus", "sell_action_bonus", "hold_action_bonus"]:
            if bonus_key in config_dict:
                root_level_bonuses[bonus_key] = float(config_dict[bonus_key])

        # Extract behavior_optimization from nested environment config if present
        behavior_opt = None
        if "environment" in config_dict and isinstance(
            config_dict["environment"], dict
        ):
            env_config = config_dict["environment"]
            if "behavior_optimization" in env_config and isinstance(
                env_config["behavior_optimization"], dict
            ):
                behavior_opt = env_config["behavior_optimization"]
                logger.debug(
                    f"Found behavior_optimization in environment: {behavior_opt}"
                )

        # Also check for behavior_optimization at root level (backward compatibility)
        if (
            behavior_opt is None
            and "behavior_optimization" in config_dict
            and isinstance(config_dict["behavior_optimization"], dict)
        ):
            behavior_opt = config_dict["behavior_optimization"]
            logger.debug(f"Found behavior_optimization at root level: {behavior_opt}")

        # Handle behavior_optimization dict
        if behavior_opt is not None:
            # 386# FIX: Store raw dict so reward_calculator can access config.behavior_optimization
            instance.behavior_optimization = behavior_opt
            if not instance.reward_settings:
                instance.reward_settings = RewardSettings()
            # Map behavior_optimization keys to reward_settings
            if isinstance(instance.reward_settings, RewardSettings):
                if "action_balance_target" in behavior_opt:
                    instance.reward_settings.action_balance_target = float(
                        behavior_opt["action_balance_target"]
                    )
                if "balance_penalty" in behavior_opt:
                    instance.reward_settings.balance_penalty = float(
                        behavior_opt["balance_penalty"]
                    )
                if "entropy_regularization" in behavior_opt:
                    instance.reward_settings.entropy_regularization = float(
                        behavior_opt["entropy_regularization"]
                    )
                if "action_smoothing" in behavior_opt:
                    instance.reward_settings.action_smoothing = float(
                        behavior_opt["action_smoothing"]
                    )
                if "consistency_penalty" in behavior_opt:
                    instance.reward_settings.consistency_penalty = float(
                        behavior_opt["consistency_penalty"]
                    )
                if "redundant_trade_penalty" in behavior_opt:
                    instance.reward_settings.redundant_trade_penalty = float(
                        behavior_opt["redundant_trade_penalty"]
                    )

        # Update fields from config_dict
        bool_fields = {
            "use_continuous_actions",
            "allow_reverse",
            "enforce_reverse_cooldown",
            "debug_internal_state",
            "correlation_reduction",
            "enable_forced_diversity",
            "memory_logging_enabled",
        }
        for key, value in config_dict.items():
            if isinstance(key, dict):
                logger.error(
                    f"Found dict as key in config_dict (2nd from_dict): key={key}, value={value}"
                )
                raise TypeError(
                    f"unhashable type: 'dict' - found dict as key in 2nd from_dict: {key}"
                )
            # Key is valid, continue processing
            if key == "environment" and isinstance(value, dict):
                logger.info(f"Environment keys: {list(value.keys())}")
                for env_key, env_value in value.items():
                    if isinstance(env_key, dict):
                        logger.error(
                            f"Found dict as env_key: env_key={env_key}, env_value={env_value}"
                        )
                        raise TypeError(
                            f"unhashable type: 'dict' - found dict as env_key: {env_key}"
                        )
                    logger.debug(
                        f"Processing env_key: {env_key}, type: {type(env_key)}, env_value type: {type(env_value)}"
                    )
                    if env_key == "behavior_optimization":
                        # Already handled above
                        continue
                    elif env_key == "action_bonuses" and isinstance(env_value, dict):
                        # Handle nested action_bonuses
                        if (
                            not root_level_bonuses
                        ):  # Don't override root-level if present
                            converted_bonuses = {}
                            for bonus_key, bonus_value in env_value.items():
                                if isinstance(bonus_key, dict):
                                    logger.error(
                                        f"Found dict as bonus_key: bonus_key={bonus_key}, bonus_value={bonus_value}"
                                    )
                                    raise TypeError(
                                        f"unhashable type: 'dict' - found dict as bonus_key: {bonus_key}"
                                    )
                                logger.debug(
                                    f"Processing bonus_key: {bonus_key}, type: {type(bonus_key)}, bonus_value type: {type(bonus_value)}"
                                )
                                converted_bonuses[bonus_key] = float(bonus_value)
                            instance.action_bonuses = converted_bonuses
                    elif env_key == "behavioral_penalty" and isinstance(
                        env_value, dict
                    ):
                        # Map behavioral_penalty keys into instance.reward_settings dataclass if possible
                        if not instance.reward_settings:
                            instance.reward_settings = RewardSettings()
                        # Update reward_settings with behavioral_penalty values
                        for bp_k, bp_v in env_value.items():
                            if hasattr(instance.reward_settings, bp_k):
                                try:
                                    setattr(instance.reward_settings, bp_k, bp_v)
                                except Exception:
                                    # fallback to custom params
                                    instance.reward_settings.custom_reward_params[
                                        bp_k
                                    ] = bp_v
                            else:
                                instance.reward_settings.custom_reward_params[
                                    bp_k
                                ] = bp_v
                    elif hasattr(instance, env_key):
                        # Map other environment keys to instance
                        try:
                            if env_key in [
                                "base_action_penalty",
                                "commission",
                                "slippage",
                            ]:
                                setattr(instance, env_key, float(env_value))
                            elif env_key == "max_position_size":
                                setattr(instance, env_key, float(env_value))
                            elif env_key == "max_action_threshold":
                                setattr(instance, env_key, float(env_value))
                                logger.info(
                                    f"Updated max_action_threshold to {env_value}"
                                )
                            elif env_key in bool_fields:
                                setattr(instance, env_key, cls._as_bool(env_value))
                            else:
                                setattr(instance, env_key, env_value)
                            logger.debug(f"set environment.{env_key} = {env_value}")
                        except Exception as e:
                            logger.debug(f"Could not set environment.{env_key}: {e}")
            elif key == "behavior_optimization":
                # Already handled above
                continue
            elif key == "exchange_profile" and isinstance(value, dict):
                profile = ExchangeProfile.from_dict(value)
                setattr(instance, key, profile)
                # Sync legacy params
                if instance.transaction_cost == 0.0 and profile.taker_fee_rate > 0:
                    instance.transaction_cost = profile.taker_fee_rate
            elif key == "domain_randomization" and isinstance(value, dict):
                setattr(instance, key, DomainRandomizationConfig.from_dict(value))
            elif hasattr(instance, key):
                logger.debug(f"Setting {key} = {value}")
                try:
                    if key == "action_bonuses" and isinstance(value, dict):
                        # Handle action_bonuses dict
                        converted_bonuses = {}
                        for bonus_key, bonus_value in value.items():
                            if isinstance(bonus_key, dict):
                                logger.error(
                                    f"Found dict as bonus_key (2): bonus_key={bonus_key}, bonus_value={bonus_value}"
                                )
                                raise TypeError(
                                    f"unhashable type: 'dict' - found dict as bonus_key (2): {bonus_key}"
                                )
                            logger.debug(
                                f"Processing bonus_key (2): {bonus_key}, type: {type(bonus_key)}, bonus_value type: {type(bonus_value)}"
                            )
                            converted_bonuses[bonus_key] = float(bonus_value)
                        setattr(instance, key, converted_bonuses)
                    elif key == "reward_settings" and isinstance(value, dict):
                        # Handle reward_settings dict
                        # 386# FIX: Merge instead of replace to preserve behavior_optimization overrides
                        new_rs = RewardSettings.from_dict(value)
                        if instance.reward_settings is not None:
                            from ztb.utils.dataclass_utils import shallow_asdict
                            existing = shallow_asdict(instance.reward_settings)
                            new_dict = shallow_asdict(new_rs)
                            # new_rs takes priority, but preserve existing non-default values
                            # that were set by behavior_optimization
                            defaults = shallow_asdict(RewardSettings())
                            for k, v in existing.items():
                                if k not in new_dict or new_dict.get(k) == defaults.get(k):
                                    if v != defaults.get(k):
                                        new_dict[k] = v
                            instance.reward_settings = RewardSettings.from_dict(new_dict)
                        else:
                            instance.reward_settings = new_rs
                    elif key in ["base_action_penalty", "commission", "slippage"]:
                        # Handle float fields
                        setattr(instance, key, float(value))
                    elif key == "max_steps":
                        # Handle int fields
                        setattr(
                            instance,
                            key,
                            int(float(value)) if value is not None else None,
                        )
                    elif key in bool_fields:
                        setattr(instance, key, cls._as_bool(value))
                    else:
                        # Default assignment
                        setattr(instance, key, value)
                except Exception as e:
                    logger.error(f"Failed to set {key} = {value}: {e}")
            elif key not in [
                "buy_action_bonus",
                "sell_action_bonus",
                "hold_action_bonus",
            ]:
                # Skip individual bonus keys (they're handled separately)
                logger.debug(f"Skipping config key (not in EnvironmentConfig): {key}")

        # Merge root-level bonuses into action_bonuses if they were found
        if root_level_bonuses:
            logger.debug(f"Merging root-level action bonuses: {root_level_bonuses}")
            if not instance.action_bonuses:
                instance.action_bonuses = {}
            instance.action_bonuses.update(root_level_bonuses)

        # Fee/commission override handling:
        # `EnvironmentConfig.__post_init__` syncs commission->transaction_cost and builds an ExchangeProfile,
        # but `from_dict()` applies overrides after instantiation, so we must re-apply fee overrides here.
        try:
            env_cfg_for_fee = None
            if "environment" in config_dict and isinstance(
                config_dict["environment"], dict
            ):
                env_cfg_for_fee = config_dict["environment"]
            elif isinstance(config_dict, dict):
                env_cfg_for_fee = config_dict

            explicit_fee_rate = None
            if isinstance(env_cfg_for_fee, dict):
                if (
                    "transaction_cost" in env_cfg_for_fee
                    and env_cfg_for_fee.get("transaction_cost") is not None
                ):
                    explicit_fee_rate = float(
                        env_cfg_for_fee.get("transaction_cost") or 0.0
                    )
                elif (
                    "commission" in env_cfg_for_fee
                    and env_cfg_for_fee.get("commission") is not None
                ):
                    explicit_fee_rate = float(env_cfg_for_fee.get("commission") or 0.0)

            if explicit_fee_rate is not None and explicit_fee_rate > 0.0:
                instance.transaction_cost = explicit_fee_rate
                instance.commission = explicit_fee_rate
                try:
                    from ztb.utils.fee_model import FixedFeeModel

                    if instance.exchange_profile is not None:
                        instance.exchange_profile.fee_model = FixedFeeModel(
                            buy_fee_rate=explicit_fee_rate,
                            sell_fee_rate=explicit_fee_rate,
                        )
                        instance.exchange_profile.maker_fee_rate = explicit_fee_rate
                        instance.exchange_profile.taker_fee_rate = explicit_fee_rate
                except Exception:
                    pass
        except Exception:
            pass

        if instance.action_space_type == "continuous":
            instance.use_continuous_actions = True
        elif instance.action_space_type == "discrete":
            instance.use_continuous_actions = False
        elif instance.use_continuous_actions:
            instance.action_space_type = "continuous"
        else:
            instance.action_space_type = "discrete"

        logger.debug(
            f"EnvironmentConfig.from_dict completed: base_action_penalty={instance.base_action_penalty}, action_bonuses={instance.action_bonuses}"
        )
        return instance

    @staticmethod
    def _as_bool(
        value: bool | int | float | str | None, default: bool = False
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

    def as_dict(self) -> dict[str, Any]:
        """Convert config back to dictionary for compatibility."""
        return dataclasses.asdict(self)

__all__ = ["EnvironmentConfig", "RewardSettings"]
