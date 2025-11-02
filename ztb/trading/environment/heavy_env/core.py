"""Core trading environment implementations."""

from __future__ import annotations

import dataclasses
import gc
import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import psutil
from numpy.typing import NDArray

# Import v444 regime classifier for advanced market regime adaptation
from ztb.analysis.market_regime_classifier import MarketRegimeClassifier
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL

# Import components for runtime use
from ztb.trading.environment.components.action_executor import ActionExecutor
from ztb.trading.environment.components.data_manager import DataManager
from ztb.trading.environment.components.statistics_calculator import (
    StatisticsCalculator,
)
from ztb.trading.environment.constants import (
    ACTION_COUNTS_INITIAL,
    POSITION_EPSILON,
    RANDOM_START_BUFFER_RATIO,
    RANDOM_START_MAX_BUFFER,
    RANDOM_START_MIN_BUFFER,
)
from ztb.trading.environment.heavy_env.components.state_manager import StateManager
from ztb.trading.environment.heavy_env.components.validation_manager import (
    ValidationManager,
)
from ztb.trading.environment.heavy_env.mixins.initialization import (
    _build_fast_access_buffers,
    _compute_scaler_from_data,
    _extract_numeric_column,
    _initialize_components,
    _initialize_data,
    _initialize_data_structures,
    _initialize_features_and_spaces,
    _initialize_remaining_components,
    _refresh_features,
    _select_features_by_correlation_in_env,
    _setup_scaler,
)
from ztb.trading.environment.heavy_env.mixins.pricing import (
    _resolve_atr,
    _resolve_price,
)
from ztb.trading.environment.heavy_env.mixins.reward import (
    _get_reward_setting_bool,
    _get_reward_setting_float,
    _get_reward_setting_int,
)
from ztb.trading.environment.heavy_env.mixins.streaming import (
    _append_streaming_rows,
    _ensure_data_available,
    _fetch_streaming_snapshot,
    _prepare_stream_batch,
    _prime_streaming_data,
)
from ztb.trading.environment.types import StatisticsDict
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.types.protocols import TradingEnvironment
from ztb.utils.errors import ConfigurationError, ValidationError
from ztb.utils.fee_model import ExchangeFeeModel
from ztb.utils.logging_utils import get_logger
from ztb.utils.type_validation import TypeValidator

if TYPE_CHECKING:
    from ztb.data.streaming_pipeline import StreamingPipeline
    from ztb.features.adaptive_selection import AdaptiveFeatureSelector
    from ztb.trading.environment.components.action_validator import ActionValidator
    from ztb.trading.environment.components.data_processor import DataProcessor
    from ztb.trading.environment.components.memory_manager import MemoryManager
    from ztb.trading.environment.components.observation_builder import (
        ObservationBuilder,
    )
    from ztb.trading.environment.components.position_manager import PositionManager
    from ztb.trading.environment.components.reward_calculator import RewardCalculator
    from ztb.trading.environment.components.streaming_handler import StreamingHandler
    from ztb.trading.live.data.stream_to_bars import StreamToBarsConverter

logger = get_logger(__name__)


def deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value

    return result


class HeavyTradingEnv(
    gym.Env[NDArray[np.float32], int],
    TradingEnvironment,
):
    """Heavy Feature-Based Trading Environment for Reinforcement Learning."""

    # Class constants for better maintainability
    DEFAULT_MEMORY_LOG_INTERVAL = 2000
    DEFAULT_GC_STEP_INTERVAL = 1000
    DEFAULT_MAX_HISTORY_LENGTH = 512
    DEFAULT_MAX_ACTION_HISTORY = 256
    DEFAULT_INVENTORY_WINDOW = 64
    DEFAULT_VOLATILITY_WINDOW = 16
    DEFAULT_MIN_WINDOW_SIZE = 8
    DEFAULT_TRADE_INTERVAL_WINDOW = 32
    DEFAULT_RANDOM_START_BUFFER = 100
    DEFAULT_STREAM_BATCH_SIZE = 256
    DEFAULT_PREPROCESS_CHUNK_SIZE = 32

    # Component types (具体的な型を指定して型安全性向上)
    memory_manager: "MemoryManager"
    data_processor: "DataProcessor"
    streaming_handler: "StreamingHandler"
    position_manager: "PositionManager"
    reward_calculator: "RewardCalculator"
    observation_builder: "ObservationBuilder"
    action_validator: "ActionValidator"
    action_executor: ActionExecutor
    data_manager: DataManager
    statistics_calculator: StatisticsCalculator
    adaptive_feature_selector: Optional["AdaptiveFeatureSelector"]

    # Data attributes
    df: pd.DataFrame
    features: List[str]
    reward_settings: RewardSettings
    reward_history: deque[float]
    position_history: deque[float]
    position_abs_history: deque[float]
    pnl_history: deque[float]
    trade_interval_history: deque[int]
    action_history: deque[int]
    _current_episode_actions: List[int]
    _action_counts: List[int]
    _feature_matrix: NDArray[np.float32]
    _price_array: Optional[NDArray[np.float32]]
    _close_array: Optional[NDArray[np.float32]]
    _atr_array: Optional[NDArray[np.float32]]
    _episode_id_array: Optional[NDArray[Any]]
    _timestamp_column: Optional[str]
    _episode_id_column: Optional[str]
    _stream_last_timestamp: Optional[pd.Timestamp]
    _stream_rows_appended: int
    _last_trade_step: Optional[int]
    _consecutive_trade_steps: int
    _max_history_length: int
    _max_action_history: int
    _previous_portfolio_value: Optional[float]
    current_step: int
    n_steps: int
    position: float
    entry_price: float
    total_pnl: float
    realized_pnl: float
    trades_count: int
    portfolio_value_history: deque[float]
    portfolio_value: float
    initial_portfolio_value: float

    # Bind helper functions as methods
    _initialize_components = _initialize_components
    _initialize_data_structures = _initialize_data_structures
    _initialize_data = _initialize_data
    _initialize_features_and_spaces = _initialize_features_and_spaces
    _initialize_remaining_components = _initialize_remaining_components
    _refresh_features = _refresh_features
    _build_fast_access_buffers = _build_fast_access_buffers
    _extract_numeric_column = _extract_numeric_column
    _setup_scaler = _setup_scaler
    _compute_scaler_from_data = _compute_scaler_from_data

    _select_features_by_correlation_in_env = _select_features_by_correlation_in_env

    _fetch_streaming_snapshot = _fetch_streaming_snapshot
    _prepare_stream_batch = _prepare_stream_batch
    _append_streaming_rows = _append_streaming_rows
    _ensure_data_available = _ensure_data_available
    _prime_streaming_data = _prime_streaming_data

    _get_reward_setting_int = _get_reward_setting_int
    _get_reward_setting_float = _get_reward_setting_float
    _get_reward_setting_bool = _get_reward_setting_bool

    _resolve_price = _resolve_price
    _resolve_atr = _resolve_atr

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        config: Optional[Union[EnvironmentConfig, Dict[str, Any]]] = None,
        *,
        random_start: bool = False,
        streaming_pipeline: Optional["StreamingPipeline"] = None,
        stream_batch_size: int = DEFAULT_STREAM_BATCH_SIZE,
        stream_to_bars_converter: Optional["StreamToBarsConverter"] = None,
        max_features: Optional[int] = None,
        optimizer_tracker: Optional["OptimizerFeatureTracker"] = None,
    ) -> None:
        super().__init__()

        if df is not None:
            TypeValidator.validate_type(df, pd.DataFrame, "df")

        if config is not None and not isinstance(config, EnvironmentConfig):
            TypeValidator.validate_type(config, Dict[str, Any], "config")

        TypeValidator.validate_type(random_start, bool, "random_start")
        TypeValidator.validate_type(stream_batch_size, int, "stream_batch_size")
        if stream_batch_size <= 0:
            raise ValidationError(
                "stream_batch_size must be a positive integer",
                details={"stream_batch_size": stream_batch_size},
            )

        if max_features is not None:
            TypeValidator.validate_type(max_features, int, "max_features")
            if max_features <= 0:
                raise ValidationError(
                    "max_features must be positive when provided",
                    details={"max_features": max_features},
                )

        if df is None and streaming_pipeline is None:
            raise ConfigurationError(
                "Either df or streaming_pipeline must be provided",
                details={"df_provided": False, "pipeline_provided": False},
            )

        # Diagnostic: record raw config received to trace conversion issues
        try:
            # Use INFO here so diagnostics are visible during test runs
            logger.info(
                "HeavyTradingEnv.__init__ raw config type=%s, repr_preview=%s",
                type(config),
                (config if isinstance(config, dict) else str(type(config))),
            )
            if isinstance(config, dict):
                logger.info(
                    "HeavyTradingEnv.__init__ raw dict contains use_continuous_actions=%s",
                    config.get("use_continuous_actions", "NOT_PRESENT"),
                )
        except Exception:
            # Ensure diagnostics never break initialization
            logger.exception("Failed to log raw config diagnostic")

        if isinstance(config, EnvironmentConfig):
            self.config = config
        else:
            self.config = EnvironmentConfig.from_dict(config)

        self.random_start = random_start

        # Get continuous-to-discrete action conversion threshold from config
        # Default to environment constant if not specified in config
        from ztb.trading.constants import SAC_CONTINUOUS_THRESHOLD

        self.action_threshold = getattr(
            self.config,
            "continuous_to_discrete_threshold",
            SAC_CONTINUOUS_THRESHOLD,
        )
        # Allow asymmetric configuration but default to symmetric negative threshold if unspecified
        configured_negative_threshold = getattr(
            self.config,
            "continuous_to_discrete_threshold_neg",
            None,
        )
        if configured_negative_threshold is not None:
            self.negative_action_threshold = float(configured_negative_threshold)
        else:
            # Mirror the positive threshold to avoid unintentional BUY/SELL bias
            self.negative_action_threshold = -abs(float(self.action_threshold))

        self._process = psutil.Process()
        self.stream_to_bars_converter = stream_to_bars_converter

        self.reward_settings: Dict[str, Any] = {
            "position_soft_cap": self.config.reward_position_soft_cap,
            "position_penalty_scale": self.config.reward_position_penalty_scale,
            "position_penalty_exponent": self.config.reward_position_penalty_exponent,
            "inventory_window": self.config.reward_inventory_window,
            "inventory_penalty_scale": self.config.reward_inventory_penalty_scale,
            "trade_frequency_penalty": self.config.reward_trade_frequency_penalty,
            "trade_frequency_halflife": self.config.reward_trade_frequency_halflife,
            "trade_cooldown_steps": self.config.reward_trade_cooldown_steps,
            "trade_cooldown_penalty": self.config.reward_trade_cooldown_penalty,
            "max_consecutive_trades": self.config.reward_max_consecutive_trades,
            "consecutive_trade_penalty": self.config.reward_consecutive_trade_penalty,
            "volatility_window": self.config.reward_volatility_window,
            "volatility_penalty_scale": self.config.reward_volatility_penalty_scale,
            "sharpe_bonus_scale": self.config.reward_sharpe_bonus_scale,
            "sortino_bonus_scale": getattr(
                self.config, "reward_sortino_bonus_scale", 0.01
            ),
            "calmar_bonus_scale": getattr(
                self.config, "reward_calmar_bonus_scale", 0.005
            ),
            "reward_clip_value": getattr(self.config, "reward_clip_value", 20.0),
            "profit_bonus_multipliers": self.config.reward_profit_bonus_multipliers,
            "enable_forced_diversity": self.config.enable_forced_diversity,
            "curriculum_stage": getattr(self.config, "curriculum_stage", "simple"),
        }

        if getattr(self.config, "reward_settings", None):
            merged = deep_merge_dict(
                self.reward_settings,
                dataclasses.asdict(self.config.reward_settings),
            )
            self.reward_settings = merged

        # Create RewardSettings object for RewardCalculator
        self.reward_settings_obj: RewardSettings = RewardSettings(
            **self.reward_settings
        )

        self.fee_model = ExchangeFeeModel()
        self.fee_model.set_exchange(self.config.exchange)

        # 🔧 CRITICAL FIX: 訓練時のtransaction_costを尊重
        # 訓練時に明示的にtransaction_costが設定されている場合、それを優先
        # fee_modelのデフォルト値で上書きしない
        if (
            not hasattr(self.config, "transaction_cost")
            or self.config.transaction_cost == 0.0
        ):
            # transaction_costが未設定またはデフォルト値(0.0)の場合のみ、fee_modelから取得
            self.config.transaction_cost = self.fee_model.get_fee_rate("buy")
            logger.info(
                f"Using fee_model transaction_cost: {self.config.transaction_cost}"
            )
        else:
            logger.info(
                f"Using configured transaction_cost: {self.config.transaction_cost} (not overriding with fee_model)"
            )

        self.initial_portfolio_value = float(self.config.initial_portfolio_value)
        self.portfolio_value = self.initial_portfolio_value

        self._timestamp_column = None
        self._episode_id_column = None
        self._stream_last_timestamp = None
        self._stream_rows_appended = 0

        # スキーマ情報の取得（オプショナル）
        self.schema_hash = getattr(self.config, "schema_hash", None)
        self.model_name = getattr(self.config, "model_name", None)
        self.feature_names = getattr(self.config, "feature_names", None)

        if self.feature_names:
            logger.info(f"Using schema-defined features: {len(self.feature_names)}")

        self._initialize_components(streaming_pipeline, stream_batch_size, df)
        self._initialize_data_structures()

        # Initialize optimizer feature tracker BEFORE feature initialization
        self.optimizer_tracker = optimizer_tracker

        # Initialize new component classes
        self.action_executor = ActionExecutor(
            action_threshold=self.action_threshold,
            negative_action_threshold=self.negative_action_threshold,
        )
        self.data_manager = DataManager()
        self.statistics_calculator = StatisticsCalculator()
        self.state_manager = StateManager(self)
        self.validation_manager = ValidationManager(self)

        # Initialize adaptive feature selector (will be set during feature initialization if enabled)
        self.adaptive_feature_selector = None
        self._initialize_data(df)
        self._initialize_features_and_spaces(max_features)
        self._setup_scaler()

        self._initialize_remaining_components()

        # Initialize v444 regime classifier for advanced market regime adaptation
        # Support both dict-style and object-style configs (Pydantic/dataclass)
        self.regime_classifier = None
        advanced_regime_config = getattr(self.config, "advanced_market_regime", None)

        # Determine whether adaptation is enabled and extract classifier config
        enabled = False
        classifier_config = {}
        try:
            if isinstance(advanced_regime_config, dict):
                enabled = bool(advanced_regime_config.get("enabled", False))
                classifier_config = (
                    advanced_regime_config.get("regime_classifier_config", {}) or {}
                )
            else:
                # object-like (Pydantic/dataclass)
                enabled = bool(getattr(advanced_regime_config, "enabled", False))
                classifier_config = (
                    getattr(advanced_regime_config, "regime_classifier_config", {})
                    or {}
                )
        except Exception:
            enabled = False
            classifier_config = {}

        if advanced_regime_config and enabled:
            # Initialize generic market regime classifier with extracted config
            self.regime_classifier = MarketRegimeClassifier(classifier_config)
            logger.info(
                "Market Regime Classifier initialized for market regime adaptation"
            )
            logger.debug(f"Regime classifier config: {type(classifier_config)}")
        else:
            logger.info(
                "Advanced market regime adaptation disabled (legacy configuration)"
            )

        self.portfolio_value_history = deque(maxlen=self.DEFAULT_MAX_HISTORY_LENGTH)
        self._previous_portfolio_value = None
        self._prev_unrealized_pnl = 0.0

        # Initialize observation space based on feature set
        # FeatureSetConfig lives in ztb.features.feature_set_config
        from ztb.features import FeatureRegistry
        from ztb.features.feature_set_config import FeatureSetConfig

        FeatureRegistry.initialize()
        feature_set = getattr(self.config, "feature_set", "high_quality")
        feature_config = FeatureSetConfig()
        feature_config.set_feature_set(feature_set)
        excluded = feature_config.get_excluded_features()

        # If features were already initialized earlier (data-driven discovery
        # including multi-timeframe merges), prefer that set to avoid
        # overwriting a validated feature list with registry defaults.
        if hasattr(self, "features") and self.features:
            logger.info(
                "Features already initialized earlier with %s features; skipping FeatureRegistry override",
                len(self.features),
            )
        else:
            all_features = list(FeatureRegistry._registry.keys())
            self.features = [
                f for f in all_features if not any(ex in f for ex in excluded)
            ]
            obs_dim = len(self.features)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            logger.info(
                f"Initialized observation space with {obs_dim} features from set '{feature_set}'"
            )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        super().reset(seed=seed)

        random_start = (
            options and options.get("random_start", False)
        ) or self.random_start

        # 🔧 DEBUG: random_start が False になる原因を特定
        logger = logging.getLogger(__name__)
        logger.debug(
            f"reset() called: options={options}, self.random_start={self.random_start}, random_start={random_start}"
        )

        if random_start:
            min_start = 0
            # 🔧 CRITICAL FIX: データ長が短い場合でもrandom_startが機能するよう修正
            # 元: max_start = max(0, self.n_steps - self.DEFAULT_RANDOM_START_BUFFER)
            # 問題: データが100行以下だとmax_start=0になりrandom_startが無意味
            # 修正: バッファをデータ長の10%に設定（最小10、最大100）
            buffer = max(
                RANDOM_START_MIN_BUFFER,
                min(
                    RANDOM_START_MAX_BUFFER,
                    int(self.n_steps * RANDOM_START_BUFFER_RATIO),
                ),
            )
            max_start = max(0, self.n_steps - buffer)
            self.current_step = np.random.randint(min_start, max_start + 1)
            logger.debug(
                f"Random start: current_step={self.current_step}, range=[{min_start}, {max_start}], buffer={buffer}, n_steps={self.n_steps}"
            )
        else:
            self.current_step = 0
            logger.debug("Fixed start: current_step=0")

        self.position_manager.reset()
        self.reward_calculator.reset()
        self.statistics_calculator.reset()
        self.state_manager.reset_state()

        # Reset per-step PnL tracking
        self._prev_unrealized_pnl = 0.0

        self._prime_streaming_data()

        if self.current_step % self.DEFAULT_MEMORY_LOG_INTERVAL == 0:
            gc.collect()

        return self._get_observation(), self._get_info()

    def _sync_from_position_manager(self) -> None:
        pos_info = self.position_manager.get_position_info()
        self.position = pos_info["position"]
        self.entry_price = pos_info["entry_price"]
        self.realized_pnl = pos_info["realized_pnl"]
        self.total_pnl = pos_info["total_pnl"]
        self.trades_count = pos_info["trades_count"]

    def get_legal_actions(self) -> Any:
        return self.action_validator.get_legal_actions(
            self.current_step,
            self.position,
            self.total_pnl,
            self.trades_count,
            self._last_trade_step,
            self._consecutive_trade_steps,
            self._close_array,
            self._price_array,
            self.df,
        )

    def action_mask(self) -> Any:
        return self.get_legal_actions().astype(np.bool_)

    def get_action_masks(self) -> Any:
        return self.action_mask()

    def _get_current_market_regime(self) -> str:
        """Get current market regime using v444 classifier if available, fallback to legacy."""
        if self.regime_classifier is not None:
            try:
                # Get recent price data for regime classification
                window_size = 20  # Minimum window for regime detection
                start_idx = max(0, self.current_step - window_size)
                end_idx = self.current_step + 1

                if end_idx > len(self.df):
                    end_idx = len(self.df)

                if end_idx - start_idx >= 10:  # Minimum data points needed
                    price_data = self.df.iloc[start_idx:end_idx].copy()
                    if "close" in price_data.columns:
                        regime_result = self.regime_classifier.detect_regime(price_data)
                        return regime_result.primary_regime.value
            except Exception as e:
                logger.warning(f"Failed to classify regime with v444 classifier: {e}")

        # Fallback to legacy regime detector
        try:
            return self.reward_calculator.market_regime_detector.current_regime
        except Exception:
            return "unknown"

    def step(
        self,
        action: Union[
            int, np.ndarray
        ],  # Can be int (discrete) or np.ndarray (continuous)
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        # Convert action (continuous or discrete) to discrete representation
        (
            discrete_action,
            continuous_action_value,
        ) = self.action_executor.convert_and_validate_action(action)

        # Validate the resulting discrete action
        actual_action = self.validation_manager.validate_action(discrete_action)

        # Debug logging for SAC continuous action output
        if continuous_action_value is not None:
            action_name = (
                "HOLD"
                if actual_action == 0
                else "BUY"
                if actual_action == 1
                else "SELL"
                if actual_action == -1
                else f"UNKNOWN_{actual_action}"
            )
            logger.debug(
                "SAC continuous action: %.6f, discrete action: %d (%s)",
                continuous_action_value,
                actual_action,
                action_name,
            )

        old_position = self.position_manager.position
        min_holding_period = getattr(self.config, "min_holding_period", 0)
        trade_pnl = self.position_manager.execute_action(
            actual_action, self.current_step, min_holding_period
        )

        self._sync_from_position_manager()

        stop_loss_threshold = self.config.stop_loss_threshold
        if self.position != 0 and self.entry_price > 0:
            current_price = self._resolve_price()
            if self.position > 0:
                loss_ratio = (self.entry_price - current_price) / self.entry_price
                if loss_ratio > stop_loss_threshold:
                    forced_close_pnl = self.position_manager.close_position(
                        self.current_step
                    )
                    trade_pnl += forced_close_pnl
                    self._sync_from_position_manager()
            elif self.position < 0:
                loss_ratio = (current_price - self.entry_price) / self.entry_price
                if loss_ratio > stop_loss_threshold:
                    forced_close_pnl = self.position_manager.close_position(
                        self.current_step
                    )
                    trade_pnl += forced_close_pnl
                    self._sync_from_position_manager()

        # Update state using StateManager
        self.state_manager.update_position_state(
            actual_action, self.current_step, trade_pnl
        )

        unrealized_pnl = self.position_manager.calculate_unrealized_pnl()
        portfolio_value = (
            self.initial_portfolio_value + self.realized_pnl + unrealized_pnl
        )
        self.portfolio_value = portfolio_value

        # Calculate per-step PnL (trade_pnl + unrealized delta)
        step_pnl = trade_pnl + (unrealized_pnl - self._prev_unrealized_pnl)
        self._prev_unrealized_pnl = unrealized_pnl

        # Calculate total pnl for info
        pnl = self.total_pnl + unrealized_pnl

        # Update portfolio state
        self.state_manager.update_portfolio_state(trade_pnl, unrealized_pnl)

        current_price = self.data_manager.get_price_at_step(self.current_step)
        atr = self.data_manager.get_atr_at_step(self.current_step)

        # Reward calculation uses discrete action
        reward = self.reward_calculator.calculate_reward(
            action=actual_action,
            current_price=current_price,
            position=self.position,
            portfolio_value=portfolio_value,
            atr=atr,
            transaction_cost=self.config.transaction_cost,
            reward_scaling=self.config.reward_scaling,
            pnl=step_pnl,  # Use per-step PnL instead of cumulative total
            old_position=old_position,
            step=self.current_step,
            observation=self._get_observation(),
            reward_history=list(self.reward_history),
            portfolio_value_history=list(self.portfolio_value_history),
        )

        # Apply market regime adaptation to reward if enabled
        if self.regime_classifier is not None and hasattr(
            self, "regime_adaptation_config"
        ):
            try:
                current_regime = self._get_current_market_regime()
                if current_regime != "unknown":
                    # Get regime-specific multiplier
                    reward_multiplier = self.regime_classifier.get_regime_multiplier(
                        current_regime, "reward"
                    )
                    penalty_multiplier = self.regime_classifier.get_regime_multiplier(
                        current_regime, "penalty"
                    )

                    # Apply multipliers based on reward sign
                    if reward > 0:
                        reward *= reward_multiplier
                    elif reward < 0:
                        reward *= penalty_multiplier

                    # Update regime statistics
                    if current_regime not in self.regime_stats["regime_counts"]:
                        self.regime_stats["regime_counts"][current_regime] = 0
                        self.regime_stats["regime_rewards"][current_regime] = []
                        self.regime_stats["regime_actions"][current_regime] = []

                    self.regime_stats["regime_counts"][current_regime] += 1
                    self.regime_stats["regime_rewards"][current_regime].append(reward)
                    self.regime_stats["regime_actions"][current_regime].append(
                        actual_action
                    )

                    # Track regime transitions
                    if self.regime_stats["current_regime"] != current_regime:
                        self.regime_stats["regime_transitions"].append(
                            {
                                "from": self.regime_stats["current_regime"],
                                "to": current_regime,
                                "step": self.current_step,
                            }
                        )
                        self.regime_stats["current_regime"] = current_regime

            except Exception as e:
                logger.warning(f"Failed to apply regime adaptation: {e}")

        # Validate reward using ValidationManager
        reward = self.validation_manager.validate_reward_calculation(reward)

        # Enhanced debug logging for SAC continuous action and reward analysis
        if continuous_action_value is not None:
            logger.debug(
                "continuous_action=%.6f, discrete_action=%d, reward=%.6f, step_pnl=%.6f, portfolio_return=%.4f%%, position=%.4f",
                continuous_action_value,
                actual_action,
                reward,
                step_pnl,
                ((portfolio_value / self.initial_portfolio_value) - 1) * 100,
                self.position,
            )

        self.current_step += 1
        self.data_manager.ensure_data_available(self.current_step)

        done = self.current_step >= self.n_steps - 1
        if not done:
            done = self.data_manager.is_episode_boundary(
                self.current_step - 1, self.current_step
            )

        next_obs = self._get_observation()

        info = self._get_info()
        position_utilisation = abs(self.position) / max(
            POSITION_EPSILON, self.config.max_position_size
        )
        info.update(
            {
                "pnl": pnl,
                "position": self.position,
                "action": action,
                "step": self.current_step,
                "portfolio_value": portfolio_value,
                "atr": atr,
                "position_utilisation": position_utilisation,
                "action_masks": self.get_legal_actions().astype(bool),
                "trade_executed": pnl != 0
                or actual_action != ACTION_HOLD,  # Add trade execution flag
                "market_regime": self._get_current_market_regime(),
            }
        )

        self.pnl_history.append(pnl)
        self.position_abs_history.append(abs(self.position))
        self.portfolio_value_history.append(portfolio_value)

        self.reward_history.append(reward)
        self.position_history.append(self.position)

        # Update statistics calculator
        self.statistics_calculator.add_reward(reward)
        self.statistics_calculator.add_position(self.position)
        self.statistics_calculator.add_portfolio_value(portfolio_value)
        self.statistics_calculator.add_action(actual_action)

        if self.memory_manager.should_log_memory(self.current_step):
            self.memory_manager.log_memory_usage(f"step_{self.current_step}")

        if self.memory_manager.should_collect_garbage:
            self.memory_manager.collect_garbage(generation=0)

        if self.current_step % self.DEFAULT_GC_STEP_INTERVAL == 0:
            self.memory_manager.collect_garbage()

        return next_obs, reward, done, False, info

    def _get_observation(self) -> Any:
        self.data_manager.ensure_data_available(self.current_step)
        obs = self.observation_builder.get_observation(
            self.current_step,
            self.n_steps,
            self.df,
        )

        # Diagnostic: log environment observation_space and returned observation shape
        try:
            logger.debug(
                "HeavyTradingEnv._get_observation: observation_space.shape=%s, returned_obs_shape=%s, optimizer_tracker_present=%s",
                getattr(self.observation_space, "shape", None),
                getattr(obs, "shape", None),
                self.optimizer_tracker is not None,
            )
        except Exception:
            pass

        return obs

    def _get_info(self) -> Any:
        base_info = self.observation_builder.get_info(
            self.current_step,
            self.n_steps,
            self.position,
            self.total_pnl,
            self.trades_count,
            self.features,
            self.config,
        )

        # Add market regime for regime-aware diagnostics
        current_price = self.data_manager.get_price_at_step(self.current_step)
        market_regime = self.reward_calculator.get_current_regime(
            current_price, self.current_step
        )

        base_info["market_regime"] = market_regime
        return base_info

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            logger.debug("Step: %s/%s", self.current_step, self.n_steps)
            logger.debug("Position: %s", self.position)
            logger.debug("Total PnL: %.4f", self.total_pnl)
            logger.debug("Trades: %s", self.trades_count)
            if self.reward_history:
                logger.debug("Last Reward: %.6f", self.reward_history[-1])

    def close(self) -> None:
        self.reward_history.clear()
        self.position_history.clear()
        self.portfolio_value_history.clear()
        self.action_history.clear()

        self.position_abs_history.clear()
        self.pnl_history.clear()
        self.trade_interval_history.clear()

        if hasattr(self, "df"):
            self.df = pd.DataFrame()

        self._current_episode_actions.clear()
        self._action_counts = ACTION_COUNTS_INITIAL.copy()

        self.statistics_calculator.reset()
        self.memory_manager.collect_garbage_aggressive()

    def get_feature_names(self) -> List[str]:
        return self.features

    def get_statistics(self) -> StatisticsDict:
        stats = self.statistics_calculator.get_statistics()
        # Add trade count from position manager
        stats["total_trades"] = self.trades_count
        return stats

    def get_trades_per_1k(self) -> float:
        return self.statistics_calculator.get_trades_per_1k_steps(self.current_step)

    def get_last_actions(self) -> List[int]:
        return self._current_episode_actions.copy()

    def enable_market_regime_adaptation(
        self,
        regime_classifier: Optional["MarketRegimeClassifier"] = None,
        adaptation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Enable market regime adaptation for the environment

        Args:
            regime_classifier: Market regime classifier instance
            adaptation_config: Configuration for regime adaptation
        """
        if regime_classifier is not None:
            self.regime_classifier = regime_classifier
            logger.info("Market regime classifier set for environment adaptation")

        if adaptation_config is not None:
            self.regime_adaptation_config = adaptation_config
            logger.info("Market regime adaptation config updated")

        # Initialize regime statistics tracking
        self.regime_stats = {
            "regime_counts": {},
            "regime_rewards": {},
            "regime_actions": {},
            "current_regime": None,
            "regime_transitions": [],
        }

        # Alias for backward compatibility
        self.regime_statistics = self.regime_stats

        # Set the flag to indicate regime adaptation is enabled
        self.market_regime_adaptation_enabled = True

        logger.info("Market regime adaptation enabled in environment")


class FlipHeavyTradingEnv(HeavyTradingEnv):
    """Flipped version of HeavyTradingEnv for symmetry testing."""

    def _get_observation(self) -> np.ndarray:
        obs = super()._get_observation()

        if not isinstance(obs, np.ndarray):
            raise TypeError(f"Expected observation to be np.ndarray, got {type(obs)}")

        flip_indices = []
        for i, feature in enumerate(self.features):
            if any(
                keyword in feature.lower()
                for keyword in [
                    "price",
                    "close",
                    "open",
                    "high",
                    "low",
                    "volume",
                    "sma",
                    "ema",
                    "macd",
                    "rsi",
                    "stoch",
                    "williams",
                    "cci",
                    "mfi",
                    "obv",
                    "vwap",
                    "pivot",
                ]
            ):
                flip_indices.append(i)

        flipped_obs = obs.copy()
        for idx in flip_indices:
            if idx < len(flipped_obs):
                flipped_obs[idx] = -flipped_obs[idx]

        return flipped_obs

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        flipped_action = action
        if action == ACTION_BUY:
            flipped_action = ACTION_SELL
        elif action == ACTION_SELL:
            flipped_action = ACTION_BUY

        obs, reward, done, truncated, info = super().step(flipped_action)

        if "position" in info:
            info["position"] = -info["position"]

        if "pnl" in info:
            info["pnl"] = -info["pnl"]

        return obs, reward, done, truncated, info

    def _get_info(self) -> Any:
        info = super()._get_info()
        if "position" in info:
            info["position"] = -info["position"]
        return info

    def enable_market_regime_adaptation(
        self,
        regime_classifier: Optional["MarketRegimeClassifier"] = None,
        adaptation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        FlipHeavyTradingEnv 用: 市場レジーム適応を有効化します。

        HeavyTradingEnv との主な差分:
        - デバッグログ (logger.debug) を追加し、呼び出しや regime_stats の初期化状況を詳細に記録します。
        - regime_stats の初期化方法や例外処理のログ出力が強化されています。
        - その他の挙動は HeavyTradingEnv と同様ですが、flipped 環境で regime 適応の挙動を検証するための追加ログが含まれます。

        Args:
            regime_classifier: Market regime classifier instance
            adaptation_config: Configuration for regime adaptation
        """
        logger.debug("enable_market_regime_adaptation called")
        try:
            if regime_classifier is not None:
                self.regime_classifier = regime_classifier
                logger.info("Market regime classifier set for environment adaptation")

            if adaptation_config is not None:
                self.regime_adaptation_config = adaptation_config
                logger.info("Market regime adaptation config updated")

            # Set adaptation enabled flag
            self.market_regime_adaptation_enabled = True

            # Initialize regime statistics tracking
            self.regime_stats = {
                "regime_counts": {},
                "regime_rewards": {},
                "regime_actions": {},
                "current_regime": None,
                "regime_transitions": [],
            }
            logger.debug(f"Regime_stats set: {hasattr(self, 'regime_stats')}")

            logger.info("Market regime adaptation enabled in environment")
        except Exception as e:
            logger.debug(f"Exception in enable_market_regime_adaptation: {e}")
            raise
