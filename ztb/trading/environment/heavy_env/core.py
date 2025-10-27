"""Core trading environment implementations."""

from __future__ import annotations

import gc
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import psutil
from numpy.typing import NDArray

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import (
    ACTION_COUNTS_INITIAL,
    NUM_DISCRETE_ACTIONS,
    POSITION_EPSILON,
    RANDOM_START_BUFFER_RATIO,
    RANDOM_START_MAX_BUFFER,
    RANDOM_START_MIN_BUFFER,
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
from ztb.trading.environment.types import EPSILON, StatisticsDict
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.types.protocols import TradingEnvironment
from ztb.utils.errors import ConfigurationError, ValidationError
from ztb.utils.fee_model import ExchangeFeeModel
from ztb.utils.logging_utils import get_logger
from ztb.utils.type_validation import TypeValidator

# Import components for runtime use
from ztb.trading.environment.components.action_executor import ActionExecutor
from ztb.trading.environment.components.data_manager import DataManager
from ztb.trading.environment.components.statistics_calculator import StatisticsCalculator
from ztb.trading.environment.heavy_env.components.state_manager import StateManager
from ztb.trading.environment.heavy_env.components.validation_manager import ValidationManager

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

        self.reward_settings: RewardSettings = {
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
            "reward_clip_value": self.config.reward_clip_value,
            "profit_bonus_multipliers": self.config.reward_profit_bonus_multipliers,
            "enable_forced_diversity": self.config.enable_forced_diversity,
        }

        if getattr(self.config, "reward_settings", None):
            merged: RewardSettings = {
                **self.reward_settings,
                **self.config.reward_settings,
            }  # type: ignore[typeddict-item]
            self.reward_settings = merged

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

        # use_standardized_observationsがTrueで、スケーラーがまだ設定されていない場合は計算
        if getattr(self.config, "use_standardized_observations", True):
            scaler_mean = getattr(self, "scaler_mean", None)
            scaler_std = getattr(self, "scaler_std", None)
            if scaler_mean is None or scaler_std is None:
                logger.info(
                    "Computing scaler from data (use_standardized_observations=True)"
                )
                self._compute_scaler_from_data()

        self._initialize_remaining_components()

        self.portfolio_value_history = deque(maxlen=self.DEFAULT_MAX_HISTORY_LENGTH)
        self._previous_portfolio_value = None

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
        import logging

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
            buffer = max(RANDOM_START_MIN_BUFFER, min(RANDOM_START_MAX_BUFFER, int(self.n_steps * RANDOM_START_BUFFER_RATIO)))
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

    def step(
        self,
        action: Union[
            int, np.ndarray
        ],  # Can be int (discrete) or np.ndarray (continuous)
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        # Validate and convert action using ValidationManager
        actual_action = self.validation_manager.validate_action(action)

        # Convert continuous action to discrete if necessary
        actual_action, continuous_action_value = self.action_executor.convert_and_validate_action(actual_action)

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
        self.state_manager.update_position_state(actual_action, self.current_step, trade_pnl)

        unrealized_pnl = self.position_manager.calculate_unrealized_pnl()
        portfolio_value = (
            self.initial_portfolio_value + self.realized_pnl + unrealized_pnl
        )
        self.portfolio_value = portfolio_value

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
            pnl=self.total_pnl,
            old_position=old_position,
            step=self.current_step,
            observation=self._get_observation(),
            reward_history=list(self.reward_history),
            portfolio_value_history=list(self.portfolio_value_history),
        )

        # Validate reward using ValidationManager
        reward = self.validation_manager.validate_reward_calculation(reward)

        self.current_step += 1
        self.data_manager.ensure_data_available(self.current_step)

        done = self.current_step >= self.n_steps - 1
        if not done:
            done = self.data_manager.is_episode_boundary(self.current_step - 1, self.current_step)

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
        return self.observation_builder.get_observation(
            self.current_step,
            self.n_steps,
            self.df,
        )

    def _get_info(self) -> Any:
        return self.observation_builder.get_info(
            self.current_step,
            self.n_steps,
            self.position,
            self.total_pnl,
            self.trades_count,
            self.features,
            self.config,
        )

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


class FlipHeavyTradingEnv(HeavyTradingEnv):
    """Flipped version of HeavyTradingEnv for symmetry testing."""

    def _get_observation(self) -> Any:
        obs = super()._get_observation()

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
