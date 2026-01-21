"""Initialization helpers for HeavyTradingEnv."""

from __future__ import annotations

import gc
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from gymnasium import spaces
from numpy.typing import NDArray
from pandas.api import types as ptypes

from ztb.features.generators.adaptive.selection import AdaptiveFeatureSelector
from ztb.trading.environment.components import (
    ActionValidator,
    DataProcessor,
    MemoryManager,
    ObservationBuilder,
    PositionManager,
    RewardCalculator,
    StreamingHandler,
)
from ztb.trading.environment.components.action_executor import ActionExecutor
from ztb.trading.environment.components.statistics_calculator import (
    StatisticsCalculator,
)
from ztb.trading.environment.components.threshold_manager import ThresholdManager
from ztb.trading.environment.heavy_env.components.state_manager import StateManager
from ztb.trading.environment.heavy_env.components.validation_manager import (
    ValidationManager,
)
from ztb.trading.execution.realistic import RealisticExecutionModel
from ztb.utils.errors import ValidationError
from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import get_project_root, safe_path_join
from ztb.utils.type_validation import TypeValidator

if TYPE_CHECKING:
    from ztb.data.streaming_pipeline import StreamingPipeline


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public helpers bound to HeavyTradingEnv
# ---------------------------------------------------------------------------


def _initialize_components(
    self: Any,
    streaming_pipeline: Optional["StreamingPipeline"],
    stream_batch_size: int,
    df: Optional[pd.DataFrame],
) -> None:
    """Initialize core helper components used by the environment."""
    raw_memory_log_path = getattr(self.config, "memory_log_path", None)
    memory_log_path: Optional[str] = None
    if isinstance(raw_memory_log_path, str):
        path_candidate = Path(raw_memory_log_path)
        if not path_candidate.is_absolute():
            path_candidate = safe_path_join(
                str(get_project_root()), raw_memory_log_path
            )
        memory_log_path = str(path_candidate.resolve())
    elif raw_memory_log_path is not None:
        TypeValidator.validate_type(raw_memory_log_path, str, "memory_log_path")

    self.memory_manager = MemoryManager(
        memory_log_path=memory_log_path,
        memory_logging_enabled=getattr(self.config, "memory_logging_enabled", False),
        memory_log_interval_steps=getattr(
            self.config,
            "memory_log_interval_steps",
            self.DEFAULT_MEMORY_LOG_INTERVAL,
        ),
        gc_step_interval=getattr(self.config, "gc_collect_interval_steps", 0),
    )

    self.data_processor = DataProcessor(
        preprocess_chunk_size=getattr(
            self.config,
            "preprocess_chunk_size",
            self.DEFAULT_PREPROCESS_CHUNK_SIZE,
        ),
        memory_logging_enabled=self.memory_manager.memory_logging_enabled,
        gc_step_interval=self.memory_manager.gc_step_interval,
    )

    timestamp_column = (
        "timestamp" if df is not None and "timestamp" in df.columns else None
    )
    episode_id_column = (
        "episode_id" if df is not None and "episode_id" in df.columns else None
    )

    self.streaming_handler = StreamingHandler(
        streaming_pipeline=streaming_pipeline,
        stream_batch_size=max(1, int(stream_batch_size)),
        timestamp_column=timestamp_column,
        episode_id_column=episode_id_column,
    )


def _initialize_data_manager(
    self: Any,
    streaming_pipeline: Optional["StreamingPipeline"],
    stream_batch_size: int,
    df: Optional[pd.DataFrame],
) -> None:
    """Initialize the DataManager component."""
    from ztb.trading.environment.components.data_manager import DataManager

    if df is None:
        raise ValueError("DataFrame cannot be None when initializing DataManager")
    self.data_manager = DataManager()
    self.data_manager.initialize_data(
        df=df,
        features=self.features,
        timestamp_column=self._timestamp_column,
        episode_id_column=self._episode_id_column,
    )


def _initialize_data_structures(self: Any) -> None:
    """Prepare deque-based histories and fast-access buffers."""
    inventory_window = max(
        self.DEFAULT_MIN_WINDOW_SIZE,
        self._get_reward_setting_int("inventory_window", self.DEFAULT_INVENTORY_WINDOW),
    )
    volatility_window = max(
        self.DEFAULT_MIN_WINDOW_SIZE,
        self._get_reward_setting_int(
            "volatility_window", self.DEFAULT_VOLATILITY_WINDOW
        ),
    )
    self.position_abs_history = deque(maxlen=inventory_window)
    self.pnl_history = deque(maxlen=volatility_window)
    self.trade_interval_history = deque(maxlen=self.DEFAULT_TRADE_INTERVAL_WINDOW)
    self._last_trade_step = None
    self._consecutive_trade_steps = 0

    # Fast-access buffers (populated after preprocessing)
    self._feature_matrix = np.empty((0, 0), dtype=np.float32)
    self._price_array = None
    self._close_array = None
    self._atr_array = None
    self._episode_id_array = None
    self._nonfinite_rows = set()
    self._nonfinite_warned_rows = set()

    # Initialize action counts for state management
    from ztb.trading.environment.constants import NUM_DISCRETE_ACTIONS

    self.ACTION_COUNTS_INITIAL = [0] * NUM_DISCRETE_ACTIONS


def _initialize_data(self: Any, df: Optional[pd.DataFrame]) -> None:
    """Load baseline market data either from dataframe, CSV file, or streaming snapshot."""
    if df is not None:
        base_df = df
    else:
        # Check if csv_path is specified in config
        csv_path = getattr(self.config, "csv_path", None)
        if csv_path is not None:
            logger.info(f"Loading data from CSV: {csv_path}")
            try:
                base_df = pd.read_csv(csv_path)
                logger.info(f"Loaded {len(base_df)} rows from CSV")
            except Exception as e:
                raise ValidationError(
                    f"Failed to load data from CSV path: {csv_path}",
                    details={"csv_path": csv_path, "error": str(e)},
                )
        else:
            base_df = self._fetch_streaming_snapshot(
                required_rows=self.streaming_handler.stream_batch_size
            )
            if base_df.empty:
                raise ValidationError(
                    "Streaming pipeline did not provide initial data",
                    details={
                        "pipeline": str(
                            type(self.streaming_handler.streaming_pipeline)
                        ),
                        "df_empty": True,
                    },
                )

    self.df = self.data_processor.preprocess_data(base_df)
    if df is None:
        del base_df
    gc.collect()
    self.memory_manager.log_memory_usage("post_init", df_override=self.df)

    if not self.df.index.is_monotonic_increasing:
        self.df.sort_index(inplace=True)

    self.n_steps = len(self.df)
    self._base_columns = list(self.df.columns)


def _initialize_features_and_spaces(self: Any, max_features: Optional[int]) -> None:
    """Derive features, apply limits, and build observation/action spaces."""
    # Check if features are specified in config (schema-based approach)
    config_features = getattr(self.config, "feature_names", None)
    correlation_reduction = getattr(self.config, "correlation_reduction", True)
    # Debug config-derived diagnostics
    try:
        logger.debug(
            "_initialize_features_and_spaces: config.feature_names_present=%s",
            config_features is not None,
        )
        if config_features is not None:
            logger.debug("config.feature_names (preview)=%s", config_features[:24])
    except Exception:
        pass
    logger.info(
        "Config features: %s, correlation_reduction: %s",
        config_features is not None,
        correlation_reduction,
    )
    if config_features is not None:
        logger.info(f"Using schema-defined features: {len(config_features)}")
        self.features = config_features
        # Validate that all required features exist in dataframe
        missing_features = set(config_features) - set(self.df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features in dataframe: {missing_features}"
            )
    else:
        # Original feature discovery logic
        exclude_by_default = {"ts", "timestamp", "exchange", "pair", "episode_id"}
        all_features = [col for col in self.df.columns if col not in exclude_by_default]
        if not all_features:
            all_features = list(self.df.columns)

        feature_set = getattr(self.config, "feature_set", "full")
        enable_feature_filtering = getattr(
            self.config, "enable_feature_filtering", False
        )
        feature_filter_mode = getattr(self.config, "feature_filter_mode", "whitelist")

        # Use FeatureSetConfig for feature filtering
        from ztb.features.feature_set_config import get_feature_config

        feature_config = get_feature_config()
        feature_config.set_feature_set(feature_set)
        feature_flags = feature_config.get_feature_flags()

        logger.info(f"Feature set: {feature_set}, flags: {feature_flags}")

        if enable_feature_filtering and feature_filter_mode == "whitelist":
            curated_features_spec = getattr(self.config, "curated_features_list", None)
            if curated_features_spec:
                all_features = _apply_curated_feature_filter(
                    curated_features_spec, all_features
                )

        max_features_limit = _resolve_max_features_limit(self, max_features)

        # Apply FeatureSetConfig filtering
        excluded_features = feature_config.get_excluded_features()
        if excluded_features:
            all_features = [f for f in all_features if f not in excluded_features]
            logger.info(
                f"Excluded {len(excluded_features)} features: {excluded_features}"
            )

        # Check if multi-timeframe features should be included
        # Force enable if not explicitly disabled, as requested by user
        include_mtf = feature_flags.get("include_multi_timeframe_features", False)
        if not include_mtf:
            logger.info("Forcing enable of multi-timeframe features (v455 requirement)")
            include_mtf = True

        if include_mtf:
            # Add multi-timeframe features if available
            import gc

            from ztb.features.multi_timeframe import MultiTimeframeFeatureSystem

            try:
                mtf_system = MultiTimeframeFeatureSystem()
                # The multi-timeframe feature system expects OHLCV + timestamp and is not
                # robust to pre-featured datasets (duplicate column names can occur when
                # generating features on top of existing engineered columns). Feed only
                # the base OHLCV frame to keep generation stable and deterministic.
                base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                base_df = self.df[[c for c in base_cols if c in self.df.columns]].copy()
                mtf_data = mtf_system.process_multi_timeframe_data(base_df)
                if not mtf_data.empty:
                    # Merge the derived multi-timeframe features into the base dataframe
                    # Align by index and avoid duplicate columns. This ensures that the
                    # engineered features exist in `self.df` before we finalize
                    # the selected feature list.
                    mtf_to_add = [
                        c for c in mtf_data.columns if c not in self.df.columns
                    ]
                    if mtf_to_add:
                        # Only append new columns to avoid overwriting base data
                        self.df = pd.concat(
                            [
                                self.df.reset_index(drop=True),
                                mtf_data[mtf_to_add].reset_index(drop=True),
                            ],
                            axis=1,
                        )
                        # Recompute n_steps and base columns after merging
                        self.n_steps = len(self.df)
                        self._base_columns = list(self.df.columns)

                    # Only extend the feature list with columns we actually merged into
                    # `self.df` (avoid re-introducing excluded base columns like OHLCV).
                    mtf_features = [
                        col for col in mtf_to_add if col not in all_features
                    ]
                    if mtf_features:
                        all_features.extend(mtf_features)
                        logger.info(
                            f"Added {len(mtf_features)} multi-timeframe features and merged into dataframe"
                        )

                    # Clear mtf_data to free memory
                    del mtf_data
                    del mtf_system
                    gc.collect()
            except Exception as e:
                logger.warning(f"Failed to add multi-timeframe features: {e}")

        self.features = all_features

        correlation_reduction = getattr(self.config, "correlation_reduction", True)
        target_feature_count = getattr(self.config, "target_feature_count", None)
        if target_feature_count is None:
            target_feature_count = getattr(self.config, "expected_features", None)
        threshold_trigger = target_feature_count or 10
        logger.info(f"Correlation reduction enabled: {correlation_reduction}")
        if correlation_reduction and len(self.features) > threshold_trigger:
            logger.info(
                "Applying correlation reduction...",
                extra={
                    "current_count": len(self.features),
                    "target": target_feature_count,
                },
            )
            correlation_threshold = getattr(self.config, "correlation_threshold", 0.95)
            (
                optimized_features,
                reduction_stats,
            ) = self._select_features_by_correlation_in_env(
                self.features,
                correlation_threshold,
                target_feature_count=target_feature_count,
            )
            if len(optimized_features) >= max(1, (target_feature_count or 1)):
                removed_count = len(self.features) - len(optimized_features)
                self.features = optimized_features
                if removed_count > 0:
                    logger.info(
                        "Applied correlation-based feature reduction",
                        extra={
                            "removed_count": removed_count,
                            "remaining": len(self.features),
                            "target": target_feature_count,
                            "dropped_non_numeric": reduction_stats.get("non_numeric"),
                            "dropped_constant": reduction_stats.get("constant"),
                            "dropped_correlated": reduction_stats.get("correlated"),
                        },
                    )
                else:
                    logger.info(
                        "Correlation reduction made no changes",
                        extra={
                            "remaining": len(self.features),
                            "target": target_feature_count,
                        },
                    )
            else:
                logger.warning(
                    "Correlation reduction would leave too few features",
                    extra={
                        "optimized_count": len(optimized_features),
                        "original_count": len(self.features),
                    },
                )

        if max_features_limit and len(self.features) > max_features_limit:
            _enforce_feature_limit(self, max_features_limit)

    # Apply max_features_limit even for schema-defined features
    max_features_limit = _resolve_max_features_limit(self, max_features)
    if max_features_limit and len(self.features) > max_features_limit:
        _enforce_feature_limit(self, max_features_limit)

    # Apply adaptive feature selection if enabled
    adaptive_selection_config = getattr(self.config, "adaptive_feature_selection", None)
    if adaptive_selection_config and getattr(
        adaptive_selection_config, "enabled", False
    ):
        logger.info("Applying adaptive feature selection...")
        try:
            selector = AdaptiveFeatureSelector(adaptive_selection_config)
            # Store the selector instance for potential reuse
            self.adaptive_feature_selector = selector
            selected_features, selection_stats = selector.select_features(
                self.df, self.features
            )
            if len(selected_features) > 0:
                removed_count = len(self.features) - len(selected_features)
                self.features = selected_features
                logger.info(
                    "Applied adaptive feature selection",
                    extra={
                        "removed_count": removed_count,
                        "remaining": len(self.features),
                        "regime": selection_stats.get("regime"),
                        "attention_weighted": selection_stats.get("attention_weighted"),
                        "selection_threshold": selection_stats.get(
                            "selection_threshold"
                        ),
                    },
                )
            else:
                logger.warning(
                    "Adaptive feature selection returned no features, keeping original set",
                    extra={"original_count": len(self.features)},
                )
        except Exception as exc:
            logger.warning(
                "Failed to apply adaptive feature selection: %s",
                exc,
                exc_info=True,
            )

    self.data_processor.apply_feature_storage_dtype(
        self.df, self.features, self.config.__dict__
    )

    _build_fast_access_buffers(self)

    # Calculate observation space dimensions
    obs_dim = len(self.features)

    # Ensure feature_names is synced with features for external components (e.g. SignalIntegrator)
    self.feature_names = self.features
    # Also update config.feature_names so SignalIntegrator can access it
    self.config.feature_names = self.features

    # Add optimizer features dimension if tracker is available
    if hasattr(self, "optimizer_tracker") and self.optimizer_tracker is not None:
        from ztb.features.processors.optimization.features import (
            OptimizerFeatureTracker,
        )

        if isinstance(self.optimizer_tracker, OptimizerFeatureTracker):
            obs_dim += len(self.optimizer_tracker.get_feature_names())
            logger.info(
                f"Added optimizer features to observation space: +{len(self.optimizer_tracker.get_feature_names())} dimensions"
            )

    # Observation space (always Box for features)
    self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dim,),
        dtype=np.float32,
    )

    # Action space: Support both discrete (PPO) and continuous (SAC)
    from ztb.trading.environment.constants import (
        CONTINUOUS_ACTION_DIM,
        CONTINUOUS_ACTION_MAX,
        CONTINUOUS_ACTION_MIN,
        NUM_DISCRETE_ACTIONS,
    )

    # Determine whether to use continuous actions. Support both explicit boolean
    # flag (`use_continuous_actions`) and legacy/string flag (`action_space_type`).
    # Treat any value that looks like 'continuous' as enabling continuous actions.
    # Handle both dict and object-style configs
    if isinstance(self.config, dict):
        explicit_continuous = self.config.get("use_continuous_actions", False)
        action_space_type = self.config.get("action_space_type", "")
    else:
        explicit_continuous = getattr(self.config, "use_continuous_actions", False)
        action_space_type = getattr(self.config, "action_space_type", "")

    try:
        action_space_type_str = (
            str(action_space_type).strip().lower()
            if action_space_type is not None
            else ""
        )
    except Exception:
        action_space_type_str = ""

    use_continuous_actions = bool(explicit_continuous) or (
        isinstance(action_space_type_str, str)
        and action_space_type_str.startswith("cont")
    )

    if use_continuous_actions:
        # Continuous action space for SAC and other continuous algorithms
        # Action value in [-1, 1]: negative=SELL, 0=HOLD, positive=BUY
        self.action_space = spaces.Box(
            low=CONTINUOUS_ACTION_MIN,
            high=CONTINUOUS_ACTION_MAX,
            shape=(CONTINUOUS_ACTION_DIM,),
            dtype=np.float32,
        )
        logger.info("Using continuous action space (SAC-compatible)")
    else:
        # Discrete action space for PPO with action masking
        self.action_space = spaces.Discrete(NUM_DISCRETE_ACTIONS)
        logger.info("Using discrete action space (PPO-compatible)")


def _setup_scaler(self: Any) -> None:
    """Setup feature scaler from config or schema data."""
    # Check if scaler data is provided in config
    scaler_mean = getattr(self.config, "scaler_mean", None)
    scaler_std = getattr(self.config, "scaler_std", None)

    if scaler_mean is not None and scaler_std is not None:
        # Store scaler data for later use
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        logger.info("Using schema-provided scaler data")
    else:
        # No scaler data provided
        self.scaler_mean = None
        self.scaler_std = None
        logger.info("No scaler data provided")


def _compute_scaler_from_data(self: Any, train_end_index: Optional[int] = None) -> None:
    """
    データからスケーラーを計算（標準化用の平均・標準偏差）。
    データリークを防ぐため、訓練データのみを使用する。
    """
    # Ensure fast access buffers are built
    self._build_fast_access_buffers()

    # Update data manager with built buffers
    self.data_manager._feature_matrix = self._feature_matrix
    self.data_manager._price_array = self._price_array
    self.data_manager._close_array = self._close_array
    self.data_manager._atr_array = self._atr_array
    self.data_manager._episode_id_array = self._episode_id_array
    self.data_manager._nonfinite_rows = self._nonfinite_rows
    self.data_manager._nonfinite_warned_rows = self._nonfinite_warned_rows

    if not hasattr(self, "_feature_matrix") or self._feature_matrix.size == 0:
        logger.warning("Feature matrix is empty. Cannot compute scaler.")
        self.scaler_mean = None
        self.scaler_std = None
        return

    # データリークを防ぐため、訓練データのみでスケーラーを計算
    if train_end_index is not None and train_end_index < self._feature_matrix.shape[0]:
        scaler_features = self._feature_matrix[:train_end_index]
        logger.info(f"Computing scaler using data up to index {train_end_index}")
    else:
        scaler_features = self._feature_matrix
        if train_end_index is None:
            logger.warning(
                "train_end_index not provided. Computing scaler on the entire dataset. "
                "This may cause data leakage if the dataset includes validation/test data."
            )
        else:
            logger.info(
                "train_end_index is beyond data length. Using entire dataset for scaler."
            )

    # 特徴量行列の訓練データ部分から平均・標準偏差を計算
    # axis=0: 各特徴量ごとに計算（列方向）
    self.scaler_mean = np.mean(scaler_features, axis=0).astype(np.float32)
    self.scaler_std = np.std(scaler_features, axis=0).astype(np.float32)

    # 標準偏差が極端に小さい特徴量を検出（ログ記録）
    near_zero_std = np.sum(self.scaler_std < 1e-8)
    if near_zero_std > 0:
        logger.info(
            f"Found {near_zero_std} features with near-zero std "
            f"(will not be scaled to avoid division by zero)"
        )

    logger.info(
        f"Computed scaler from data: "
        f"mean range [{self.scaler_mean.min():.2f}, {self.scaler_mean.max():.2f}], "
        f"std range [{self.scaler_std.min():.2f}, {self.scaler_std.max():.2f}]"
    )


def _initialize_remaining_components(self: Any) -> None:
    """Finalize runtime component setup once data is ready."""

    # Initialize Online Scaler (v455)
    # We keep this as it addresses the data leakage issue
    from ztb.processing.online_scaler import OnlineScaler

    obs_dim = len(self.features)
    if hasattr(self, "optimizer_tracker") and self.optimizer_tracker is not None:
        from ztb.features.processors.optimization.features import (
            OptimizerFeatureTracker,
        )

        if isinstance(self.optimizer_tracker, OptimizerFeatureTracker):
            obs_dim += len(self.optimizer_tracker.get_feature_names())

    self.online_scaler = OnlineScaler(shape=(obs_dim,), clip=5.0)
    logger.info(f"Initialized OnlineScaler with dimension {obs_dim}")

    # Initialize Execution Model if configured
    execution_model = None
    execution_config = getattr(self.config, "execution_model", None)

    if execution_config:
        try:
            # If it's a dict, use it as kwargs
            if isinstance(execution_config, dict):
                execution_model = RealisticExecutionModel(**execution_config)
            # If it's an object (pydantic), convert to dict
            elif hasattr(execution_config, "__dict__"):
                # Filter out internal attributes
                kwargs = {
                    k: v
                    for k, v in execution_config.__dict__.items()
                    if not k.startswith("_")
                }
                execution_model = RealisticExecutionModel(**kwargs)
            # If it's just a boolean True, use defaults
            elif execution_config is True:
                execution_model = RealisticExecutionModel()

            if execution_model:
                logger.info(
                    f"Initialized RealisticExecutionModel: {execution_model.__dict__}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize ExecutionModel: {e}")
            execution_model = None

    self.position_manager = PositionManager(
        config=self.config,
        get_price_callback=self._resolve_price,
        execution_model=execution_model,
    )

    # Check for v457 "PnL Centered" bypass
    reward_type = None
    if self.reward_settings_obj and hasattr(
        self.reward_settings_obj, "custom_reward_params"
    ):
        reward_type = self.reward_settings_obj.custom_reward_params.get("type")

    if reward_type == "pnl_centered":
        from ztb.trading.environment.components.calculators.v457_reward_calculator import (
            V457RewardCalculator,
        )

        logger.info(
            "Using V457 PnL-Centered Reward Calculator PnL (Bypassing v456 logic)"
        )
        self.reward_calculator = V457RewardCalculator(
            config=self.config,
            reward_settings=self.reward_settings_obj,
            initial_portfolio_value=self.initial_portfolio_value,
        )
    else:
        self.reward_calculator = RewardCalculator(
            config=self.config,
            reward_settings=self.reward_settings_obj,
            initial_portfolio_value=self.initial_portfolio_value,
        )

    # Diagnostic: log features and feature_matrix shape to debug unexpected obs dims
    try:
        fm_shape = (
            getattr(self, "_feature_matrix", None).shape
            if getattr(self, "_feature_matrix", None) is not None
            else None
        )
        df_cols = (
            list(self.df.columns)[:50] if getattr(self, "df", None) is not None else []
        )
        logger.debug(
            "Initializing ObservationBuilder: features_len=%d, feature_matrix_shape=%s, df_columns_preview=%s",
            len(self.features),
            fm_shape,
            df_cols,
        )
    except Exception:
        pass

    # Ensure the feature matrix matches the final feature list. It's possible that
    # downstream steps (adaptive selection, filtering, or schema injection)
    # modified `self.features` after the initial buffer build. Rebuild here to
    # guarantee consistency between `self.features` and `self._feature_matrix`.
    try:
        if not hasattr(self, "_feature_matrix") or self._feature_matrix is None:
            self._build_fast_access_buffers()
        elif len(self.features) != getattr(self, "_feature_matrix").shape[1]:
            logger.warning(
                "Feature list length (%s) does not match feature_matrix columns (%s). Rebuilding feature matrix.",
                len(self.features),
                getattr(self, "_feature_matrix").shape[1],
            )
            # Rebuild using the current self.features
            self._build_fast_access_buffers()
    except Exception:
        logger.exception(
            "Failed to ensure feature_matrix consistency before ObservationBuilder creation"
        )

    # v455: Disable global scaler in ObservationBuilder in favor of OnlineScaler
    # We pass None for scaler_mean/std so ObservationBuilder returns raw features.
    # OnlineScaler will handle scaling of the full observation vector.
    self.observation_builder = ObservationBuilder(
        features=self.features,
        feature_matrix=self._feature_matrix,
        nonfinite_rows=self._nonfinite_rows,
        nonfinite_warned_rows=self._nonfinite_warned_rows,
        scaler_mean=None,
        scaler_std=None,
        optimizer_tracker=self.optimizer_tracker,
    )

    # Final consistency check: ensure observation builder feature matrix matches
    # the finalized feature list. If mismatch is detected, attempt to rebuild
    # the feature matrix and update the observation builder so runtime
    # observations remain consistent.
    try:
        fm = getattr(self, "_feature_matrix", None)
        if fm is None or fm.size == 0 or len(self.features) != fm.shape[1]:
            logger.warning(
                "Post-init feature mismatch detected (features=%s, feature_matrix_cols=%s). Rebuilding and updating ObservationBuilder.",
                len(self.features),
                fm.shape[1] if fm is not None else None,
            )
            self._build_fast_access_buffers()
            # Update observation builder in-place
            if (
                hasattr(self, "observation_builder")
                and self.observation_builder is not None
            ):
                self.observation_builder.update_features(self.features)
                self.observation_builder.update_feature_matrix(
                    self._feature_matrix, self._nonfinite_rows
                )
    except Exception:
        logger.exception("Failed final feature/feature_matrix consistency check")

    self.action_validator = ActionValidator(
        config=self.config,
        initial_portfolio_value=self.initial_portfolio_value,
    )

    self.statistics_calculator = StatisticsCalculator()

    self.state_manager = StateManager(self)

    self.threshold_manager = ThresholdManager(
        config=self.config,
    )

    self.action_executor = ActionExecutor(
        action_threshold=self.action_threshold,
        negative_action_threshold=self.negative_action_threshold,
    )

    self.validation_manager = ValidationManager(self)

    self._timestamp_column = "timestamp" if "timestamp" in self.df.columns else None
    self._episode_id_column = "episode_id" if "episode_id" in self.df.columns else None
    self._stream_last_timestamp = None
    self._stream_rows_appended = len(self.df) if not self._timestamp_column else 0

    self.current_step = 0
    self.position = 0.0
    self.entry_price = 0.0
    self.total_pnl = 0.0
    self.realized_pnl = 0.0
    self.trades_count = 0

    self._max_history_length = getattr(
        self.config,
        "max_history_length",
        self.DEFAULT_MAX_HISTORY_LENGTH,
    )
    self.reward_history = deque(maxlen=self._max_history_length)
    self.position_history = deque(maxlen=self._max_history_length)
    self._action_counts = [0, 0, 0]  # [BUY, SELL, HOLD]
    self._current_episode_actions = []

    action_history_limit = getattr(self.config, "max_action_history", None)
    try:
        self._max_action_history = max(
            10, int(action_history_limit or self.DEFAULT_MAX_ACTION_HISTORY)
        )
    except (TypeError, ValueError):
        self._max_action_history = self.DEFAULT_MAX_ACTION_HISTORY

    self.action_history = deque(maxlen=self._max_action_history)


# ---------------------------------------------------------------------------
# Internal helpers used by the public initialization functions
# ---------------------------------------------------------------------------


def _apply_curated_feature_filter(
    curated_features_spec: str, all_features: List[str]
) -> List[str]:
    """Apply curated feature whitelist if available."""
    if "::" not in curated_features_spec:
        logger.warning(
            "Invalid curated_features_list format",
            extra={"curated_spec": curated_features_spec},
        )
        return all_features

    module_path_str, var_name = curated_features_spec.split("::", maxsplit=1)
    module_path = Path(module_path_str)
    if not module_path.is_absolute():
        module_path = safe_path_join(str(get_project_root()), module_path_str)
    module_path = module_path.resolve()

    if not module_path.exists():
        logger.warning(
            "Curated features module path does not exist",
            extra={"module_path": str(module_path)},
        )
        return all_features

    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        module_path.stem,
        module_path.as_posix(),
    )
    if not spec or not spec.loader:
        logger.warning(
            "Failed to create module spec for curated features",
            extra={"module_path": str(module_path)},
        )
        return all_features

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path.stem] = module
    spec.loader.exec_module(module)
    curated_list = getattr(module, var_name, None)

    if not curated_list:
        logger.warning(
            "Curated features variable not found in module",
            extra={"module": module_path.stem, "variable": var_name},
        )
        return all_features

    filtered = [feat for feat in curated_list if feat in all_features]
    removed = len(all_features) - len(filtered)
    logger.info(
        "Applied curated features filter",
        extra={
            "original_count": len(all_features),
            "kept": len(filtered),
            "removed": removed,
            "module": module_path.stem,
            "variable": var_name,
        },
    )
    return filtered


def _resolve_max_features_limit(
    self: Any, max_features: Optional[int]
) -> Optional[int]:
    """Resolve the effective max_features limit from various config locations."""
    max_features_limit = max_features
    if max_features_limit is None and hasattr(self.config, "get"):
        config_dict = self.config
        max_features_limit = (
            config_dict.get("max_features")
            or (config_dict.get("memory_optimization", {}) or {}).get("max_features")
            or (config_dict.get("ppo", {}) or {}).get("max_features")
        )

    if max_features_limit is None:
        max_features_limit = getattr(self.config, "max_features", None)
    return max_features_limit if max_features_limit else None


def _enforce_feature_limit(self: Any, max_features_limit: int) -> None:
    """Keep only the highest variance features within the specified limit."""
    logger.warning(
        "MEMORY OPTIMIZATION: Reducing features from %s to %s",
        len(self.features),
        max_features_limit,
        extra={"original_count": len(self.features), "limit": max_features_limit},
    )

    feature_variances: List[Tuple[str, float]] = []
    for feature_name in self.features:
        if feature_name not in self.df.columns:
            continue
        try:
            variance_value = self.df[feature_name].var()
            if pd.notna(variance_value) and isinstance(
                variance_value, (int, float, np.number)
            ):
                variance = float(variance_value)
                if not np.isnan(variance) and not np.isinf(variance):
                    feature_variances.append((feature_name, variance))
            else:
                feature_variances.append((feature_name, 0.0))
        except (TypeError, ValueError):
            continue

    feature_variances.sort(key=lambda item: item[1], reverse=True)
    selected_features = [name for name, _ in feature_variances[:max_features_limit]]
    removed_count = len(self.features) - len(selected_features)
    self.features = selected_features
    logger.info(
        "Removed %s low-variance features",
        removed_count,
        extra={"removed": removed_count, "final_count": len(self.features)},
    )


def _refresh_features(self: Any) -> None:
    """Update cached feature list and observation space when schema changes."""
    exclude_cols = ["ts", "timestamp", "exchange", "pair", "episode_id"]
    self.features = [column for column in self.df.columns if column not in exclude_cols]
    if not self.features:
        self.features = list(self.df.columns)
    self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(self.features),),
        dtype=np.float32,
    )


def _build_fast_access_buffers(
    self: Any,
) -> None:  # pragma: no cover - exercised via environment usage
    """Precompute numpy buffers for efficient runtime access."""
    if not self.features:
        self._feature_matrix = np.empty((0, 0), dtype=np.float32)
        self._price_array = None
        self._close_array = None
        self._atr_array = None
        self._episode_id_array = None
        self._nonfinite_rows = set()
        self._nonfinite_warned_rows = set()
        return

    feature_view = self.df[self.features]
    feature_matrix = feature_view.to_numpy(dtype=np.float32, copy=False)
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(-1, 1)
    self._feature_matrix = np.ascontiguousarray(feature_matrix)

    # Diagnostic logging: confirm feature matrix shape and feature list
    try:
        logger.info(
            "[DBG] _build_fast_access_buffers: feature_matrix.shape=%s, features_len=%s",
            getattr(self, "_feature_matrix").shape,
            len(getattr(self, "features", [])),
        )
        # Log a short preview of feature names and dataframe tail columns for mapping checks
        logger.info(
            "features preview=%s, df_columns_tail=%s",
            getattr(self, "features", [])[:12],
            list(self.df.columns[-6:]) if hasattr(self, "df") else None,
        )
        # Debug logging for fast access buffers
        try:
            logger.debug(
                "_build_fast_access_buffers: feature_matrix.shape=%s, features_len=%d",
                getattr(self, "_feature_matrix").shape,
                len(getattr(self, "features", [])),
            )
            logger.debug(
                "features preview=%s, df_columns_tail=%s",
                getattr(self, "features", [])[:24],
                list(self.df.columns[-12:]) if hasattr(self, "df") else None,
            )
        except Exception:
            pass
    except Exception:
        logger.exception("Failed to emit debug diagnostics for feature matrix")

    mask = ~np.isfinite(self._feature_matrix)
    if np.any(mask):
        affected_rows = np.where(np.any(mask, axis=1))[0]
        self._nonfinite_rows = {int(idx) for idx in affected_rows}
        np.nan_to_num(self._feature_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if affected_rows.size and affected_rows.size not in self._nonfinite_warned_rows:
            logger.warning(
                "Feature matrix contained non-finite values. Sanitized %s rows.",
                affected_rows.size,
                extra={"affected_rows": affected_rows.size},
            )
            self._nonfinite_warned_rows.add(affected_rows.size)
    else:
        self._nonfinite_rows = set()
    self._nonfinite_warned_rows = set()
    self._feature_matrix.setflags(write=False)

    self._price_array = _extract_numeric_column(
        self, ("price", "close", "adj_close", "open"), fallback=None
    )
    self._close_array = _extract_numeric_column(
        self, ("close", "price", "adj_close", "open"), fallback=None
    )
    self._atr_array = _extract_numeric_column(
        self,
        ("atr_10", "atr_14", "atr_simplified", "ATR", "ATR_simplified"),
        fallback=1.0,
    )

    if self._price_array is not None and self._price_array.size:
        self._price_array.setflags(write=False)
    if self._close_array is not None and self._close_array.size:
        self._close_array.setflags(write=False)
    if self._atr_array is not None and self._atr_array.size:
        self._atr_array.setflags(write=False)

    if self._episode_id_column and self._episode_id_column in self.df.columns:
        self._episode_id_array = self.df[self._episode_id_column].to_numpy(copy=False)
    else:
        self._episode_id_array = None


def _extract_numeric_column(
    self: Any,
    candidates: Iterable[str],
    *,
    fallback: Optional[float],
) -> Optional[NDArray[np.float32]]:
    """Extract contiguous float32 array from first available numeric column."""
    for name in candidates:
        if name not in self.df.columns:
            continue
        series = self.df[name]
        if not ptypes.is_numeric_dtype(series):
            continue
        # Force copy=True to ensure the array is writable for np.nan_to_num
        array = np.ascontiguousarray(series.to_numpy(dtype=np.float32, copy=True))
        if fallback is not None and array.size:
            np.nan_to_num(
                array, copy=False, nan=fallback, posinf=fallback, neginf=fallback
            )
        return array

    if fallback is None:
        return None

    if self.n_steps <= 0:
        return np.empty(0, dtype=np.float32)

    return np.full(self.n_steps, fallback, dtype=np.float32)


def _select_features_by_correlation_in_env(
    self: Any,
    features: List[str],
    correlation_threshold: float = 0.95,
    *,
    target_feature_count: Optional[int] = None,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Select features by removing highly correlated ones based on current DataFrame.

    Args:
        features: List of feature names to consider.
        correlation_threshold: Correlation threshold above which features are considered redundant.
        target_feature_count: Optional hard limit for remaining features after reduction.

    Returns:
        Tuple[List[str], Dict[str, List[str]]]: reduced feature list (preserving original order)
        and metadata describing removed feature categories.
    """
    reduction_stats: Dict[str, List[str]] = {
        "non_numeric": [],
        "constant": [],
        "correlated": [],
    }

    if not features or len(features) <= 1:
        return features, reduction_stats

    # Sanitize target count
    try:
        if target_feature_count is not None:
            target_feature_count = int(target_feature_count)
            if target_feature_count <= 0:
                target_feature_count = None
    except (TypeError, ValueError):
        target_feature_count = None

    available_features = [name for name in features if name in self.df.columns]
    if not available_features:
        return features, reduction_stats

    for name in available_features:
        series = self.df[name]
        if not ptypes.is_numeric_dtype(series):
            reduction_stats["non_numeric"].append(name)

    numeric_features = [
        name
        for name in available_features
        if name not in reduction_stats["non_numeric"]
    ]

    if not numeric_features:
        return features, reduction_stats

    numeric_frame = self.df[numeric_features]
    variances = numeric_frame.var(ddof=0).astype(float)

    constant_features = [
        name
        for name, value in variances.items()
        if not np.isfinite(value) or value <= 1e-12
    ]
    if constant_features:
        reduction_stats["constant"].extend(constant_features)
        numeric_features = [
            name for name in numeric_features if name not in constant_features
        ]
        numeric_frame = numeric_frame.drop(columns=constant_features, errors="ignore")
        variances = variances.drop(labels=constant_features, errors="ignore")

    if not numeric_features:
        # Fall back to the highest variance features among the original numeric set
        top_features = (
            variances.sort_values(ascending=False)
            .head(target_feature_count or 1)
            .index.tolist()
        )
        return top_features, reduction_stats

    if len(numeric_features) == 1:
        final_features = [name for name in features if name == numeric_features[0]]
        return final_features, reduction_stats

    corr_matrix = numeric_frame.corr().abs().fillna(0.0)
    remaining = list(numeric_features)
    variance_map = variances.to_dict()
    original_positions = {name: idx for idx, name in enumerate(features)}

    while len(remaining) > 1:
        corr_subset = corr_matrix.loc[remaining, remaining].to_numpy(copy=True)
        if corr_subset.size == 0:
            break
        np.fill_diagonal(corr_subset, 0.0)
        max_corr = float(np.nanmax(corr_subset))
        if not np.isfinite(max_corr) or max_corr < correlation_threshold:
            break
        idx_flat = int(np.nanargmax(corr_subset))
        subset_size = len(remaining)
        i, j = divmod(idx_flat, subset_size)
        if i == j:
            break
        feature_i = remaining[i]
        feature_j = remaining[j]
        var_i = float(variance_map.get(feature_i, 0.0))
        var_j = float(variance_map.get(feature_j, 0.0))
        mean_corr_i = float(corr_matrix.loc[feature_i, remaining].mean())
        mean_corr_j = float(corr_matrix.loc[feature_j, remaining].mean())

        if not np.isfinite(var_i):
            drop = feature_i
        elif not np.isfinite(var_j):
            drop = feature_j
        else:
            if var_i == var_j:
                drop = feature_i if mean_corr_i >= mean_corr_j else feature_j
            else:
                drop = feature_i if var_i < var_j else feature_j

        reduction_stats["correlated"].append(drop)
        remaining.remove(drop)
        if target_feature_count is not None and len(remaining) <= target_feature_count:
            break

    if target_feature_count is not None and len(remaining) > target_feature_count:
        ranked = sorted(
            remaining,
            key=lambda name: (
                variance_map.get(name, -np.inf),
                -original_positions.get(name, 0),
            ),
            reverse=True,
        )
        keep = set(ranked[:target_feature_count])
        extra_removed = [name for name in remaining if name not in keep]
        reduction_stats["correlated"].extend(extra_removed)
        remaining = [name for name in remaining if name in keep]

    if not remaining:
        ranked = sorted(
            numeric_features,
            key=lambda name: (
                variance_map.get(name, -np.inf),
                -original_positions.get(name, 0),
            ),
            reverse=True,
        )
        remaining = ranked[: max(1, target_feature_count or 1)]

    final_features = [name for name in features if name in remaining]
    return final_features, reduction_stats
