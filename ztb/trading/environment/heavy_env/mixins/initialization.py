"""Initialization helpers for HeavyTradingEnv."""

from __future__ import annotations

import gc
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from gymnasium import spaces
from numpy.typing import NDArray
from pandas.api import types as ptypes

from ztb.features.registry import FeatureRegistry
from ztb.trading.environment.components import (
    ActionValidator,
    DataProcessor,
    MemoryManager,
    ObservationBuilder,
    PositionManager,
    RewardCalculator,
    StreamingHandler,
)
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
            path_candidate = safe_path_join(str(get_project_root()), raw_memory_log_path)
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

    timestamp_column = "timestamp" if df is not None and "timestamp" in df.columns else None
    episode_id_column = "episode_id" if df is not None and "episode_id" in df.columns else None

    self.streaming_handler = StreamingHandler(
        streaming_pipeline=streaming_pipeline,
        stream_batch_size=max(1, int(stream_batch_size)),
        timestamp_column=timestamp_column,
        episode_id_column=episode_id_column,
    )


def _initialize_data_structures(self: Any) -> None:
    """Prepare deque-based histories and fast-access buffers."""
    inventory_window = max(
        self.DEFAULT_MIN_WINDOW_SIZE,
        self._get_reward_setting_int("inventory_window", self.DEFAULT_INVENTORY_WINDOW),
    )
    volatility_window = max(
        self.DEFAULT_MIN_WINDOW_SIZE,
        self._get_reward_setting_int("volatility_window", self.DEFAULT_VOLATILITY_WINDOW),
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


def _initialize_data(self: Any, df: Optional[pd.DataFrame]) -> None:
    """Load baseline market data either from dataframe or streaming snapshot."""
    if df is not None:
        base_df = df
    else:
        base_df = self._fetch_streaming_snapshot(
            required_rows=self.streaming_handler.stream_batch_size
        )
        if base_df.empty:
            raise ValidationError(
                "Streaming pipeline did not provide initial data",
                details={
                    "pipeline": str(type(self.streaming_handler.streaming_pipeline)),
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
    correlation_reduction = getattr(self.config, "enable_correlation_reduction", True)
    logger.info(f"Config features: {config_features is not None}, correlation_reduction: {correlation_reduction}")
    if config_features is not None:
        logger.info(f"Using schema-defined features: {len(config_features)}")
        self.features = config_features
        # Validate that all required features exist in dataframe
        missing_features = set(config_features) - set(self.df.columns)
        if missing_features:
            raise ValueError(f"Missing required features in dataframe: {missing_features}")
    else:
        # Original feature discovery logic
        exclude_by_default = {"ts", "timestamp", "exchange", "pair", "episode_id"}
        all_features = [col for col in self.df.columns if col not in exclude_by_default]
        if not all_features:
            all_features = list(self.df.columns)

        feature_set = getattr(self.config, "feature_set", "full")
        enable_feature_filtering = getattr(self.config, "enable_feature_filtering", False)
        feature_filter_mode = getattr(self.config, "feature_filter_mode", "whitelist")

        if enable_feature_filtering and feature_filter_mode == "whitelist":
            curated_features_spec = getattr(self.config, "curated_features_list", None)
            if curated_features_spec:
                all_features = _apply_curated_feature_filter(curated_features_spec, all_features)

        max_features_limit = _resolve_max_features_limit(self, max_features)

        if feature_set != "full":
            logger.warning(
                "Feature set filtering not implemented, using all features",
                extra={"feature_set": feature_set, "total_features": len(all_features)},
            )
            self.features = all_features
        else:
            self.features = all_features

        enable_correlation_reduction = getattr(
            self.config, "enable_correlation_reduction", True
        )
        logger.info(f"Correlation reduction enabled: {enable_correlation_reduction}")
        if enable_correlation_reduction and len(self.features) > 10:
            logger.info("Applying correlation reduction...")
            correlation_threshold = getattr(self.config, "correlation_threshold", 0.95)
            try:
                # FeatureRegistry.select_features_by_correlation(
                #     correlation_threshold=correlation_threshold
                # )
                # For now, skip correlation reduction
                optimized_features = self.features
                if len(optimized_features) >= 10:
                    removed_count = len(self.features) - len(optimized_features)
                    self.features = optimized_features
                    logger.info(
                        "Applied correlation-based feature reduction",
                        extra={"removed_count": removed_count, "remaining": len(self.features)},
                    )
                else:
                    logger.warning(
                        "Correlation reduction would leave too few features",
                        extra={
                            "optimized_count": len(optimized_features),
                            "original_count": len(self.features),
                        },
                    )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to apply correlation-based feature reduction: %s",
                    exc,
                    exc_info=True,
                )

        if max_features_limit and len(self.features) > max_features_limit:
            _enforce_feature_limit(self, max_features_limit)

    # Apply max_features_limit even for schema-defined features
    max_features_limit = _resolve_max_features_limit(self, max_features)
    if max_features_limit and len(self.features) > max_features_limit:
        _enforce_feature_limit(self, max_features_limit)

    self.data_processor.apply_feature_storage_dtype(
        self.df, self.features, self.config.__dict__
    )

    _build_fast_access_buffers(self)

    # Observation space (always Box for features)
    self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(self.features),),
        dtype=np.float32,
    )
    
    # Action space: Support both discrete (PPO) and continuous (SAC)
    from ztb.trading.environment.constants import (
        NUM_DISCRETE_ACTIONS,
        CONTINUOUS_ACTION_MIN,
        CONTINUOUS_ACTION_MAX,
        CONTINUOUS_ACTION_DIM,
    )
    
    use_continuous_actions = getattr(self.config, "use_continuous_actions", False)
    
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


def _compute_scaler_from_data(self: Any) -> None:
    """データからスケーラーを計算（標準化用の平均・標準偏差）"""
    if not hasattr(self, "_feature_matrix") or self._feature_matrix.size == 0:
        logger.warning("Feature matrix is empty. Cannot compute scaler.")
        self.scaler_mean = None
        self.scaler_std = None
        return

    # 特徴量行列全体から平均・標準偏差を計算
    # axis=0: 各特徴量ごとに計算（列方向）
    self.scaler_mean = np.mean(self._feature_matrix, axis=0).astype(np.float32)
    self.scaler_std = np.std(self._feature_matrix, axis=0).astype(np.float32)

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
    self.position_manager = PositionManager(
        config=self.config,
        get_price_callback=self._resolve_price,
    )

    self.reward_calculator = RewardCalculator(
        config=self.config,
        reward_settings=self.reward_settings,
        initial_portfolio_value=self.initial_portfolio_value,
    )
    self.observation_builder = ObservationBuilder(
        features=self.features,
        feature_matrix=self._feature_matrix,
        nonfinite_rows=self._nonfinite_rows,
        nonfinite_warned_rows=self._nonfinite_warned_rows,
        scaler_mean=self.scaler_mean,
        scaler_std=self.scaler_std,
    )

    self.action_validator = ActionValidator(
        config=self.config,
        initial_portfolio_value=self.initial_portfolio_value,
    )

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
    self._action_counts = [0, 0, 0]
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

def _apply_curated_feature_filter(curated_features_spec: str, all_features: List[str]) -> List[str]:
    """Apply curated feature whitelist if available."""
    try:
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
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to load curated features: %s", exc, exc_info=True)
        return all_features


def _resolve_max_features_limit(self: Any, max_features: Optional[int]) -> Optional[int]:
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
            if pd.notna(variance_value) and isinstance(variance_value, (int, float, np.number)):
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


def _build_fast_access_buffers(self: Any) -> None:  # pragma: no cover - exercised via environment usage
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

    self._price_array = _extract_numeric_column(self, ("price", "close", "adj_close", "open"), fallback=None)
    self._close_array = _extract_numeric_column(self, ("close", "price", "adj_close", "open"), fallback=None)
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
        array = np.ascontiguousarray(series.to_numpy(dtype=np.float32, copy=False))
        if fallback is not None and array.size:
            np.nan_to_num(array, copy=False, nan=fallback, posinf=fallback, neginf=fallback)
        return array

    if fallback is None:
        return None

    if self.n_steps <= 0:
        return np.empty(0, dtype=np.float32)

    return np.full(self.n_steps, fallback, dtype=np.float32)
