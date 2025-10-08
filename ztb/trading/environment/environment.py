# Heavy Trading Environment for Reinforcement Learning
# 重特徴量ベースの取引環境

import gc
import time
from collections import deque
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import gymnasium as gym
import numpy as np
import pandas as pd
import psutil
from gymnasium import spaces
from numpy.typing import NDArray
from pandas.api import types as ptypes

from ztb.features.registry import FeatureRegistry
from ztb.trading.constants import ACTION_BUY, ACTION_SELL
from ztb.utils.fee_model import ExchangeFeeModel
from ztb.utils.memory.dtypes import optimize_dtypes
from ztb.types.protocols import TradingEnvironment
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.trading.environment.components import PositionManager, RewardCalculator

# Type aliases for better type safety
Observation = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
Action = int
Reward = float


class InfoDict(TypedDict, total=False):
    """Info dictionary returned by environment step/reset.
    
    Extends gymnasium's standard info dict with trading-specific information.
    """
    # Gymnasium standard fields (inherited)
    # current_step: int
    # total_steps: int
    
    # Trading-specific fields
    position: float
    total_pnl: float
    trades_count: int
    features: List[str]
    config: "EnvironmentConfig"  # Forward reference to avoid circular import
    pnl: float
    action: int
    step: int
    portfolio_value: float
    atr: float
    position_utilisation: float
    action_masks: NDArray[np.bool_]


class StatisticsDict(TypedDict, total=False):
    """Statistics dictionary returned by get_statistics."""
    total_reward: float
    mean_reward: float
    std_reward: float
    sharpe_ratio: float
    max_reward: float
    total_trades: int
    win_rate: float


Info = InfoDict


EPSILON = 1e-6  # 小さい値（ゼロ除算防止用）


if TYPE_CHECKING:
    from ztb.data.streaming_pipeline import StreamingPipeline
    from ztb.trading.live.data.stream_to_bars import StreamToBarsConverter


class HeavyTradingEnv(gym.Env[NDArray[np.float32], spaces.Discrete], TradingEnvironment):
    """
    Heavy Feature-Based Trading Environment for Reinforcement Learning.

    This environment implements a sophisticated trading simulation using comprehensive
    feature sets including price, technical, and risk indicators. It supports both
    long and short positions with realistic trading mechanics.

    Features:
    - State: All price, technical, and risk features from FeatureRegistry
    - Actions: 0=hold, 1=buy (open long/close short), 2=sell (close long/open short)
    - Reward: Position-adjusted PnL normalized by ATR, with risk penalties
    - Position: -1.0 (short), 0.0 (neutral), 1.0 (long)
    - NaN Handling: Zero-filling for missing values
    - Action Masking: Prevents illegal actions based on current position

    Reward Function:
    - Base: (position * step_pnl) / (atr_14 + epsilon)
    - Penalties: Position size limits, trade frequency, inventory costs
    - Normalization: ATR-based scaling for market volatility adaptation

    Action Space Details:
    - BUY (1): Legal when position ≤ 0, opens long or closes short
    - SELL (2): Legal when position ≥ 0, closes long or opens short
    - HOLD (0): Always legal, maintains current position

    Observation Space:
    - Shape: (n_features,) where n_features varies by configuration
    - Dtype: float32 for neural network compatibility
    - Features: Price data, technical indicators, risk metrics, position info
    """

    _current_episode_actions: List[int]
    portfolio_value_history: deque[float]  # Memory optimized: deque with maxlen
    # action_history: deque[int] - defined in __init__ with maxlen

    def _log_memory_usage(
        self, context: str, *, df_override: Optional[pd.DataFrame] = None
    ) -> None:
        if not getattr(self, "_memory_logging_enabled", False):
            return

        process = getattr(self, "_process", None)
        if process is None:
            process = psutil.Process()
            self._process = process

        rss_mb = process.memory_info().rss / 1024 / 1024
        target_df = (
            df_override if df_override is not None else getattr(self, "df", None)
        )
        df_mem_mb = (
            target_df.memory_usage(deep=True).sum() / 1024 / 1024
            if isinstance(target_df, pd.DataFrame)
            else 0.0
        )

        message = f"[Memory][HeavyTradingEnv][{context}] df={df_mem_mb:.2f} MB RSS={rss_mb:.2f} MB"
        print(message)

        log_path = getattr(self, "_memory_log_path", None)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{pd.Timestamp.now().isoformat()},{context},{df_mem_mb:.4f},{rss_mb:.4f}\n"
                )

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        config: Optional[Union[EnvironmentConfig, Dict[str, Any]]] = None,
        *,
        random_start: bool = False,
        streaming_pipeline: Optional["StreamingPipeline"] = None,
        stream_batch_size: int = 256,
        stream_to_bars_converter: Optional["StreamToBarsConverter"] = None,
        max_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        if df is None and streaming_pipeline is None:
            raise ValueError("Either df or streaming_pipeline must be provided")

        # Convert config to EnvironmentConfig
        if isinstance(config, EnvironmentConfig):
            self.config = config
        else:
            self.config = EnvironmentConfig.from_dict(config)

        self._process = psutil.Process()
        # 報酬関連の安全なデフォルトを設定
        self.reward_settings: RewardSettings = {
            "position_soft_cap": self.config.reward_position_soft_cap,
            "position_penalty_scale": self.config.reward_position_penalty_scale,
            "position_penalty_exp": self.config.reward_position_penalty_exponent,
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

        # configからreward_settingsをマージ（上書き）
        if hasattr(self.config, "reward_settings") and self.config.reward_settings:
            self.reward_settings = {**self.reward_settings, **self.config.reward_settings}  # type: ignore[assignment]

        # 取引コストを動的に設定（取引所に基づく）
        self.fee_model = ExchangeFeeModel()
        self.fee_model.set_exchange(self.config.exchange)
        self.config.transaction_cost = self.fee_model.get_fee_rate(
            "buy"
        )  # デフォルトで買いの手数料を使用

        self.initial_portfolio_value = float(self.config.initial_portfolio_value)
        self.portfolio_value = self.initial_portfolio_value

        inventory_window = max(8, self._get_reward_setting_int("inventory_window", 64))
        volatility_window = max(
            8, self._get_reward_setting_int("volatility_window", 16)
        )
        self.position_abs_history: deque[float] = deque(maxlen=inventory_window)
        self.pnl_history: deque[float] = deque(maxlen=volatility_window)
        self.trade_interval_history: deque[int] = deque(maxlen=32)
        self._last_trade_step: Optional[int] = None
        self._consecutive_trade_steps = 0

        memory_log_path_cfg = getattr(self.config, "memory_log_path", None)
        self._memory_log_path = (
            Path(memory_log_path_cfg) if memory_log_path_cfg else None
        )
        if self._memory_log_path and not self._memory_log_path.exists():
            self._memory_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._memory_log_path.write_text(
                "timestamp,context,df_mb,rss_mb\n", encoding="utf-8"
            )

        self._memory_logging_enabled = bool(
            getattr(self.config, "memory_logging_enabled", False)
        )

        memory_interval_cfg = getattr(self.config, "memory_log_interval_steps", None)
        try:
            self._memory_log_interval_steps = max(1, int(memory_interval_cfg or 2000))
        except (TypeError, ValueError):
            self._memory_log_interval_steps = 2000

        gc_interval_cfg = getattr(self.config, "gc_collect_interval_steps", None)
        try:
            self._gc_step_interval = max(0, int(gc_interval_cfg or 0))  # Default to 0 for memory efficiency
        except (TypeError, ValueError):
            self._gc_step_interval = 0

        preprocess_chunk_cfg = getattr(self.config, "preprocess_chunk_size", None)
        try:
            self._preprocess_chunk_size = max(1, int(preprocess_chunk_cfg or 32))
        except (TypeError, ValueError):
            self._preprocess_chunk_size = 32

        self._last_memory_log_step = 0
        # Use random_start from parameter, fallback to config if available
        self.random_start = random_start or getattr(self.config, "random_start", False)

        self.streaming_pipeline = streaming_pipeline
        self.stream_batch_size = max(1, int(stream_batch_size))
        self.stream_to_bars_converter = stream_to_bars_converter
        self._stream_last_timestamp: Optional[pd.Timestamp] = None
        self._stream_rows_appended = 0
        self._base_columns: List[str] = []

        # Fast-access buffers (populated after preprocessing)
        self._feature_matrix: NDArray[np.float32] = np.empty((0, 0), dtype=np.float32)
        self._price_array: Optional[NDArray[np.float32]] = None
        self._close_array: Optional[NDArray[np.float32]] = None
        self._atr_array: Optional[NDArray[np.float32]] = None
        self._episode_id_array: Optional[np.ndarray] = None
        self._nonfinite_rows: Set[int] = set()
        self._nonfinite_warned_rows: Set[int] = set()

        # Memory optimization: avoid unnecessary copy
        if df is not None:
            base_df = df  # Use reference instead of copy
        else:
            base_df = self._fetch_streaming_snapshot(
                required_rows=self.stream_batch_size
            )
            if base_df.empty:
                raise ValueError("Streaming pipeline did not provide initial data")

        # データの前処理
        self.df = self._preprocess_data(base_df)
        if df is None:  # Only delete if we created it
            del base_df
        gc.collect()
        self._log_memory_usage("post_init")

        # 積極的なメモリ最適化
        # Note: Removed redundant copy() - _preprocess_data already returns a copy if needed
        if hasattr(self, "_memory_logging_enabled") and self._memory_logging_enabled:
            # インデックスを最適化（インプレース操作）
            if not self.df.index.is_monotonic_increasing:
                self.df.sort_index(inplace=True)

        self.n_steps = len(self.df)
        self._base_columns = list(self.df.columns)

        # 特徴量の選択（除外する列を指定）
        exclude_cols = [
            "ts",
            "timestamp",
            "exchange",
            "pair",
            "episode_id",
            "side",
            "source",
        ]
        all_features = [c for c in self.df.columns if c not in exclude_cols]
        if not all_features:
            # 全特徴量が除外された場合は全列を利用
            all_features = list(self.df.columns)

        # 特徴量セットに基づいてフィルタリング
        feature_set = getattr(self.config, "feature_set", "full")
        
        # ========================================================================
        # CURATED FEATURES FILTERING (v377)
        # ========================================================================
        # Apply whitelist filtering if enabled
        enable_feature_filtering = getattr(self.config, "enable_feature_filtering", False)
        feature_filter_mode = getattr(self.config, "feature_filter_mode", "whitelist")
        
        if enable_feature_filtering and feature_filter_mode == "whitelist":
            # Try to load curated features list
            curated_features_spec = getattr(self.config, "curated_features_list", None)
            if curated_features_spec:
                try:
                    # Parse spec like "curated_features.py::CURATED_FEATURES"
                    if "::" in curated_features_spec:
                        module_name = curated_features_spec.split("::")[0].replace(".py", "")
                        var_name = curated_features_spec.split("::")[1]
                        
                        # Dynamic import
                        import importlib.util
                        import sys
                        spec = importlib.util.spec_from_file_location(
                            module_name,
                            curated_features_spec.split("::")[0]
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = module
                            spec.loader.exec_module(module)
                            curated_list = getattr(module, var_name, None)
                            
                            if curated_list:
                                # Filter to only curated features that exist in data
                                original_count = len(all_features)
                                all_features = [f for f in curated_list if f in all_features]
                                removed = original_count - len(all_features)
                                print(f"✅ Applied curated features filter: kept {len(all_features)}/{original_count} features (removed {removed})")
                            else:
                                print(f"⚠️  Warning: Could not find {var_name} in {module_name}")
                    else:
                        print(f"⚠️  Warning: Invalid curated_features_list format: {curated_features_spec}")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to load curated features: {e}")
                    import traceback
                    traceback.print_exc()
        
        # ========================================================================
        # UNIFIED MAX_FEATURES EXTRACTION (Bug #52 fix)
        # ========================================================================
        # Extract max_features from multiple possible sources with priority:
        # 1) Constructor argument (max_features parameter)
        # 2) Top-level config.max_features
        # 3) memory_optimization.max_features section
        # 4) ppo.max_features section
        # 5) Direct config attribute
        max_features_limit = max_features
        if max_features_limit is None:
            # Try dict-like access first (for Dict[str, Any] config)
            if hasattr(self.config, 'get') and callable(getattr(self.config, 'get')):
                config_dict = self.config  # type: ignore[assignment]
                max_features_limit = (
                    config_dict.get("max_features") or  # type: ignore[attr-defined]
                    (config_dict.get("memory_optimization", {}) or {}).get("max_features") or  # type: ignore[attr-defined]
                    (config_dict.get("ppo", {}) or {}).get("max_features")  # type: ignore[attr-defined]
                )
            
            # Fall back to attribute access (for EnvironmentConfig objects)
            if max_features_limit is None:
                max_features_limit = getattr(self.config, "max_features", None)
        
        if feature_set != "full":
            # TODO: Implement feature set filtering
            # For now, use all available features
            print(f"Warning: Feature set filtering not implemented, using all features")
            self.features = all_features
        else:
            self.features = all_features

        # 相関に基づく特徴量削減を適用
        enable_correlation_reduction = getattr(
            self.config, "enable_correlation_reduction", True
        )
        if enable_correlation_reduction and len(self.features) > 10:
            correlation_threshold = getattr(self.config, "correlation_threshold", 0.95)
            try:
                optimized_features = FeatureRegistry.select_features_by_correlation(
                    correlation_threshold=correlation_threshold
                )
                # データに存在する最適化された特徴量のみを選択
                optimized_features = [
                    f for f in optimized_features if f in self.features
                ]
                if len(optimized_features) >= 10:  # 最低10個の特徴量を確保
                    removed_count = len(self.features) - len(optimized_features)
                    self.features = optimized_features
                    print(
                        f"Applied correlation-based feature reduction: removed {removed_count} highly correlated features"
                    )
                else:
                    print(
                        f"Warning: Correlation reduction would leave too few features ({len(optimized_features)}), keeping original set"
                    )
            except Exception as e:
                print(
                    f"Warning: Failed to apply correlation-based feature reduction: {e}"
                )
        
        # CRITICAL: Apply hard feature limit if specified
        if max_features_limit and len(self.features) > max_features_limit:
            # Select features with highest variance (most informative)
            print(f"⚠️  MEMORY OPTIMIZATION: Reducing features from {len(self.features)} to {max_features_limit}")
            
            # Calculate variance for each feature
            feature_variances: List[Tuple[str, float]] = []
            for feat in self.features:
                if feat in self.df.columns:
                    try:
                        var_value = self.df[feat].var()
                        if pd.notna(var_value) and isinstance(var_value, (int, float, np.number)):
                            var = float(var_value)
                            if not np.isnan(var) and not np.isinf(var):
                                feature_variances.append((feat, var))
                        else:
                            var = 0.0
                            feature_variances.append((feat, var))
                    except (TypeError, ValueError):
                        # Skip features that can't be processed for variance
                        continue
            
            # Sort by variance (descending) and take top N
            feature_variances.sort(key=lambda x: x[1], reverse=True)
            selected_features = [f for f, _ in feature_variances[:max_features_limit]]
            
            removed_count = len(self.features) - len(selected_features)
            self.features = selected_features
            print(f"   Removed {removed_count} low-variance features")
            print(f"   Final feature count: {len(self.features)}")

        self._apply_feature_storage_dtype()

        # Precompute fast-access numpy buffers to avoid per-step pandas overhead
        self._build_fast_access_buffers()

        # 状態空間: 特徴量ベクトル
        self.observation_space = cast(
            spaces.Space[NDArray[np.float32]],
            spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32
            ),
        )

        # 行動空間: hold, buy, sell
        self.action_space = cast(spaces.Space[spaces.Discrete], spaces.Discrete(3))

        # 環境状態
        self.current_step = 0
        
        # Initialize PositionManager
        self.position_manager = PositionManager(
            config=self.config,
            get_price_callback=self._resolve_price,
        )
        
        # Backward compatibility: expose position manager properties
        self.position = 0.0  # Will be synced with position_manager
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.trades_count = 0

        # Initialize RewardCalculator
        self.reward_calculator = RewardCalculator(
            config=self.config,
            reward_settings=self.reward_settings,
            initial_portfolio_value=self.initial_portfolio_value,
        )

        # ストリーミング関連
        self._timestamp_column = "timestamp" if "timestamp" in self.df.columns else None
        self._episode_id_column = (
            "episode_id" if "episode_id" in self.df.columns else None
        )
        if not self._timestamp_column:
            self._stream_rows_appended = len(self.df)

        # 報酬計算用の履歴
        # Memory optimization: Use deque with maxlen instead of list
        self._max_history_length = getattr(self.config, "max_history_length", 512)
        self.reward_history: deque[float] = deque(maxlen=self._max_history_length)
        self.position_history: deque[float] = deque(maxlen=self._max_history_length)
        self._action_counts: list[int] = [
            0,
            0,
            0,
        ]  # Track action usage for balance bonus
        self._current_episode_actions: list[int] = []
        
        # Memory optimization: Use deque for action_history with maxlen
        action_history_limit = getattr(self.config, "max_action_history", None)
        try:
            self._max_action_history = max(10, int(action_history_limit or 256))
        except (TypeError, ValueError):
            self._max_action_history = 256
        
        self.action_history: deque[int] = deque(maxlen=self._max_action_history)

        # Note: _max_history_length is already set above for reward/position history

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データの前処理とメモリ最適化"""
        if df.empty:
            return df.copy()  # Empty DataFrame, safe to copy

        # Memory optimization: Use inplace operations where possible
        df_processed = df.fillna(0)
        
        # Reset index only if needed, using inplace operation
        if not df_processed.index.equals(
            pd.RangeIndex(start=0, stop=len(df_processed), step=1)
        ):
            df_processed.reset_index(drop=True, inplace=True)
        # Note: Removed unnecessary copy() when index is already correct

        exclude_cols = [
            "ts",
            "timestamp",
            "exchange",
            "pair",
            "episode_id",
            "side",
            "source",
        ]
        df_processed = df_processed.drop(
            columns=[c for c in exclude_cols if c in df_processed.columns],
            errors="ignore",
        )

        optimized, _ = optimize_dtypes(
            df_processed,
            target_float_dtype="float32",
            target_int_dtype="int32",
            convert_objects_to_category=True,
            chunk_size=self._preprocess_chunk_size,
            memory_report=self._memory_logging_enabled,
        )

        numeric_cols = optimized.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            optimized[numeric_cols] = optimized[numeric_cols].astype(
                np.float32, copy=False
            )

        bool_cols = optimized.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            optimized[bool_cols] = optimized[bool_cols].astype(np.int8, copy=False)

        self._log_memory_usage("preprocess", df_override=optimized)

        del df_processed
        if not self._gc_step_interval:
            gc.collect()

        return optimized

    def _fetch_streaming_snapshot(self, required_rows: int) -> pd.DataFrame:
        """ストリーミングパイプラインから初期スナップショットを取得"""
        if not self.streaming_pipeline:
            return pd.DataFrame()

        snapshot = self.streaming_pipeline.buffer.to_dataframe(
            last_n=max(required_rows, self.stream_batch_size)
        )
        if snapshot.empty:
            return snapshot
        return snapshot.reset_index(drop=True)

    def _prepare_stream_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """環境が扱える形式にストリーミングデータを整形"""
        if batch.empty:
            return batch

        if not self._base_columns:
            self._base_columns = list(batch.columns)

        missing = [col for col in self._base_columns if col not in batch.columns]
        for col in missing:
            batch[col] = 0.0

        extra = [col for col in batch.columns if col not in self._base_columns]
        if extra:
            self._base_columns.extend(extra)
            self.df = self.df.reindex(columns=self._base_columns, fill_value=0)

        batch = batch[self._base_columns]
        return self._preprocess_data(batch)

    def _append_streaming_rows(self) -> bool:
        """ストリーミングバッファから新規行を取り込み"""
        if not self.streaming_pipeline:
            return False

        buffer_df = self.streaming_pipeline.buffer.to_dataframe()
        if buffer_df.empty:
            return False

        if self._timestamp_column and "timestamp" in buffer_df.columns:
            buffer_df = buffer_df.sort_values("timestamp").reset_index(drop=True)
            if self._stream_last_timestamp is not None:
                buffer_df = buffer_df[
                    buffer_df["timestamp"] > self._stream_last_timestamp
                ]
        else:
            buffer_df = buffer_df.iloc[self._stream_rows_appended :]

        if buffer_df.empty:
            return False

        if self.stream_batch_size:
            buffer_df = buffer_df.tail(self.stream_batch_size)

        prepared = self._prepare_stream_batch(buffer_df)
        if prepared.empty:
            return False

        self.df = pd.concat([self.df, prepared], ignore_index=True, copy=False)
        self.n_steps = len(self.df)
        self._stream_rows_appended += len(prepared)

        if self._timestamp_column and "timestamp" in buffer_df.columns:
            self._stream_last_timestamp = pd.to_datetime(buffer_df["timestamp"]).max()

        self._refresh_features()
        self._apply_feature_storage_dtype()
        self._build_fast_access_buffers()
        self._log_memory_usage("stream_append")

        del prepared
        del buffer_df
        if not self._gc_step_interval:
            gc.collect()

        return True

    def _refresh_features(self) -> None:
        """特徴量と観測空間を更新"""
        exclude_cols = ["ts", "timestamp", "exchange", "pair", "episode_id"]
        self.features = [c for c in self.df.columns if c not in exclude_cols]
        if not self.features:
            self.features = list(self.df.columns)
        self.observation_space = cast(
            spaces.Space[NDArray[np.float32]],
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.features),),
                dtype=np.float32,
            ),
        )

    def _apply_feature_storage_dtype(self) -> None:
        """Ensure feature columns use the configured storage dtype"""
        feature_dtype = str(
            getattr(self.config, "feature_storage_dtype", "float16")
        ).lower()
        dtype_map = {"float16": np.float16, "float32": np.float32}
        target_dtype = dtype_map.get(feature_dtype, np.float32)

        protected = {
            str(col).lower() for col in getattr(self.config, "precision_columns", [])
        }
        candidate_features = [
            col
            for col in self.features
            if col in self.df.columns
            and ptypes.is_numeric_dtype(self.df[col])
            and col.lower() not in protected
        ]
        if not candidate_features:
            return

        safe_features = []
        if target_dtype is np.float16:
            max_float16 = np.finfo(np.float16).max
            for col in candidate_features:
                series = self.df[col]
                if series.isnull().all():
                    safe_features.append(col)
                    continue
                max_abs = float(
                    np.nanmax(np.abs(series.to_numpy(dtype=np.float32, copy=False)))
                )
                if max_abs <= max_float16:
                    safe_features.append(col)
        else:
            safe_features = candidate_features

        if not safe_features:
            return

        self.df[safe_features] = self.df[safe_features].astype(target_dtype, copy=False)
        if self._memory_logging_enabled:
            self._log_memory_usage("feature_dtype")

    def _build_fast_access_buffers(self) -> None:
        """Precompute numpy buffers to avoid repeated pandas operations during training."""
        if not self.features:
            self._feature_matrix = np.empty((0, 0), dtype=np.float32)
            self._price_array = None
            self._close_array = None
            self._atr_array = None
            self._episode_id_array = None
            self._nonfinite_rows.clear()
            self._nonfinite_warned_rows.clear()
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
            np.nan_to_num(
                self._feature_matrix,
                copy=False,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if affected_rows.size:
                print(
                    f"⚠️  Feature matrix contained non-finite values. Sanitized {affected_rows.size} rows."
                )
        else:
            self._nonfinite_rows.clear()
        self._nonfinite_warned_rows.clear()
        self._feature_matrix.setflags(write=False)

        self._price_array = self._extract_numeric_column(
            ("price", "close", "adj_close", "open"), fallback=None
        )
        self._close_array = self._extract_numeric_column(
            ("close", "price", "adj_close", "open"), fallback=None
        )
        self._atr_array = self._extract_numeric_column(
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
            self._episode_id_array = self.df[self._episode_id_column].to_numpy(
                copy=False
            )
        else:
            self._episode_id_array = None

    def _extract_numeric_column(
        self, candidates: Iterable[str], *, fallback: Optional[float]
    ) -> Optional[NDArray[np.float32]]:
        """
        Extract a contiguous float32 numpy array for the first available candidate column.

        Args:
            candidates: Ordered iterable of column names to attempt.
            fallback: Value used to replace non-finite entries. If None, values are preserved.

        Returns:
            Contiguous float32 numpy array or None if no candidate column exists.
        """
        for name in candidates:
            if name not in self.df.columns:
                continue
            series = self.df[name]
            if not ptypes.is_numeric_dtype(series):
                continue
            array = np.ascontiguousarray(series.to_numpy(dtype=np.float32, copy=False))
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

    def _ensure_data_available(self, index: int) -> None:
        """必要なインデックスまでデータを拡張"""
        if index < self.n_steps:
            return
        if not self.streaming_pipeline:
            return
        self.streaming_pipeline.prefetch_async()
        attempts = 0
        while index >= self.n_steps:
            if self._append_streaming_rows():
                attempts = 0
                continue
            attempts += 1
            if attempts >= 5:
                break
            time.sleep(0.01)

    def _prime_streaming_data(self) -> None:
        """リセット時にストリーミングデータを確保"""
        if not self.streaming_pipeline:
            return
        self._append_streaming_rows()
        self._ensure_data_available(self.current_step)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        """環境のリセット"""
        super().reset(seed=seed)

        # Check if random start is requested (for evaluation)
        random_start = (
            options and options.get("random_start", False)
        ) or self.random_start

        if random_start:
            # Use random start point for evaluation
            min_start = 0
            max_start = max(0, self.n_steps - 100)  # Leave at least 100 steps
            self.current_step = np.random.randint(min_start, max_start + 1)
        else:
            self.current_step = 0

        # Reset PositionManager and RewardCalculator
        self.position_manager.reset()
        self.reward_calculator.reset()
        
        # Backward compatibility: sync properties
        self.position = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades_count = 0
        self.realized_pnl = 0.0
        
        # Reset reward tracking
        self.reward_history.clear()
        self.position_history.clear()
        self.portfolio_value = self.initial_portfolio_value
        self.pnl_history.clear()
        self.position_abs_history.clear()
        self.trade_interval_history.clear()
        self._last_trade_step = None
        self._consecutive_trade_steps = 0
        self._current_episode_actions.clear()
        self._action_counts = [0, 0, 0]
        self.portfolio_value_history = deque(maxlen=self._max_history_length)  # Memory optimized
        self.action_history.clear()

        # Reset previous portfolio value for step-wise reward calculation
        self._previous_portfolio_value = None

        self._prime_streaming_data()

        # Memory cleanup (reduced frequency for efficiency)
        if self.current_step % 2000 == 0:
            gc.collect()

        return self._get_observation(), self._get_info()

    def _get_reward_setting_int(self, key: str, default: int) -> int:
        """Type-safe getter for integer reward settings."""
        value = self.reward_settings.get(key, default)
        if isinstance(value, (int, float)):
            return int(value)
        return default

    def _get_reward_setting_float(self, key: str, default: float) -> float:
        """Type-safe getter for float reward settings."""
        value = self.reward_settings.get(key, default)
        if isinstance(value, (int, float)):
            return float(value)
        return default

    def _get_reward_setting_bool(self, key: str, default: bool) -> bool:
        """Type-safe getter for boolean reward settings."""
        value = self.reward_settings.get(key, default)
        if isinstance(value, bool):
            return value
        return default

    def _sync_from_position_manager(self) -> None:
        """
        Sync all state from PositionManager to maintain backward compatibility.
        
        Centralizes synchronization logic to prevent bugs like Bug #24 where
        attributes were forgotten during manual syncing.
        
        Note: This method should be called after ANY PositionManager operation
        that modifies state (execute_action, close_position, etc.).
        """
        pos_info = self.position_manager.get_position_info()
        self.position = pos_info["position"]
        self.entry_price = pos_info["entry_price"]
        self.realized_pnl = pos_info["realized_pnl"]
        self.total_pnl = pos_info["total_pnl"]
        self.trades_count = pos_info["trades_count"]
        # Note: consecutive_trade_steps is internal state, not exposed in get_position_info

    def get_legal_actions(self) -> NDArray[np.int_]:
        """現在の状態で合法なアクションを返す（1=合法, 0=非法）"""
        legal = np.zeros(3, dtype=np.int_)  # [HOLD, BUY, SELL] - デフォルト非法

        current_price = self._resolve_price()
        portfolio_value = self.initial_portfolio_value + self.total_pnl
        position_size = self.config.max_position_size
        transaction_cost = self.config.transaction_cost

        # HOLDは常に合法
        legal[0] = 1

        # 取引所別取引頻度制限（Coincheckは手数料無料なので制限緩和）
        exchange = getattr(self.config, "exchange", "coincheck").lower()
        if exchange != "coincheck":
            # Coincheck以外は取引頻度制限を適用
            # 最小ホールド期間チェック
            min_holding_period = getattr(self.config, "min_holding_period", 3)
            if hasattr(self, "_last_trade_step") and self._last_trade_step is not None:
                steps_since_last_trade = self.current_step - self._last_trade_step
                if steps_since_last_trade < min_holding_period:
                    # 最小ホールド期間中でも、ポジションクローズは許可（リスク管理上重要）
                    if self.position > 0:
                        # ロングポジション保有中: SELLでクローズ可能
                        legal[2] = 1
                    elif self.position < 0:
                        # ショートポジション保有中: BUYでクローズ可能
                        legal[1] = 1
                    # その他の新規建ては制限
                    return legal

            # 連続取引制限チェック
            max_consecutive_trades = getattr(self.config, "max_consecutive_trades", 5)
            if self._consecutive_trade_steps >= max_consecutive_trades:
                # 連続取引上限に達した場合もポジションクローズは許可
                if self.position > 0:
                    legal[2] = 1  # SELL to close long
                elif self.position < 0:
                    legal[1] = 1  # BUY to close short
                return legal

        # 市場ボラティリティチェック（高ボラティリティ時は取引制限）
        volatility_threshold = getattr(
            self.config, "volatility_trade_threshold", 0.02
        )  # 2%ボラティリティ閾値
        if self.current_step > 20:
            price_slice: Optional[np.ndarray] = None
            start_idx = max(0, self.current_step - 20)
            end_idx = self.current_step

            if (
                self._close_array is not None
                and self._close_array.size >= end_idx
                and end_idx > start_idx
            ):
                price_slice = self._close_array[start_idx:end_idx]
            elif (
                self._price_array is not None
                and self._price_array.size >= end_idx
                and end_idx > start_idx
            ):
                price_slice = self._price_array[start_idx:end_idx]
            elif hasattr(self, "df"):
                try:
                    recent_prices = self.df.iloc[start_idx:end_idx]["close"]
                    if len(recent_prices) > 1:
                        price_slice = recent_prices.to_numpy(dtype=np.float32, copy=False)
                except Exception:
                    price_slice = None

            if price_slice is not None and price_slice.size > 1:
                finite_mask = np.isfinite(price_slice)
                if np.count_nonzero(finite_mask) > 1:
                    valid_prices = price_slice[finite_mask]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        denominators = valid_prices[:-1]
                        returns = np.diff(valid_prices) / np.where(
                            denominators == 0.0, np.nan, denominators
                        )
                    returns = returns[np.isfinite(returns)]
                    if returns.size:
                        current_volatility = float(np.std(returns))
                        if current_volatility > volatility_threshold:
                            return legal

        # BUY: ショートまたはフラットの場合、かつ十分な残高がある場合
        if self.position <= 0:
            buy_cost = position_size * current_price * (1 + transaction_cost)
            if portfolio_value >= buy_cost:
                legal[1] = 1

        # SELL: ロングまたはフラットの場合、かつショートポジションがある場合
        if self.position >= 0:
            # ショートポジションを開く場合、ポジションサイズ分の価値が必要
            sell_value = position_size * current_price
            if portfolio_value >= sell_value:
                legal[2] = 1

        # Safety check: ensure at least one action is legal (HOLD should always be legal)
        if not np.any(legal):
            # This should never happen since HOLD is always legal, but add safety
            legal[0] = 1  # Force HOLD to be legal

        return legal

    def action_mask(self) -> NDArray[np.bool_]:
        """Return action mask for gymnasium ActionMasker wrapper."""
        return self.get_legal_actions().astype(np.bool_)

    def get_action_masks(self) -> NDArray[np.bool_]:
        """Return action masks for SB3 MaskablePPO."""
        return self.action_mask()

    def step(self, action: int) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        """ステップ実行"""
        # Execute action using PositionManager with min_holding_period constraint
        old_position = self.position_manager.position
        min_holding_period = getattr(self.config, "min_holding_period", 0)
        trade_pnl = self.position_manager.execute_action(action, self.current_step, min_holding_period)
        
        # Sync backward compatibility properties using centralized method
        self._sync_from_position_manager()

        # ストップロスチェック（損失が閾値を超えたらポジション強制クローズ）
        stop_loss_threshold = self.config.stop_loss_threshold
        if self.position != 0 and self.entry_price > 0:
            current_price = self._resolve_price()
            if self.position > 0:  # ロングポジション
                loss_ratio = (self.entry_price - current_price) / self.entry_price
                if loss_ratio > stop_loss_threshold:
                    # Close position and capture realized PnL for reward
                    forced_close_pnl = self.position_manager.close_position(self.current_step)
                    trade_pnl += forced_close_pnl  # Add forced close PnL to action PnL
                    # Sync ALL properties using centralized method (Bug #24 fix)
                    self._sync_from_position_manager()
            elif self.position < 0:  # ショートポジション
                loss_ratio = (current_price - self.entry_price) / self.entry_price
                if loss_ratio > stop_loss_threshold:
                    # Close position and capture realized PnL for reward
                    forced_close_pnl = self.position_manager.close_position(self.current_step)
                    trade_pnl += forced_close_pnl  # Add forced close PnL to action PnL
                    # Sync ALL properties using centralized method (Bug #24 fix)
                    self._sync_from_position_manager()

        # Track action for forced diversity
        self._current_episode_actions.append(action)

        # Track rolling action history for diversity-focused rewards
        self.action_history.append(action)
        # Note: deque with maxlen automatically removes old items, no manual deletion needed

        # Calculate unrealized PnL using PositionManager
        unrealized_pnl = self.position_manager.calculate_unrealized_pnl()

        # Portfolio value = initial + realized + unrealized
        portfolio_value = self.initial_portfolio_value + self.realized_pnl + unrealized_pnl
        self.portfolio_value = portfolio_value
        
        # pnl for reward calculation (trade PnL from this specific action, not total unrealized)
        pnl = trade_pnl

        current_price = self._resolve_price()
        atr = self._resolve_atr()

        # Calculate reward using RewardCalculator
        reward = self.reward_calculator.calculate_reward(
            action=action,
            current_price=current_price,
            position=self.position,
            portfolio_value=portfolio_value,
            atr=atr,
            transaction_cost=self.config.transaction_cost,
            reward_scaling=self.config.reward_scaling,
            pnl=pnl,
            old_position=old_position,
            step=self.current_step,
            observation=self._get_observation(),
            reward_history=list(self.reward_history),  # Convert deque to list
            portfolio_value_history=list(self.portfolio_value_history),  # Convert deque to list
        )

        # 次のステップへ
        self.current_step += 1
        self._ensure_data_available(self.current_step)

        # エピソード終了判定
        done = self.current_step >= self.n_steps - 1
        if (
            not done
            and self._episode_id_array is not None
            and self.current_step < len(self._episode_id_array)
        ):
            prev_idx = max(self.current_step - 1, 0)
            if prev_idx < len(self._episode_id_array):
                current_episode = self._episode_id_array[prev_idx]
                next_episode = self._episode_id_array[self.current_step]
                if current_episode != next_episode:
                    done = True

        # 次の状態
        next_obs = self._get_observation()

        # 情報
        info = self._get_info()
        position_utilisation = abs(self.position) / max(
            1e-8, self.config.max_position_size
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
                "action_masks": self.get_legal_actions().astype(
                    bool
                ),  # Add action masks for MaskablePPO
            }
        )

        # 過去履歴の更新
        self.pnl_history.append(pnl)
        self.position_abs_history.append(abs(self.position))
        self.portfolio_value_history.append(
            portfolio_value
        )  # Track portfolio value for stagnation penalty

        # 報酬履歴の更新（dequeのmaxlenで自動制限）
        self.reward_history.append(reward)
        self.position_history.append(self.position)

        # Note: All histories use deque with maxlen for automatic size limiting
        # No manual pop(0) needed anymore

        if self._memory_logging_enabled and self._memory_log_interval_steps:
            if self.current_step % self._memory_log_interval_steps == 0 and (
                self.current_step != self._last_memory_log_step
            ):
                self._log_memory_usage(f"step_{self.current_step}")
                self._last_memory_log_step = self.current_step

        # Aggressive garbage collection for memory-constrained environments
        if self._gc_step_interval and self.current_step % self._gc_step_interval == 0:
            gc.collect(generation=0)  # Quick generation 0 collection
        
        # Full collection every 1000 steps (reduced frequency for memory efficiency)
        if self.current_step % 1000 == 0:
            gc.collect()  # Full collection

        return next_obs, reward, done, False, info

    def _resolve_price(self, step: Optional[int] = None) -> float:
        step = (
            self.current_step if step is None else max(0, min(step, self.n_steps - 1))
        )
        if self._price_array is not None and self._price_array.size:
            idx = min(step, self._price_array.shape[0] - 1)
            value = float(self._price_array[idx])
            if np.isfinite(value):
                return value
        try:
            row = self.df.iloc[step]
        except (IndexError, KeyError):
            return 0.0

        for column in ("price", "close", "adj_close", "open"):
            if column in row.index:
                value = row[column]
                if pd.notna(value):
                    return float(value)
        numeric_candidates = [
            v for v in row.values if isinstance(v, (int, float, np.floating))
        ]
        if numeric_candidates:
            return float(numeric_candidates[0])
        return 0.0

    def _resolve_atr(self, step: Optional[int] = None, default: float = 1.0) -> float:
        step = (
            self.current_step if step is None else max(0, min(step, self.n_steps - 1))
        )
        if self._atr_array is not None and self._atr_array.size:
            idx = min(step, self._atr_array.shape[0] - 1)
            value = float(self._atr_array[idx])
            if np.isfinite(value) and value > 0:
                return value
        if step >= len(self.df):
            return default
        row = self.df.iloc[step]
        for column in (
            "atr_10",
            "atr_14",
            "atr_simplified",
            "ATR",
            "ATR_simplified",
        ):
            if column in row.index:
                value = row[column]
                if pd.notna(value) and value > 0:
                    return float(value)
        return default

    def _get_observation(self) -> NDArray[np.float32]:
        """現在の状態を取得"""
        self._ensure_data_available(self.current_step)

        if self._feature_matrix.size:
            max_index = self._feature_matrix.shape[0] - 1
            index = min(max(self.current_step, 0), max_index)
            obs = self._feature_matrix[index]

            if (
                self._nonfinite_rows
                and index in self._nonfinite_rows
                and index not in self._nonfinite_warned_rows
            ):
                if len(self._nonfinite_warned_rows) < 5 or index % 1000 == 0:
                    print(
                        f"Warning: Step {index} had non-finite feature values. Replaced with zeros for stability."
                    )
                self._nonfinite_warned_rows.add(index)

            return obs

        # Fallback path (should rarely execute) - preserve previous behaviour
        if self.current_step >= self.n_steps:
            step_data = self.df.iloc[-1]
        else:
            step_data = self.df.iloc[self.current_step]

        feature_list = list(self.features)
        obs = step_data[feature_list].to_numpy(dtype=np.float32, copy=False)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_info(self) -> Dict[str, Any]:
        """追加情報を取得"""
        return {
            "current_step": self.current_step,
            "total_steps": self.n_steps,
            "position": self.position,
            "total_pnl": self.total_pnl,
            "trades_count": self.trades_count,
            "features": self.features,
            "config": self.config,
            "pnl": self.total_pnl,
        }

    def render(self, mode: str = "human") -> None:
        """環境の描画"""
        if mode == "human":
            print(f"Step: {self.current_step}/{self.n_steps}")
            print(f"Position: {self.position}")
            print(f"Total PnL: {self.total_pnl:.4f}")
            print(f"Trades: {self.trades_count}")
            if len(self.reward_history) > 0:
                print(f"Last Reward: {self.reward_history[-1]:.6f}")
            print("-" * 40)

    def close(self) -> None:
        """環境のクリーンアップ（メモリ最適化）"""
        # Clear all histories
        self.reward_history.clear()
        self.position_history.clear()
        self.portfolio_value_history.clear()
        self.action_history.clear()
        
        # Clear deques
        self.position_abs_history.clear()
        self.pnl_history.clear()
        self.trade_interval_history.clear()
        
        # Clear DataFrame
        if hasattr(self, 'df'):
            self.df = pd.DataFrame()
        
        # Clear large objects
        self._current_episode_actions.clear()
        self._action_counts = [0, 0, 0]
        
        # Force garbage collection
        gc.collect()

    # ユーティリティメソッド
    def get_feature_names(self) -> list[str]:
        """特徴量名を取得"""
        return self.features

    def get_statistics(self) -> StatisticsDict:
        """統計情報を取得"""
        if len(self.reward_history) == 0:
            return {}

        rewards = np.array(self.reward_history)

        return {
            "total_reward": float(np.sum(rewards)),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "sharpe_ratio": float(np.mean(rewards) / (np.std(rewards) + EPSILON)),
            "max_reward": float(np.max(rewards)),
            "total_trades": self.trades_count,
            "win_rate": float(np.sum(rewards > 0) / len(rewards)) if len(rewards) > 0 else 0.0,
        }

    def get_trades_per_1k(self) -> float:
        """1000ステップあたりの取引回数を取得"""
        if self.current_step == 0:
            return 0.0
        return self.trades_count / (self.current_step / 1000)

    def get_last_actions(self) -> List[int]:
        """Get the actions taken in the last episode for action distribution analysis."""
        # Return actions from the current episode if available
        if hasattr(self, "_current_episode_actions"):
            return self._current_episode_actions.copy()
        return []


class FlipHeavyTradingEnv(HeavyTradingEnv):
    """
    Flipped version of HeavyTradingEnv for symmetry testing.
    Flips signs of observations, actions, and positions to test reward symmetry.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_observation(self) -> NDArray[np.float32]:
        """Get flipped observation (signs inverted for key features)."""
        obs = super()._get_observation()

        # Flip signs of key features that should be symmetric
        flip_indices = []
        for i, feature in enumerate(self.features):
            # Flip price-related and position-related features
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

    def step(self, action: int) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        """Execute flipped action and return flipped results."""
        # Flip action: BUY<->SELL, HOLD remains
        flipped_action = action
        if action == ACTION_BUY:
            flipped_action = ACTION_SELL
        elif action == ACTION_SELL:
            flipped_action = ACTION_BUY

        # Execute flipped action
        obs, reward, done, truncated, info = super().step(flipped_action)

        # Flip position in info
        if "position" in info:
            info["position"] = -info["position"]

        # Flip PnL (since position is flipped, PnL should be flipped)
        if "pnl" in info:
            info["pnl"] = -info["pnl"]

        return obs, reward, done, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Return flipped info."""
        info = super()._get_info()
        if "position" in info:
            info["position"] = -info["position"]
        return info


# Register environments with gymnasium
gym.register(
    id="HeavyTradingEnv",
    entry_point="ztb.trading.environment:HeavyTradingEnv",
    kwargs={},
)