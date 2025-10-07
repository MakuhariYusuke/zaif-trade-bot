# Heavy Trading Environment for Reinforcement Learning
# 重特徴量ベースの取引環境

import dataclasses
import gc
import math
import time
from collections import deque
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
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
from ztb.utils.fee_model import ExchangeFeeModel
from ztb.utils.memory.dtypes import optimize_dtypes
from ztb.types.protocols import TradingEnvironment
from ztb.training.ppo_config import (
    DEFAULT_REWARD_SCALING, DEFAULT_RISK_FREE_RATE, DEFAULT_STOP_LOSS_THRESHOLD,
    DEFAULT_MAX_CONSECUTIVE_TRADES, DEFAULT_MIN_HOLDING_PERIOD,
    DEFAULT_REWARD_POSITION_SOFT_CAP, DEFAULT_REWARD_POSITION_PENALTY_SCALE,
    DEFAULT_REWARD_POSITION_PENALTY_EXPONENT, DEFAULT_REWARD_INVENTORY_WINDOW,
    DEFAULT_REWARD_INVENTORY_PENALTY_SCALE, DEFAULT_REWARD_TRADE_FREQUENCY_PENALTY,
    DEFAULT_REWARD_TRADE_FREQUENCY_HALFLIFE, DEFAULT_REWARD_TRADE_COOLDOWN_STEPS,
    DEFAULT_REWARD_TRADE_COOLDOWN_PENALTY, DEFAULT_REWARD_MAX_CONSECUTIVE_TRADES,
    DEFAULT_REWARD_CONSECUTIVE_TRADE_PENALTY, DEFAULT_REWARD_VOLATILITY_WINDOW,
    DEFAULT_REWARD_VOLATILITY_PENALTY_SCALE, DEFAULT_REWARD_SHARPE_BONUS_SCALE,
    DEFAULT_REWARD_CLIP_VALUE
)

# Type aliases for better type safety
Observation = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
Action = int
Reward = float
Info = Dict[str, Any]


class RewardSettings(TypedDict, total=False):
    """Type-safe reward settings configuration."""

    position_soft_cap: float
    position_penalty_scale: float
    position_penalty_exp: float
    inventory_window: int
    inventory_penalty_scale: float
    trade_frequency_penalty: float
    trade_frequency_halflife: float
    trade_cooldown_steps: int
    trade_cooldown_penalty: float
    max_consecutive_trades: int
    consecutive_trade_penalty: float
    volatility_window: int
    volatility_penalty_scale: float
    sharpe_bonus_scale: float
    sortino_bonus_scale: float
    calmar_bonus_scale: float
    reward_clip_value: float
    profit_bonus_multipliers: List[float]
    enable_forced_diversity: bool
    custom_reward_params: Dict[str, float]


# Configuration dataclass to replace Dict[str, Any]
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
    initial_portfolio_value: float = 1_000_000.0
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
    
    # Trading behavior settings
    allow_reverse: bool = True  # If False, SELL from Long/BUY from Short only closes position (no immediate reverse)

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

        # Convert dictionary to config, handling type conversions
        config_kwargs = {}
        for field in dataclasses.fields(cls):
            if field.name in config_dict:
                value = config_dict[field.name]
                # Basic type conversion for common cases
                if field.name in ["enable_forced_diversity", "allow_reverse"] and not isinstance(
                    value, bool
                ):
                    value = cls._as_bool(value)  # type: ignore[arg-type]
                elif field.name in [
                    "max_consecutive_trades",
                    "min_holding_period",
                    "reward_inventory_window",
                    "reward_trade_cooldown_steps",
                    "reward_max_consecutive_trades",
                    "reward_volatility_window",
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
                ] and isinstance(value, str):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        pass  # Keep original value
                config_kwargs[field.name] = value
            # Field will use default if not in config_dict

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
    portfolio_value_history: List[float]
    action_history: List[int]

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
            self.reward_settings = cast(
                RewardSettings, {**self.reward_settings, **self.config.reward_settings}
            )

        # 取引コストを動的に設定（取引所に基づく）
        self.fee_model = ExchangeFeeModel()
        self.fee_model.set_exchange(self.config.exchange)
        self.config.transaction_cost = self.fee_model.get_fee_rate(
            "buy"
        )  # デフォルトで買いの手数料を使用

        self.initial_portfolio_value = float(self.config.initial_portfolio_value)
        self.portfolio_value = self.initial_portfolio_value

        inventory_window = max(8, self._get_reward_setting_int("inventory_window", 128))
        volatility_window = max(
            8, self._get_reward_setting_int("volatility_window", 32)
        )
        self.position_abs_history: deque[float] = deque(maxlen=inventory_window)
        self.pnl_history: deque[float] = deque(maxlen=volatility_window)
        self.trade_interval_history: deque[int] = deque(maxlen=64)
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

        self._memory_logging_enabled = EnvironmentConfig._as_bool(
            getattr(self.config, "memory_logging_enabled", False)
        )

        memory_interval_cfg = getattr(self.config, "memory_log_interval_steps", None)
        try:
            self._memory_log_interval_steps = max(1, int(memory_interval_cfg or 1000))
        except (TypeError, ValueError):
            self._memory_log_interval_steps = 1000

        gc_interval_cfg = getattr(self.config, "gc_collect_interval_steps", None)
        try:
            self._gc_step_interval = max(0, int(gc_interval_cfg or 0))
        except (TypeError, ValueError):
            self._gc_step_interval = 0

        preprocess_chunk_cfg = getattr(self.config, "preprocess_chunk_size", None)
        try:
            self._preprocess_chunk_size = max(1, int(preprocess_chunk_cfg or 32))
        except (TypeError, ValueError):
            self._preprocess_chunk_size = 32

        self._last_memory_log_step = 0
        self.random_start = random_start

        self.streaming_pipeline = streaming_pipeline
        self.stream_batch_size = max(1, int(stream_batch_size))
        self.stream_to_bars_converter = stream_to_bars_converter
        self._stream_last_timestamp: Optional[pd.Timestamp] = None
        self._stream_rows_appended = 0
        self._base_columns: List[str] = []

        base_df = df.copy() if df is not None else None
        if base_df is None:
            base_df = self._fetch_streaming_snapshot(
                required_rows=self.stream_batch_size
            )
            if base_df.empty:
                raise ValueError("Streaming pipeline did not provide initial data")

        # データの前処理
        self.df = self._preprocess_data(base_df)
        del base_df
        gc.collect()
        self._log_memory_usage("post_init")

        # 積極的なメモリ最適化
        if hasattr(self, "_memory_logging_enabled") and self._memory_logging_enabled:
            # DataFrameの断片化を防ぐ
            self.df = self.df.copy()
            # インデックスを最適化
            if not self.df.index.is_monotonic_increasing:
                self.df = self.df.sort_index()

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

        # 特徴量数を制限
        if max_features is not None and len(self.features) > max_features:
            self.features = self.features[:max_features]
            print(f"Limited features to {max_features}: {self.features}")

        self._apply_feature_storage_dtype()

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
        self.position = 0.0  # -1, 0, 1
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.realized_pnl = 0.0  # Realized PnL (accumulated on position close)
        self.trades_count = 0

        # ストリーミング関連
        self._timestamp_column = "timestamp" if "timestamp" in self.df.columns else None
        self._episode_id_column = (
            "episode_id" if "episode_id" in self.df.columns else None
        )
        if not self._timestamp_column:
            self._stream_rows_appended = len(self.df)

        # 報酬計算用の履歴
        self.reward_history: list[float] = []
        self.position_history: list[float] = []
        self._action_counts: list[int] = [
            0,
            0,
            0,
        ]  # Track action usage for balance bonus
        self._current_episode_actions: list[int] = []
        self.action_history: list[int] = []
        action_history_limit = getattr(self.config, "max_action_history", None)
        try:
            self._max_action_history = max(10, int(action_history_limit or 512))
        except (TypeError, ValueError):
            self._max_action_history = 512

        # Limit history lengths to prevent memory leaks during long training
        self._max_history_length = getattr(self.config, "max_history_length", 1000)

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データの前処理とメモリ最適化"""
        if df.empty:
            return df.copy()

        df_processed = df.fillna(0)
        if not df_processed.index.equals(
            pd.RangeIndex(start=0, stop=len(df_processed), step=1)
        ):
            df_processed = df_processed.reset_index(drop=True)
        else:
            df_processed = df_processed.copy()

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

        self.position = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades_count = 0
        self.reward_history = []
        self.position_history = []
        self.portfolio_value = self.initial_portfolio_value
        self.pnl_history.clear()
        self.position_abs_history.clear()
        self.trade_interval_history.clear()
        self._last_trade_step = None
        self._consecutive_trade_steps = 0
        self._current_episode_actions.clear()  # Reset action tracking for forced diversity
        self._action_counts = [
            0,
            0,
            0,
        ]  # Reset action counts for forced diversity per episode
        self.portfolio_value_history = (
            []
        )  # Initialize portfolio value history for stagnation penalty
        self.action_history.clear()

        # Reset previous portfolio value for step-wise reward calculation
        self._previous_portfolio_value = None

        self._prime_streaming_data()

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
                    # 最小ホールド期間が経過していない場合、BUY/SELLを非法に
                    return legal

            # 連続取引制限チェック
            max_consecutive_trades = getattr(self.config, "max_consecutive_trades", 5)
            if self._consecutive_trade_steps >= max_consecutive_trades:
                # 連続取引上限に達したら、BUY/SELLを非法に
                return legal

        # 市場ボラティリティチェック（高ボラティリティ時は取引制限）
        volatility_threshold = getattr(
            self.config, "volatility_trade_threshold", 0.02
        )  # 2%ボラティリティ閾値
        if hasattr(self, "df") and self.current_step > 20:
            try:
                # 直近20期間のボラティリティを計算
                recent_prices = self.df.iloc[
                    max(0, self.current_step - 20) : self.current_step
                ]["close"]
                if len(recent_prices) > 1:
                    returns = recent_prices.pct_change().dropna()
                    current_volatility = returns.std()
                    if current_volatility > volatility_threshold:
                        # 高ボラティリティ時はBUY/SELLを非法に
                        return legal
            except:
                pass  # ボラティリティ計算失敗時は無視

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

        return legal

    def action_mask(self) -> NDArray[np.bool_]:
        """Return action mask for gymnasium ActionMasker wrapper."""
        return self.get_legal_actions().astype(np.bool_)

    def get_action_masks(self) -> NDArray[np.bool_]:
        """Return action masks for SB3 MaskablePPO."""
        return self.action_mask()

    def step(self, action: int) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        """ステップ実行"""
        # 行動の実行
        old_position = self.position
        self._execute_action(action)

        # ストップロスチェック（損失が閾値を超えたらポジション強制クローズ）
        stop_loss_threshold = self.config.stop_loss_threshold
        if self.position != 0 and self.entry_price > 0:
            current_price = self._resolve_price()
            if self.position > 0:  # ロングポジション
                loss_ratio = (self.entry_price - current_price) / self.entry_price
                if loss_ratio > stop_loss_threshold:
                    self._close_position()
            elif self.position < 0:  # ショートポジション
                loss_ratio = (current_price - self.entry_price) / self.entry_price
                if loss_ratio > stop_loss_threshold:
                    self._close_position()

        # Track action for forced diversity
        self._current_episode_actions.append(action)

        # Track rolling action history for diversity-focused rewards
        self.action_history.append(action)
        max_history = getattr(self, "_max_action_history", 512)
        if len(self.action_history) > max_history:
            del self.action_history[:-max_history]

        # Unrealized PnL calculation (for open positions)
        unrealized_pnl = self._calculate_pnl()

        # Portfolio value = initial + realized + unrealized
        # Note: total_pnl is kept as realized_pnl for backward compatibility
        portfolio_value = self.initial_portfolio_value + self.realized_pnl + unrealized_pnl
        self.portfolio_value = portfolio_value
        
        # pnl for reward calculation (unrealized PnL)
        pnl = unrealized_pnl

        current_price = self._resolve_price()
        atr = self._resolve_atr()

        reward = self._calculate_reward(
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
        )

        # 次のステップへ
        self.current_step += 1
        self._ensure_data_available(self.current_step)

        # エピソード終了判定
        done = self.current_step >= self.n_steps - 1
        if not done and self._episode_id_column and self.current_step < self.n_steps:
            current_episode = self.df.iloc[self.current_step - 1][
                self._episode_id_column
            ]
            next_episode = self.df.iloc[self.current_step][self._episode_id_column]
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

        # 報酬履歴の更新（メモリリーク防止のため長さを制限）
        self.reward_history.append(reward)
        self.position_history.append(self.position)

        # Limit history lengths to prevent memory leaks
        if len(self.reward_history) > self._max_history_length:
            self.reward_history.pop(0)
        if len(self.position_history) > self._max_history_length:
            self.position_history.pop(0)
        if len(self.portfolio_value_history) > self._max_history_length:
            self.portfolio_value_history.pop(0)

        if self._memory_logging_enabled and self._memory_log_interval_steps:
            if self.current_step % self._memory_log_interval_steps == 0 and (
                self.current_step != self._last_memory_log_step
            ):
                self._log_memory_usage(f"step_{self.current_step}")
                self._last_memory_log_step = self.current_step

        if self._gc_step_interval and self.current_step % self._gc_step_interval == 0:
            gc.collect()

        return next_obs, reward, done, False, info

    def _resolve_price(self, step: Optional[int] = None) -> float:
        step = (
            self.current_step if step is None else max(0, min(step, self.n_steps - 1))
        )
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
        if step >= len(self.df):
            return default
        row = self.df.iloc[step]
        if isinstance(row, pd.Series):
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

    def _execute_action(self, action: int) -> None:
        """行動の実行
        
        allow_reverse=True (default): 従来通りの動作
            - Long中のSELL: Close→即Short
            - Short中のBUY: Close→即Long
        
        allow_reverse=False: 反転禁止モード
            - Long中のSELL: Close only (Flatに戻る)
            - Short中のBUY: Close only (Flatに戻る)
            - Flat状態からのSELL/BUYは従来通り開く
        """
        if action == 0:  # HOLD
            # HOLDの場合は連続取引カウンターをリセット
            self._consecutive_trade_steps = 0
            pass  # ポジション維持
        elif action == 1:  # BUY
            if self.position < 0:  # ショートポジション保有中
                # ショートクローズ
                self._close_position()
                self._consecutive_trade_steps += 1
                
                # allow_reverse=Trueの場合のみ、即座にロングを開く
                if self.config.allow_reverse:
                    self._open_position(1)
            elif self.position == 0:  # フラット状態
                # 通常のロングオープン
                self._open_position(1)
                self._consecutive_trade_steps += 1
            # position > 0 (既にロング)の場合は何もしない
        elif action == 2:  # SELL
            if self.position > 0:  # ロングポジション保有中
                # ロングクローズ
                self._close_position()
                self._consecutive_trade_steps += 1
                
                # allow_reverse=Trueの場合のみ、即座にショートを開く
                if self.config.allow_reverse:
                    self._open_position(-1)
            elif self.position == 0:  # フラット状態
                # 通常のショートオープン
                self._open_position(-1)
                self._consecutive_trade_steps += 1
            # position < 0 (既にショート)の場合は何もしない

    def _open_position(self, direction: int) -> None:
        """ポジションオープン（エントリーコストを即座に反映）"""
        current_price = self._resolve_price()
        position_size = getattr(self.config, "max_position_size", 1.0)
        
        # Calculate entry cost
        entry_cost = abs(position_size) * current_price * self.config.transaction_cost
        
        # Deduct entry cost from realized PnL
        self.realized_pnl -= entry_cost
        self.total_pnl = self.realized_pnl  # Keep total_pnl in sync
        
        # Open position
        self.position = direction * position_size
        self.entry_price = current_price
        self.trades_count += 1
        self._last_trade_step = self.current_step

    def _close_position(self) -> None:
        """ポジションクローズ"""
        if self.position != 0:
            # Calculate realized PnL before closing position
            current_price = self._resolve_price()
            price_change = current_price - self.entry_price
            realized_trade_pnl = float(self.position) * price_change
            
            # Deduct transaction cost (exit cost)
            exit_cost = abs(self.position) * current_price * self.config.transaction_cost
            realized_trade_pnl -= exit_cost
            
            # Accumulate realized PnL
            self.realized_pnl += realized_trade_pnl
            self.total_pnl = self.realized_pnl  # Keep total_pnl in sync for backward compatibility
            
            self.trades_count += 1
            self.position = 0.0
            self.entry_price = 0.0

    def _calculate_pnl(self) -> float:
        """Calculate unrealized PnL for current open position.
        
        Realized PnL is accumulated in _close_position().
        This method only returns unrealized PnL for reward calculation.
        """
        if self.position == 0:
            return 0.0

        current_price = self._resolve_price()
        entry_price = self.entry_price
        price_change = current_price - entry_price

        # Unrealized PnL (price change only, no exit cost yet)
        unrealized_pnl = float(self.position) * price_change

        return float(unrealized_pnl)

    def _calculate_reward(
        self,
        action: int,
        current_price: float,
        position: float,
        portfolio_value: float,
        atr: float,
        transaction_cost: float,
        reward_scaling: float,
        pnl: float,
        old_position: float,
        step: int,
        observation: Optional[Any],
    ) -> float:
        """Calculate reward with curriculum learning stages."""
        curriculum_stage = self.config.curriculum_stage
        print(
            f"DEBUG Curriculum stage: {curriculum_stage}, position: {position}, action: {action}"
        )

        eps = 1e-8
        atr = atr if atr > eps else 1.0
        max_position_size = max(eps, self.config.max_position_size)

        atr_normalised = pnl / atr
        portfolio_return = pnl / max(abs(self.initial_portfolio_value), eps)

        # Initialize balance_bonus for all curriculum stages
        balance_bonus = 0.0

        # Curriculum learning stages
        if curriculum_stage == "forced_balance":
            # Stage 0: Force balanced action distribution (33% each action)
            # Track action counts and reward only when actions are balanced
            action_counts = getattr(self, "_action_counts", [0, 0, 0])
            action_counts[action] += 1
            self._action_counts = action_counts

            total_actions = sum(action_counts)
            if total_actions >= 3:  # Need at least 3 actions to evaluate balance
                action_ratios = [count / total_actions for count in action_counts]
                target_ratio = 1.0 / 3.0  # 33.33% each

                # Calculate balance score (lower is better balance)
                balance_penalty = sum(
                    abs(ratio - target_ratio) for ratio in action_ratios
                )

                # Reward for good balance, penalty for imbalance
                if balance_penalty < 0.1:  # Very balanced (within 10% of target)
                    reward = 2.0
                elif balance_penalty < 0.2:  # Moderately balanced
                    reward = 1.0
                elif balance_penalty < 0.3:  # Somewhat balanced
                    reward = 0.5
                else:  # Imbalanced
                    reward = -1.0
            else:
                # Early stage - small reward for any action to encourage exploration
                reward = 0.1

            return float(reward)
        elif curriculum_stage == "balanced_transition":
            # Stage 1: Normal reward function but with balance penalty to maintain diversity
            # Use standard reward calculation but add penalty for imbalanced actions
            action_counts = getattr(self, "_action_counts", [0, 0, 0])
            action_counts[action] += 1
            self._action_counts = action_counts

            total_actions = sum(action_counts)
            balance_penalty = 0.0

            if total_actions >= 10:  # Require minimum actions before enforcing balance
                action_ratios = [count / total_actions for count in action_counts]
                target_ratio = 1.0 / 3.0

                # Calculate balance deviation
                balance_penalty = sum(
                    abs(ratio - target_ratio) for ratio in action_ratios
                )

                # Strong penalty for very imbalanced distributions
                if (
                    balance_penalty > 0.5
                ):  # More than 50% deviation from perfect balance
                    balance_penalty *= 2.0  # Double the penalty

                # Debug output
                print(
                    f"DEBUG Balance: total_actions={total_actions}, ratios={action_ratios}, penalty={balance_penalty:.3f}"
                )

            # Calculate base reward using standard logic (from "full" stage)
            base_component = 0.0

            # Profit bonus for profitable trades (significantly increased for scalping)
            base_profit_bonus = (
                max(0.0, 1.5 * atr_normalised + 1.2 * portfolio_return)
                if pnl > 0
                else 0.0
            )

            # Trend-aware adjustment based on SMA_20/SMA_50 ratio
            trend_multiplier = 1.0
            if observation is not None and len(observation) > 5:
                # Get raw SMA values from dataframe instead of normalized observation
                if self.current_step >= self.n_steps:
                    step_data = self.df.iloc[-1]
                else:
                    step_data = self.df.iloc[self.current_step]

                sma_20 = step_data.get("sma_short", 0.0)
                sma_50 = step_data.get("sma_long", 0.0)
                rsi = step_data.get(
                    "rsi", 50.0
                )  # Add RSI for downward signal enhancement

                print(
                    f"DEBUG Trend: sma_20={sma_20:.4f}, sma_50={sma_50:.4f}, rsi={rsi:.4f}"
                )
                if sma_50 > eps:
                    trend_ratio = sma_20 / sma_50
                    print(f"DEBUG Trend: trend_ratio={trend_ratio:.4f}")
                    if (
                        trend_ratio > 1.02
                    ):  # Strong bullish trend (strengthened threshold)
                        if action == 1:  # BUY
                            trend_multiplier = 1.5  # Enhanced bonus
                        elif action == 2:  # SELL
                            trend_multiplier = 0.5  # Enhanced penalty
                    elif (
                        trend_ratio < 0.98
                    ):  # Strong bearish trend (strengthened threshold)
                        if action == 1:  # BUY
                            trend_multiplier = 0.5  # Enhanced penalty
                        elif action == 2:  # SELL
                            trend_multiplier = 1.5  # Enhanced bonus
                    elif trend_ratio > 1.005:  # Mild bullish trend
                        if action == 1:  # BUY
                            trend_multiplier = 1.1
                        elif action == 2:  # SELL
                            trend_multiplier = 0.9
                    elif trend_ratio < 0.995:  # Mild bearish trend
                        if action == 1:  # BUY
                            trend_multiplier = 0.9
                        elif action == 2:  # SELL
                            trend_multiplier = 1.1
                    # HOLD gets neutral multiplier

                    # Enhanced downward signal: RSI oversold (< 40) boosts SELL in bearish trends
                    if rsi < 40.0 and trend_ratio < 1.0:  # Relaxed downward signal
                        if action == 2:  # SELL
                            trend_multiplier *= 1.3  # Additional boost for SELL in oversold bearish conditions
                        elif action == 1:  # BUY
                            trend_multiplier *= (
                                0.7  # Additional penalty for BUY in oversold conditions
                            )

            # Balance BUY/SELL actions to encourage balanced trading (configurable multipliers)
            multipliers = self.reward_settings.get(
                "profit_bonus_multipliers", [1.0, 1.0, 0.8]
            )
            if action == 1:  # BUY action
                profit_bonus = base_profit_bonus * multipliers[0] * trend_multiplier
            elif action == 2:  # SELL action
                profit_bonus = base_profit_bonus * multipliers[1] * trend_multiplier
            else:  # HOLD
                profit_bonus = base_profit_bonus * multipliers[2] * trend_multiplier

            # Symmetrical action penalty for BUY/SELL
            action_penalty = 0.015 if action in [1, 2] else 0.0

            # Dynamic HOLD penalty based on position size and market volatility
            if action == 0:
                position_size_factor = abs(position) / max_position_size
                volatility_factor = min(
                    atr / (current_price * 0.01), 1.0
                )  # Normalized ATR
                action_penalty = 0.01 + (
                    0.04 * position_size_factor * volatility_factor
                )  # Range: 0.01-0.05

            # Loss penalty for unprofitable trades (simplified for symmetry)
            loss_penalty = -0.2 * abs(atr_normalised) if pnl < 0 else 0.0

            # No hold bonus to encourage active trading
            hold_bonus = 0.0

            position_utilisation = abs(position) / max_position_size
            soft_cap = self._get_reward_setting_float("position_soft_cap", 0.8)
            position_penalty = 0.0
            if position_utilisation > soft_cap:
                overuse = position_utilisation - soft_cap
                position_penalty = self._get_reward_setting_float(
                    "position_penalty_scale", 0.5
                ) * (
                    math.exp(
                        overuse
                        * self._get_reward_setting_float("position_penalty_exp", 4.0)
                    )
                    - 1.0
                )

            recent_positions = list(self.position_abs_history)
            recent_positions.append(abs(position))
            avg_inventory = (
                sum(recent_positions) / len(recent_positions)
                if recent_positions
                else 0.0
            )
            inventory_penalty = (
                self._get_reward_setting_float("inventory_penalty_scale", 0.1)
                * avg_inventory
            )

            position_changed = abs(position - old_position) > 1e-6
            trade_penalty = 0.0
            delta_steps = None
            if position_changed:
                if self._last_trade_step is not None:
                    delta_steps = max(1, step - self._last_trade_step)
                    self.trade_interval_history.append(delta_steps)
                else:
                    delta_steps = self.reward_settings.get("trade_cooldown_steps", 2)
                self._last_trade_step = step
                self._consecutive_trade_steps += 1
            else:
                self._consecutive_trade_steps = 0

            if position_changed:
                halflife = max(
                    1.0,
                    self._get_reward_setting_float("trade_frequency_halflife", 8.0),
                )
                trade_penalty = self._get_reward_setting_float(
                    "trade_frequency_penalty", 0.2
                ) * math.exp(-(delta_steps or halflife) / halflife)
                if (
                    delta_steps is not None
                    and delta_steps
                    < self._get_reward_setting_int("trade_cooldown_steps", 2)
                ):
                    trade_penalty += self._get_reward_setting_float(
                        "trade_cooldown_penalty", 0.2
                    )
                if self._consecutive_trade_steps > self._get_reward_setting_int(
                    "max_consecutive_trades", 5
                ):
                    trade_penalty += self._get_reward_setting_float(
                        "consecutive_trade_penalty", 0.1
                    ) * (
                        self._consecutive_trade_steps
                        - self._get_reward_setting_int("max_consecutive_trades", 5)
                    )

            projected_returns = list(self.pnl_history)
            projected_returns.append(pnl)
            volatility_penalty = 0.0
            sharpe_bonus = 0.0
            # Remove sharpe_bonus entirely to avoid rewarding HOLD in uptrend
            # if len(projected_returns) >= 2:
            #     mean_return = float(np.mean(projected_returns))
            #     std_return = float(np.std(projected_returns))
            #     if std_return > eps:
            #         sharpe_ratio = mean_return / std_return
            #         # Only apply sharpe bonus for profitable trades to avoid rewarding BUY bias in uptrend
            #         if pnl > 0:
            #             sharpe_bonus = self.reward_settings.get("sharpe_bonus_scale", 0.02) * max(0.0, sharpe_ratio)

            # Combine all reward components
            reward = (
                base_component
                + profit_bonus
                - loss_penalty
                - action_penalty
                + hold_bonus
                - position_penalty
                - inventory_penalty
                - trade_penalty
                - volatility_penalty
                + sharpe_bonus
            )

            # Clip reward
            reward = np.clip(
                reward,
                -self._get_reward_setting_float("reward_clip_value", 2.0),
                self._get_reward_setting_float("reward_clip_value", 2.0),
            )

            # Add balance penalty to maintain diversity during transition
            reward -= balance_penalty * 2.0  # Moderate penalty for imbalance
            print(
                f"DEBUG Reward: action={action}, base_reward={profit_bonus:.3f}, balance_penalty={balance_penalty * 2.0:.3f}, trend_multiplier={trend_multiplier:.1f}, final_reward={reward:.3f}"
            )
            return float(reward)
        elif curriculum_stage == "simple_portfolio":
            # Completely action-focused reward: ignore PnL, reward SELL heavily, penalize HOLD/BUY
            # Allow custom reward parameters for optimization
            custom_params: Dict[str, float] = (
                self.reward_settings.get("custom_reward_params", {})
                if self.reward_settings
                else {}
            )
            print(f"DEBUG Simple portfolio: custom_params={custom_params}")
            # Position-dependent rewards to balance BUY/SELL - SELL bias correction
            if self.position == 0:
                # No position: strongly encourage BUY, mildly penalize SELL
                if action == 0:  # HOLD
                    reward = custom_params.get("no_position_hold_penalty", -2.0)
                elif action == 1:  # BUY
                    reward = custom_params.get("no_position_buy_reward", 2.0)
                else:  # SELL
                    reward = custom_params.get("no_position_sell_penalty", -2.0)
            else:
                # Have position: encourage SELL, penalize BUY
                if action == 0:  # HOLD
                    reward = custom_params.get("has_position_hold_penalty", -2.0)
                elif action == 1:  # BUY
                    reward = custom_params.get("has_position_buy_penalty", -2.0)
                else:  # SELL
                    reward = custom_params.get("has_position_sell_reward", 2.0)

            print(f"DEBUG Simple portfolio reward: {reward}")
            return float(reward)
        elif curriculum_stage == "hold_only":
            # Stage 1: Only HOLD is rewarded, trading is heavily penalized
            base_component = 0.0
            profit_bonus = 0.0
            loss_penalty = 0.0  # No loss penalty in hold_only stage
            hold_bonus = 0.5 if action == 0 else 0.0
            action_penalty = -0.5 if action in [1, 2] else 0.0
        elif curriculum_stage == "profit_only":
            # Stage 2: Only profitable trades are rewarded, HOLD gets small reward
            base_component = 0.0
            profit_bonus = (
                max(0.0, 0.6 * atr_normalised + 0.4 * portfolio_return)
                if pnl > 0
                else 0.0
            )
            loss_penalty = 0.0  # No loss penalty in profit_only stage
            hold_bonus = 0.01 if action == 0 and abs(position) < 1e-6 else 0.0
            action_penalty = 0.05 if action in [1, 2] else 0.0
        else:  # "full"
            # Stage 3: Scalping-optimized reward function to encourage high-frequency profitable trading
            base_component = 0.0

            # Profit bonus for profitable trades (significantly increased for scalping)
            base_profit_bonus = (
                max(0.0, 1.5 * atr_normalised + 1.2 * portfolio_return)
                if pnl > 0
                else 0.0
            )

            # Balance BUY/SELL actions to encourage balanced trading (configurable multipliers)
            multipliers = self.reward_settings.get(
                "profit_bonus_multipliers", [1.0, 1.0, 0.8]
            )
            if action == 1:  # BUY action
                profit_bonus = base_profit_bonus * multipliers[0]
                action_penalty = 0.015  # Symmetrized penalty
            elif action == 2:  # SELL action
                profit_bonus = base_profit_bonus * multipliers[1]
                action_penalty = 0.015  # Symmetrized penalty
            else:  # HOLD
                profit_bonus = base_profit_bonus * multipliers[2]
                # Dynamic HOLD penalty based on position size and market volatility
                position_size_factor = abs(position) / max_position_size
                volatility_factor = min(
                    atr / (current_price * 0.01), 1.0
                )  # Normalized ATR
                action_penalty = 0.01 + (
                    0.04 * position_size_factor * volatility_factor
                )  # Range: 0.01-0.05

            # Loss penalty for unprofitable trades (simplified for symmetry)
            loss_penalty = -0.2 * abs(atr_normalised) if pnl < 0 else 0.0

            # Action balance bonus to encourage using all actions (optional forced diversity)
            if self._get_reward_setting_bool("enable_forced_diversity", False):
                action_counts = getattr(self, "_action_counts", [0, 0, 0])
                action_counts[action] += 1
                self._action_counts = action_counts

                total_actions = sum(action_counts)
                if (
                    total_actions >= 5
                ):  # Require minimum actions before enforcing diversity - reduced from 10
                    action_ratios = [count / total_actions for count in action_counts]
                    min_required_ratio = 0.1  # Require at least 10% for each action

                    # Strong penalty for not using actions at all
                    unused_penalty = 0.0
                    for _, count in enumerate(action_counts):
                        if count == 0:
                            unused_penalty += 1.0  # Equal penalty for all actions

                    # Penalty for actions below minimum ratio
                    ratio_penalty = 0.0
                    for ratio in action_ratios:
                        if ratio < min_required_ratio and ratio > 0:
                            ratio_penalty += (min_required_ratio - ratio) * 2.0

                    balance_bonus = max(
                        0.0, 0.5 - unused_penalty - ratio_penalty
                    )  # Max 0.5 bonus for good balance
                else:
                    balance_bonus = 0.0
            else:
                balance_bonus = 0.0

            # No hold bonus to encourage active trading
            hold_bonus = 0.0

        position_utilisation = abs(position) / max_position_size
        soft_cap = self.reward_settings.get("position_soft_cap", 0.8)
        position_penalty = 0.0
        if position_utilisation > soft_cap:
            overuse = position_utilisation - soft_cap
            position_penalty = self.reward_settings.get(
                "position_penalty_scale", 0.5
            ) * (
                math.exp(
                    overuse * self.reward_settings.get("position_penalty_exp", 4.0)
                )
                - 1.0
            )

        recent_positions = list(self.position_abs_history)
        recent_positions.append(abs(position))
        avg_inventory = (
            sum(recent_positions) / len(recent_positions) if recent_positions else 0.0
        )
        inventory_penalty = (
            self.reward_settings.get("inventory_penalty_scale", 0.1) * avg_inventory
        )

        position_changed = abs(position - old_position) > 1e-6
        trade_penalty = 0.0
        delta_steps = None
        if position_changed:
            if self._last_trade_step is not None:
                delta_steps = max(1, step - self._last_trade_step)
                self.trade_interval_history.append(delta_steps)
            else:
                delta_steps = self.reward_settings.get("trade_cooldown_steps", 2)
            self._last_trade_step = step
            self._consecutive_trade_steps += 1
        else:
            self._consecutive_trade_steps = 0

        if position_changed:
            halflife = max(
                1.0, float(self.reward_settings.get("trade_frequency_halflife", 8.0))
            )
            trade_penalty = self.reward_settings.get(
                "trade_frequency_penalty", 0.2
            ) * math.exp(-(delta_steps or halflife) / halflife)
            if delta_steps is not None and delta_steps < self.reward_settings.get(
                "trade_cooldown_steps", 2
            ):
                trade_penalty += self.reward_settings.get("trade_cooldown_penalty", 0.2)
            if self._consecutive_trade_steps > self.reward_settings.get(
                "max_consecutive_trades", 5
            ):
                trade_penalty += self.reward_settings.get(
                    "consecutive_trade_penalty", 0.1
                ) * (
                    self._consecutive_trade_steps
                    - self.reward_settings.get("max_consecutive_trades", 5)
                )

        projected_returns = list(self.pnl_history)
        projected_returns.append(pnl)
        volatility_penalty = 0.0
        sharpe_bonus = 0.0
        if len(projected_returns) >= 2:
            mean_return = float(np.mean(projected_returns))
            std_return = float(np.std(projected_returns))
            if std_return > eps:
                volatility_penalty = (
                    self._get_reward_setting_float("volatility_penalty_scale", 0.05)
                    * std_return
                )
                sharpe_ratio = mean_return / (std_return + eps)
                if sharpe_ratio > 0:
                    sharpe_bonus = (
                        self._get_reward_setting_float("sharpe_bonus_scale", 0.02)
                        * sharpe_ratio
                    )

                # Sortino ratio calculation (downside deviation only)
                negative_returns = [r for r in projected_returns if r < 0]
                if negative_returns:
                    downside_std = float(np.std(negative_returns))
                    if downside_std > eps:
                        sortino_ratio = mean_return / (downside_std + eps)
                        if sortino_ratio > 0:
                            sortino_bonus = (
                                self._get_reward_setting_float(
                                    "sortino_bonus_scale", 0.01
                                )
                                * sortino_ratio
                            )
                            sharpe_bonus += sortino_bonus

                # Calmar ratio calculation (annualized return / max drawdown)
                if len(self.portfolio_value_history) >= 10:
                    recent_values = self.portfolio_value_history[-10:]
                    peak = max(recent_values)
                    trough = min(recent_values)
                    if peak > eps:
                        max_drawdown = (peak - trough) / peak
                        if max_drawdown > eps:
                            calmar_ratio = mean_return / max_drawdown
                            if calmar_ratio > 0:
                                calmar_bonus = (
                                    self._get_reward_setting_float(
                                        "calmar_bonus_scale", 0.005
                                    )
                                    * calmar_ratio
                                )
                                sharpe_bonus += calmar_bonus

        drawdown_penalty = self._calculate_drawdown_penalty()
        win_streak_bonus = self._calculate_win_streak_bonus()
        stagnation_penalty = self._calculate_stagnation_penalty()
        growth_bonus = self._calculate_growth_bonus()

        cost_penalty = transaction_cost * abs(position - old_position)

        # Forced diversity penalty to encourage balanced action distribution
        forced_diversity_penalty = 0.0
        if self._get_reward_setting_bool("enable_forced_diversity", False):
            if len(self.action_history) >= 10:
                recent_actions = self.action_history[-10:]
                buy_count = sum(1 for a in recent_actions if a == 1)
                sell_count = sum(1 for a in recent_actions if a == 2)
                total_trades = buy_count + sell_count
                if total_trades > 0:
                    buy_ratio = buy_count / total_trades
                    sell_ratio = sell_count / total_trades
                    # Penalize BUY if BUY ratio is too high (>20% - further strengthened threshold)
                    if action == 1 and buy_ratio > 0.20:
                        forced_diversity_penalty = (
                            buy_ratio - 0.20
                        ) * 2.0  # Further strengthened penalty
                    # Penalize SELL if SELL ratio is too high (>20% - further strengthened threshold)
                    elif action == 2 and sell_ratio > 0.20:
                        forced_diversity_penalty = (
                            sell_ratio - 0.20
                        ) * 2.0  # Further strengthened penalty
        # Use curriculum-defined action_penalty instead of overriding it
        # action_penalty is already set in curriculum stages above

        total_penalty = (
            position_penalty
            + inventory_penalty
            + trade_penalty
            + volatility_penalty
            + drawdown_penalty
            + stagnation_penalty  # Add stagnation penalty
            + cost_penalty
            + action_penalty
            + loss_penalty  # Add loss penalty
            + forced_diversity_penalty  # Add forced diversity penalty
        )

        reward = (
            base_component
            - total_penalty
            + win_streak_bonus
            + sharpe_bonus
            + profit_bonus
            + hold_bonus
            + balance_bonus
            + growth_bonus
        )
        reward *= reward_scaling

        clip_value = self.reward_settings.get("reward_clip_value", 2.0)
        if clip_value > 0:
            reward = max(-clip_value, min(clip_value, reward))

        if not math.isfinite(reward):
            reward = 0.0

        return float(reward)

    def _calculate_drawdown_penalty(self) -> float:
        """ドローダウンペナルティの計算（50%超えの場合）"""
        if len(self.reward_history) < 20:  # より長い期間でチェック
            return 0.0

        # 最近20ステップの累積リワード
        recent_rewards = self.reward_history[-20:]
        cumulative_reward = sum(recent_rewards)

        # 基準となる初期累積リワード（最初の10ステップ）
        if len(self.reward_history) >= 30:
            initial_rewards = self.reward_history[-30:-20]
            initial_cumulative = sum(initial_rewards)

            # ドローダウンが50%超えた場合のみペナルティ
            if initial_cumulative > 0:
                drawdown_ratio = (
                    initial_cumulative - cumulative_reward
                ) / initial_cumulative
                if drawdown_ratio > 0.5:  # 50%超え
                    return drawdown_ratio * 0.05  # 軽めのペナルティ（5%）

        return 0.0

    def _calculate_stagnation_penalty(self) -> float:
        """資産停滞ペナルティの計算（資産が増加していない場合）"""
        if len(self.portfolio_value_history) < 30:  # より長い期間が必要
            return 0.0

        # 最近30ステップのポートフォリオ価値を取得
        recent_values = self.portfolio_value_history[-30:]
        initial_value = recent_values[0]
        final_value = recent_values[-1]

        # ポートフォリオ価値の変化率を計算
        if initial_value > 0:
            growth_rate = (final_value - initial_value) / initial_value

            # 変化率がマイナスまたは非常に小さい場合に軽いペナルティ（HOLDを促す）
            stagnation_threshold = -0.005  # -0.5%を最低ラインとする
            if growth_rate < stagnation_threshold:
                # 停滞度に応じた軽いペナルティ（最大0.02）
                stagnation_penalty: float = min(
                    0.02, abs(growth_rate - stagnation_threshold) * 0.5
                )
                return stagnation_penalty

        return 0.0

    def _calculate_growth_bonus(self) -> float:
        """資産増加ボーナスの計算（資産が増加した場合）"""
        if len(self.portfolio_value_history) < 30:  # 十分な履歴が必要
            return 0.0

        # 最近30ステップのポートフォリオ価値を取得
        recent_values = self.portfolio_value_history[-30:]
        initial_value = recent_values[0]
        final_value = recent_values[-1]

        # ポートフォリオ価値の変化率を計算
        if initial_value > 0:
            growth_rate = (final_value - initial_value) / initial_value

            # 変化率がプラスの場合にボーナス
            growth_threshold = 0.005  # 0.5%以上の増加でボーナス
            if growth_rate > growth_threshold:
                # 増加度に応じたボーナス（最大0.05）
                growth_bonus: float = min(0.05, growth_rate * 0.5)
                return growth_bonus

        return 0.0

    def _calculate_win_streak_bonus(self) -> float:
        """連勝ボーナスの計算"""
        if len(self.reward_history) < 5:
            return 0.0

        # 最近5ステップの勝ち数をカウント
        recent_rewards = self.reward_history[-5:]
        win_count = sum(1 for r in recent_rewards if r > 0)

        # 3勝以上でボーナス
        if win_count >= 3:
            bonus = win_count * 0.01  # 1% per win
            return bonus

        return 0.0

    def _get_observation(self) -> NDArray[np.float32]:
        """現在の状態を取得"""
        self._ensure_data_available(self.current_step)

        if self.current_step >= self.n_steps:
            step_data = self.df.iloc[-1]
        else:
            step_data = self.df.iloc[self.current_step]

        # 特徴量ベクトルの作成
        try:
            obs = step_data[self.features].to_numpy(dtype=np.float32, copy=False)
        except (KeyError, IndexError, TypeError) as e:
            # デバッグ情報
            available_cols = (
                list(step_data.index) if hasattr(step_data, "index") else []
            )
            missing_cols = [f for f in self.features if f not in available_cols]
            raise ValueError(
                f"Missing features in observation: {missing_cols}. Available: {available_cols[:10]}..."
            ) from e

        return obs

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
        """環境のクリーンアップ"""
        self.reward_history.clear()
        self.position_history.clear()
        self.df = pd.DataFrame()
        if not self._gc_step_interval:
            gc.collect()

    # ユーティリティメソッド
    def get_feature_names(self) -> list[str]:
        """特徴量名を取得"""
        return self.features

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        if len(self.reward_history) == 0:
            return {}

        rewards = np.array(self.reward_history)

        return {
            "total_reward": np.sum(rewards),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "sharpe_ratio": np.mean(rewards) / (np.std(rewards) + EPSILON),
            "max_reward": np.max(rewards),
            "total_trades": self.trades_count,
            "win_rate": np.sum(rewards > 0) / len(rewards) if len(rewards) > 0 else 0,
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
        if action == 1:  # BUY -> SELL
            flipped_action = 2
        elif action == 2:  # SELL -> BUY
            flipped_action = 1

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
