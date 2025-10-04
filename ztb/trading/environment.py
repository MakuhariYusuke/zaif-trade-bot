# Heavy Trading Environment for Reinforcement Learning
# 重特徴量ベースの取引環境

import gc
import math
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import psutil
from gymnasium import spaces
from numpy.typing import NDArray
from pandas.api import types as ptypes

from ztb.features.registry import FeatureRegistry
from ztb.utils.memory.dtypes import optimize_dtypes

EPSILON = 1e-6  # 小さい値（ゼロ除算防止用）


if TYPE_CHECKING:
    from ztb.data.streaming_pipeline import StreamingPipeline
    from ztb.trading.live.stream_to_bars import StreamToBarsConverter


class HeavyTradingEnv(gym.Env[spaces.Box, spaces.Discrete]):
    """
    重特徴量ベースの取引環境

    特徴:
    - 状態: 価格系・テクニカル系・リスク系のすべての特徴量を使用
    - 行動: 0=hold, 1=buy, 2=sell
    - リワード: (position * pnl) / (atr_14 + 1e-6) - リスク調整型
    - position: -1 (short) / 0 (flat) / 1 (long)
    - NaN処理: ゼロ埋め
    """

    @staticmethod
    def _as_bool(value: Union[bool, int, float, str, None], default: bool = False) -> bool:
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
        config: Optional[Dict[str, Any]] = None,
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

        # デフォルト設定
        self.config = config or {
            "reward_scaling": 1.0,  # 報酬スケーリング
            "transaction_cost": 0.0,  # Coincheckは0%手数料
            "max_position_size": 1.0,  # 最大ポジションサイズ
            "risk_free_rate": 0.0,  # 無リスク金利（年率）
            "timeframe": "1m",  # 時間枠: 1m, 5s, 15s, 30s など
            "feature_set": "full",  # 特徴量セット
            "curriculum_stage": "forced_balance",  # カレンシー学習段階: "forced_balance", "simple_portfolio", "hold_only", "profit_only", "full"
            "feature_storage_dtype": "float16",  # 特徴量ストレージのデータ型
            "precision_columns": ["close", "open", "high", "low", "volume"],  # 高精度列
        }

        self._process = psutil.Process()
        # 報酬関連の安全なデフォルトを設定
        reward_defaults = {
            "reward_position_soft_cap": 0.8,  # より大きなポジションを許容
            "reward_position_penalty_scale": 0.5,  # ペナルティを軽く
            "reward_position_penalty_exponent": 4.0,  # 指数を小さく
            "reward_inventory_window": 128,  # ウィンドウを小さく
            "reward_inventory_penalty_scale": 0.1,  # ペナルティを軽く
            "reward_trade_frequency_penalty": 0.2,  # 取引頻度ペナルティを軽く
            "reward_trade_frequency_halflife": 8.0,  # 半減期を短く
            "reward_trade_cooldown_steps": 2,  # クールダウンを短く
            "reward_trade_cooldown_penalty": 0.2,  # ペナルティを軽く
            "reward_max_consecutive_trades": 5,  # 最大連続取引を増やす
            "reward_consecutive_trade_penalty": 0.1,  # ペナルティを軽く
            "reward_volatility_window": 32,  # ウィンドウを小さく
            "reward_volatility_penalty_scale": 0.05,  # ペナルティを軽く
            "reward_sharpe_bonus_scale": 0.02,  # ボーナスを小さく
            "reward_clip_value": 2.0,  # クリップ値を小さく
            "enable_forced_diversity": True,  # 強制的なアクション多様性
            "initial_portfolio_value": 1_000_000.0,  # 初期ポートフォリオ価値
        }
        for key, value in reward_defaults.items():
            self.config.setdefault(key, value)

        self.reward_settings = {
            "position_soft_cap": float(self.config["reward_position_soft_cap"]),
            "position_penalty_scale": float(
                self.config["reward_position_penalty_scale"]
            ),
            "position_penalty_exp": float(
                self.config["reward_position_penalty_exponent"]
            ),
            "inventory_window": int(self.config["reward_inventory_window"]),
            "inventory_penalty_scale": float(
                self.config["reward_inventory_penalty_scale"]
            ),
            "trade_frequency_penalty": float(
                self.config["reward_trade_frequency_penalty"]
            ),
            "trade_frequency_halflife": float(
                self.config["reward_trade_frequency_halflife"]
            ),
            "trade_cooldown_steps": int(self.config["reward_trade_cooldown_steps"]),
            "trade_cooldown_penalty": float(
                self.config["reward_trade_cooldown_penalty"]
            ),
            "max_consecutive_trades": int(self.config["reward_max_consecutive_trades"]),
            "consecutive_trade_penalty": float(
                self.config["reward_consecutive_trade_penalty"]
            ),
            "volatility_window": int(self.config["reward_volatility_window"]),
            "volatility_penalty_scale": float(
                self.config["reward_volatility_penalty_scale"]
            ),
            "sharpe_bonus_scale": float(self.config["reward_sharpe_bonus_scale"]),
            "reward_clip_value": float(self.config["reward_clip_value"]),
            "profit_bonus_multipliers": self.config.get(
                "reward_profit_bonus_multipliers", [1.1, 1.15, 0.8]
            ),
            "enable_forced_diversity": self._as_bool(
                self.config.get("enable_forced_diversity", False)
            ),
        }

        self.initial_portfolio_value = float(self.config["initial_portfolio_value"])
        self.portfolio_value = self.initial_portfolio_value

        inventory_window = max(8, self.reward_settings["inventory_window"])
        volatility_window = max(8, self.reward_settings["volatility_window"])
        self.position_abs_history: deque[float] = deque(maxlen=inventory_window)
        self.pnl_history: deque[float] = deque(maxlen=volatility_window)
        self.trade_interval_history: deque[int] = deque(maxlen=64)
        self._last_trade_step: Optional[int] = None
        self._consecutive_trade_steps = 0

        memory_log_path_cfg = self.config.get("memory_log_path")
        self._memory_log_path = (
            Path(memory_log_path_cfg) if memory_log_path_cfg else None
        )
        if self._memory_log_path and not self._memory_log_path.exists():
            self._memory_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._memory_log_path.write_text(
                "timestamp,context,df_mb,rss_mb\n", encoding="utf-8"
            )

        self._memory_logging_enabled = self._as_bool(
            self.config.get("memory_logging", False)
        )

        memory_interval_cfg = self.config.get("memory_log_interval_steps")
        try:
            self._memory_log_interval_steps = max(1, int(memory_interval_cfg))
        except (TypeError, ValueError):
            self._memory_log_interval_steps = 1000

        gc_interval_cfg = self.config.get("gc_collect_interval_steps")
        try:
            self._gc_step_interval = max(0, int(gc_interval_cfg))
        except (TypeError, ValueError):
            self._gc_step_interval = 0

        preprocess_chunk_cfg = self.config.get("preprocess_chunk_size")
        try:
            self._preprocess_chunk_size = max(1, int(preprocess_chunk_cfg))
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
        feature_set = self.config.get("feature_set", "full")
        if feature_set != "full":
            from ztb.features import get_feature_manager

            manager = get_feature_manager()
            selected_features = manager.get_feature_set(feature_set)
            # データに存在する特徴量のみを選択
            self.features = [f for f in selected_features if f in all_features]
            if not self.features:
                print(
                    f"Warning: No features found for set '{feature_set}', using all available features"
                )
                self.features = all_features
        else:
            self.features = all_features

        # 相関に基づく特徴量削減を適用
        enable_correlation_reduction = self.config.get(
            "enable_correlation_reduction", True
        )
        if enable_correlation_reduction and len(self.features) > 10:
            correlation_threshold = self.config.get("correlation_threshold", 0.95)
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
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32
        )

        # 行動空間: hold, buy, sell
        self.action_space = spaces.Discrete(3)

        # 環境状態
        self.current_step = 0
        self.position = 0  # -1, 0, 1
        self.entry_price = 0.0
        self.total_pnl = 0.0
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
        self.position_history: list[int] = []
        self._action_counts: list[int] = [
            0,
            0,
            0,
        ]  # Track action usage for balance bonus

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
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.features),),
            dtype=np.float32,
        )

    def _apply_feature_storage_dtype(self) -> None:
        """Ensure feature columns use the configured storage dtype"""
        feature_dtype = str(self.config.get("feature_storage_dtype", "float16")).lower()
        dtype_map = {"float16": np.float16, "float32": np.float32}
        target_dtype = dtype_map.get(feature_dtype, np.float32)

        protected = {
            str(col).lower() for col in self.config.get("precision_columns", [])
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
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        self.position = 0
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
        self._current_episode_actions = []  # Reset action tracking for forced diversity
        self._action_counts = [0, 0, 0]  # Reset action counts for forced diversity per episode
        self._action_counts = [0, 0, 0]  # Initialize action counts for balance bonus
        self.portfolio_value_history = (
            []
        )  # Initialize portfolio value history for stagnation penalty

        # Reset previous portfolio value for step-wise reward calculation
        self._previous_portfolio_value = None

        self._prime_streaming_data()

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:  # type: ignore
        """ステップ実行"""
        # 行動の実行
        old_position = self.position
        self._execute_action(action)

        # Track action for forced diversity
        if not hasattr(self, '_current_episode_actions'):
            self._current_episode_actions = []
        self._current_episode_actions.append(action)

        # DEBUG: Print actions taken
        if len(self._current_episode_actions) <= 10:  # Only print first 10 actions per episode
            print(f"DEBUG Action {len(self._current_episode_actions)}: {action} (H=0, B=1, S=2)")

        # PnLの計算
        pnl = self._calculate_pnl()

        # 累積PnLとポートフォリオ価値の更新
        self.total_pnl += pnl
        portfolio_value = self.initial_portfolio_value + self.total_pnl
        self.portfolio_value = portfolio_value

        current_price = self._resolve_price()
        atr = self._resolve_atr()

        reward = self._calculate_reward(
            action=action,
            current_price=current_price,
            position=self.position,
            portfolio_value=portfolio_value,
            atr=atr,
            transaction_cost=self.config.get("transaction_cost", 0.001),
            reward_scaling=self.config.get("reward_scaling", 1.0),
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
            1e-8, float(self.config.get("max_position_size", 1.0))
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
            }
        )

        # 過去履歴の更新
        self.pnl_history.append(pnl)
        self.position_abs_history.append(abs(self.position))
        self.portfolio_value_history.append(
            portfolio_value
        )  # Track portfolio value for stagnation penalty

        # 報酬履歴の更新
        self.reward_history.append(reward)
        self.position_history.append(self.position)

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

        if isinstance(row, pd.Series):
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
        row = self.df.iloc[step] if isinstance(self.df, pd.DataFrame) else None
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
        """行動の実行"""
        if action == 0:  # HOLD
            pass  # ポジション維持
        elif action == 1:  # BUY
            if self.position <= 0:  # ショートまたはフラットの場合
                # ポジション変更
                if self.position < 0:  # ショートクローズ
                    self._close_position()
                self._open_position(1)
        elif action == 2:  # SELL
            if self.position >= 0:  # ロングまたはフラットの場合
                # ポジション変更
                if self.position > 0:  # ロングクローズ
                    self._close_position()
                self._open_position(-1)

    def _open_position(self, direction: int) -> None:
        """ポジションオープン"""
        current_price = self._resolve_price()
        self.position = direction * self.config["max_position_size"]
        self.entry_price = current_price
        self.trades_count += 1

    def _close_position(self) -> None:
        """ポジションクローズ"""
        if self.position != 0:
            self.trades_count += 1
            self.position = 0
            self.entry_price = 0.0

    def _calculate_pnl(self) -> float:
        """PnLの計算"""
        if self.position == 0:
            return 0.0

        current_price = self._resolve_price()
        entry_price = self.entry_price
        price_change = current_price - entry_price

        # 基本PnL
        pnl = float(self.position) * price_change

        # 取引コストの考慮（エントリー時のみ）
        if abs(self.position) > 0:
            transaction_cost = (
                abs(self.position) * entry_price * self.config["transaction_cost"]
            )
            pnl -= transaction_cost

        return float(pnl)

    def _calculate_reward(
        self,
        action: int,
        current_price: float,
        position: float,
        portfolio_value: float,
        atr: float,
        transaction_cost: float,
        reward_scaling: float,
        **kwargs: Any,
    ) -> float:
        """Calculate reward with curriculum learning stages."""
        curriculum_stage = self.config.get("curriculum_stage", "full")

        pnl = float(kwargs.get("pnl", 0.0))
        old_position = float(kwargs.get("old_position", 0.0))
        step = int(kwargs.get("step", self.current_step))

        eps = 1e-8
        atr = atr if atr > eps else 1.0
        max_position_size = max(eps, float(self.config.get("max_position_size", 1.0)))

        atr_normalised = pnl / atr
        portfolio_return = pnl / max(abs(portfolio_value), eps)

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
                balance_penalty = sum(abs(ratio - target_ratio) for ratio in action_ratios)

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
                balance_penalty = sum(abs(ratio - target_ratio) for ratio in action_ratios)

                # Strong penalty for very imbalanced distributions
                if balance_penalty > 0.5:  # More than 50% deviation from perfect balance
                    balance_penalty *= 2.0  # Double the penalty

                # Debug output
                print(f"DEBUG Balance: total_actions={total_actions}, ratios={action_ratios}, penalty={balance_penalty:.3f}")

            # Calculate base reward using standard logic (from "full" stage)
            base_component = 0.0

            # Profit bonus for profitable trades (significantly increased for scalping)
            base_profit_bonus = (
                max(0.0, 1.5 * atr_normalised + 1.2 * portfolio_return)
                if pnl > 0
                else 0.0
            )

            # Trend-aware adjustment based on SMA_20/SMA_50 ratio
            observation = kwargs.get("observation")
            trend_multiplier = 1.0
            if observation is not None and len(observation) > 5:
                # Get raw SMA values from dataframe instead of normalized observation
                if self.current_step >= self.n_steps:
                    step_data = self.df.iloc[-1]
                else:
                    step_data = self.df.iloc[self.current_step]

                sma_20 = step_data.get('sma_short', 0.0)
                sma_50 = step_data.get('sma_long', 0.0)

                print(f"DEBUG Trend: sma_20={sma_20:.4f}, sma_50={sma_50:.4f}")
                if sma_50 > eps:
                    trend_ratio = sma_20 / sma_50
                    print(f"DEBUG Trend: trend_ratio={trend_ratio:.4f}")
                    if trend_ratio > 1.01:  # Bullish trend
                        if action == 1:  # BUY
                            trend_multiplier = 1.2
                        elif action == 2:  # SELL
                            trend_multiplier = 0.8
                    elif trend_ratio < 0.99:  # Bearish trend
                        if action == 1:  # BUY
                            trend_multiplier = 0.8
                        elif action == 2:  # SELL
                            trend_multiplier = 1.2
                    # HOLD gets neutral multiplier

            # Balance BUY/SELL actions to encourage balanced trading (configurable multipliers)
            multipliers = self.reward_settings["profit_bonus_multipliers"]
            if action == 1:  # BUY action
                profit_bonus = base_profit_bonus * multipliers[0] * trend_multiplier
                action_penalty = 0.02
            elif action == 2:  # SELL action
                profit_bonus = base_profit_bonus * multipliers[1] * trend_multiplier
                action_penalty = 0.01
            else:  # HOLD
                profit_bonus = base_profit_bonus * multipliers[2] * trend_multiplier
                # Dynamic HOLD penalty based on position size and market volatility
                position_size_factor = abs(position) / max_position_size
                volatility_factor = min(
                    atr / (current_price * 0.01), 1.0
                )  # Normalized ATR
                action_penalty = 0.01 + (
                    0.04 * position_size_factor * volatility_factor
                )  # Range: 0.01-0.05

            # Loss penalty for unprofitable trades (reduced to encourage more trading)
            loss_penalty = (
                -0.2 * abs(atr_normalised + portfolio_return) if pnl < 0 else 0.0
            )

            # No hold bonus to encourage active trading
            hold_bonus = 0.0

            position_utilisation = abs(position) / max_position_size
            soft_cap = self.reward_settings["position_soft_cap"]
            position_penalty = 0.0
            if position_utilisation > soft_cap:
                overuse = position_utilisation - soft_cap
                position_penalty = self.reward_settings["position_penalty_scale"] * (
                    math.exp(overuse * self.reward_settings["position_penalty_exp"]) - 1.0
                )

            recent_positions = list(self.position_abs_history)
            recent_positions.append(abs(position))
            avg_inventory = (
                sum(recent_positions) / len(recent_positions) if recent_positions else 0.0
            )
            inventory_penalty = (
                self.reward_settings["inventory_penalty_scale"] * avg_inventory
            )

            position_changed = abs(position - old_position) > 1e-6
            trade_penalty = 0.0
            delta_steps = None
            if position_changed:
                if self._last_trade_step is not None:
                    delta_steps = max(1, step - self._last_trade_step)
                    self.trade_interval_history.append(delta_steps)
                else:
                    delta_steps = self.reward_settings["trade_cooldown_steps"]
                self._last_trade_step = step
                self._consecutive_trade_steps += 1
            else:
                self._consecutive_trade_steps = 0

            if position_changed:
                halflife = max(1.0, self.reward_settings["trade_frequency_halflife"])
                trade_penalty = self.reward_settings["trade_frequency_penalty"] * math.exp(
                    -(delta_steps or halflife) / halflife
                )
                if (
                    delta_steps is not None
                    and delta_steps < self.reward_settings["trade_cooldown_steps"]
                ):
                    trade_penalty += self.reward_settings["trade_cooldown_penalty"]
                if (
                    self._consecutive_trade_steps
                    > self.reward_settings["max_consecutive_trades"]
                ):
                    trade_penalty += self.reward_settings["consecutive_trade_penalty"] * (
                        self._consecutive_trade_steps
                        - self.reward_settings["max_consecutive_trades"]
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
            #             sharpe_bonus = self.reward_settings["sharpe_bonus_scale"] * max(0.0, sharpe_ratio)

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
                -self.reward_settings["reward_clip_value"],
                self.reward_settings["reward_clip_value"],
            )

            # Add balance penalty to maintain diversity during transition
            reward -= balance_penalty * 2.0  # Moderate penalty for imbalance
            print(f"DEBUG Reward: action={action}, base_reward={profit_bonus:.3f}, balance_penalty={balance_penalty * 2.0:.3f}, trend_multiplier={trend_multiplier:.1f}, final_reward={reward:.3f}")
            return float(reward)
        elif curriculum_stage == "simple_portfolio":
            # Completely action-focused reward: ignore PnL, reward SELL heavily, penalize HOLD/BUY
            # Allow custom reward parameters for optimization
            custom_params = self.reward_settings.get("custom_reward_params", {})
            hold_penalty = custom_params.get("hold_penalty", -1.0)
            buy_penalty = custom_params.get("buy_penalty", -0.5)
            sell_reward = custom_params.get("sell_reward", 2.0)

            if action == 0:  # HOLD - penalty
                reward = hold_penalty
            elif action == 1:  # BUY - penalty
                reward = buy_penalty
            else:  # SELL (action == 2) - reward
                reward = sell_reward

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
            multipliers = self.reward_settings["profit_bonus_multipliers"]
            if action == 1:  # BUY action
                profit_bonus = base_profit_bonus * multipliers[0]
                action_penalty = 0.02
            elif action == 2:  # SELL action
                profit_bonus = base_profit_bonus * multipliers[1]
                action_penalty = 0.01
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

            # Loss penalty for unprofitable trades (reduced to encourage more trading)
            loss_penalty = (
                -0.2 * abs(atr_normalised + portfolio_return) if pnl < 0 else 0.0
            )

            # Action balance bonus to encourage using all actions (optional forced diversity)
            if self.reward_settings.get("enable_forced_diversity", False):
                action_counts = getattr(self, "_action_counts", [0, 0, 0])
                action_counts[action] += 1
                self._action_counts = action_counts

                total_actions = sum(action_counts)
                if total_actions >= 5:  # Require minimum actions before enforcing diversity - reduced from 10
                    action_ratios = [count / total_actions for count in action_counts]
                    min_required_ratio = 0.1  # Require at least 10% for each action

                    # Strong penalty for not using actions at all - increased for SELL
                    unused_penalty = 0.0
                    for i, count in enumerate(action_counts):
                        if count == 0:
                            if i == 2:  # Extra penalty for not using SELL
                                unused_penalty += 2.0  # Increased from 1.0
                            else:
                                unused_penalty += 1.0

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
        soft_cap = self.reward_settings["position_soft_cap"]
        position_penalty = 0.0
        if position_utilisation > soft_cap:
            overuse = position_utilisation - soft_cap
            position_penalty = self.reward_settings["position_penalty_scale"] * (
                math.exp(overuse * self.reward_settings["position_penalty_exp"]) - 1.0
            )

        recent_positions = list(self.position_abs_history)
        recent_positions.append(abs(position))
        avg_inventory = (
            sum(recent_positions) / len(recent_positions) if recent_positions else 0.0
        )
        inventory_penalty = (
            self.reward_settings["inventory_penalty_scale"] * avg_inventory
        )

        position_changed = abs(position - old_position) > 1e-6
        trade_penalty = 0.0
        delta_steps = None
        if position_changed:
            if self._last_trade_step is not None:
                delta_steps = max(1, step - self._last_trade_step)
                self.trade_interval_history.append(delta_steps)
            else:
                delta_steps = self.reward_settings["trade_cooldown_steps"]
            self._last_trade_step = step
            self._consecutive_trade_steps += 1
        else:
            self._consecutive_trade_steps = 0

        if position_changed:
            halflife = max(1.0, self.reward_settings["trade_frequency_halflife"])
            trade_penalty = self.reward_settings["trade_frequency_penalty"] * math.exp(
                -(delta_steps or halflife) / halflife
            )
            if (
                delta_steps is not None
                and delta_steps < self.reward_settings["trade_cooldown_steps"]
            ):
                trade_penalty += self.reward_settings["trade_cooldown_penalty"]
            if (
                self._consecutive_trade_steps
                > self.reward_settings["max_consecutive_trades"]
            ):
                trade_penalty += self.reward_settings["consecutive_trade_penalty"] * (
                    self._consecutive_trade_steps
                    - self.reward_settings["max_consecutive_trades"]
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
                    self.reward_settings["volatility_penalty_scale"] * std_return
                )
                sharpe_ratio = mean_return / (std_return + eps)
                if sharpe_ratio > 0:
                    sharpe_bonus = (
                        self.reward_settings["sharpe_bonus_scale"] * sharpe_ratio
                    )

                # Sortino ratio calculation (downside deviation only)
                negative_returns = [r for r in projected_returns if r < 0]
                if negative_returns:
                    downside_std = float(np.std(negative_returns))
                    if downside_std > eps:
                        sortino_ratio = mean_return / (downside_std + eps)
                        if sortino_ratio > 0:
                            sortino_bonus = (
                                self.reward_settings.get("sortino_bonus_scale", 0.01)
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
                                    self.reward_settings.get(
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

        clip_value = self.reward_settings["reward_clip_value"]
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
                stagnation_penalty = min(
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
                growth_bonus = min(0.05, growth_rate * 0.5)
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

    def _get_observation(self) -> NDArray[np.floating]:  # type: ignore[override]
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
        if hasattr(self, '_current_episode_actions'):
            return self._current_episode_actions.copy()
        return []

    def get_last_step_time(self) -> float:
        """最後のステップ時間を取得（秒単位）"""
        # 簡易実装: 固定値
        return 0.001
