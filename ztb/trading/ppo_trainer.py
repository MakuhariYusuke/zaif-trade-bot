# PPO Training Script for Heavy Trading Environment
# 重特徴量取引環境でのPPO学習

import json
import logging
import os
import pickle
import sys
import zlib
from datetime import datetime
from logging.handlers import BufferingHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from optuna import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

# ローカルモジュールのインポート
# sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # プロジェクトルートを追加
from ztb.training import (
    ResumeHandler,
    TrainingCheckpointConfig,
    TrainingCheckpointManager,
)
from ztb.training.evaluation_callback import DSREvaluationCallback
from ztb.utils.seed_manager import set_global_seed

if TYPE_CHECKING:
    from ztb.data.streaming_pipeline import StreamingPipeline

from ztb.utils.observability import setup_observability

from ..utils.cache.feature_cache import FeatureCache
from ..utils.memory.dtypes import downcast_df
from ..utils.perf.cpu_tune import apply_cpu_tuning
from .environment import HeavyTradingEnv


def save_checkpoint_async(
    model: BaseAlgorithm,
    path: str,
    notifier: Optional[Any] = None,
    light_mode: bool = False,
    compressor: str = "auto",
) -> None:
    """非同期チェックポイント保存（圧縮対応）"""

    def _job() -> None:
        try:
            # 圧縮方式の選択
            if compressor == "auto":
                # データサイズ推定（500MB基準）
                estimated_size = 500 * 1024 * 1024
                selected_compressor = _select_checkpoint_compressor(estimated_size)
            else:
                selected_compressor = compressor

            if light_mode:
                # 軽量モード: policy + value_net + scaler の最小保存セット
                checkpoint_data = {
                    "policy": model.policy.state_dict(),
                    "value_net": model.value_net.state_dict()
                    if hasattr(model, "value_net")
                    else None,  # type: ignore
                    "scaler": getattr(model, "scaler", None),
                }

                # pickle化して圧縮
                import pickle

                data = pickle.dumps(checkpoint_data, protocol=pickle.HIGHEST_PROTOCOL)
                compressed_data = _compress_data(data, selected_compressor)

                # 原子的保存
                tmp_path = path + "_light.tmp"
                final_path = path + "_light.zip"
                Path(tmp_path).write_bytes(compressed_data)
                os.replace(tmp_path, final_path)

            else:
                # 通常モード: 完全保存
                final_path = path + ".zip"
                model.save(final_path)

            logging.info(
                f"[CKPT] saved: {final_path} (compressor: {selected_compressor})"
            )
            if notifier:
                notifier.send_custom_notification(
                    "💾 Checkpoint Saved", f"Saved to {final_path}", color=0x0000FF
                )

        except Exception as e:
            logging.exception(f"[CKPT] save failed: {e}")
            if notifier:
                notifier.send_error_notification(
                    "Checkpoint Save Error", f"Failed to save {path}: {str(e)}"
                )

    # 非同期実行
    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(_job)
    executor.shutdown(wait=False)


def _select_checkpoint_compressor(data_size_bytes: int) -> str:
    """チェックポイント用圧縮方式選択（FeatureCache._select_compressorと共通ロジック）"""
    try:
        import zstandard as zstd

        has_zstd = True
    except ImportError:
        has_zstd = False

    try:
        import lz4.frame

        has_lz4 = True
    except ImportError:
        has_lz4 = False

    # チェックポイントは基本的にarchival用途
    if data_size_bytes < 50 * 1024 * 1024:  # < 50MB
        return "lz4" if has_lz4 else "zlib"
    else:
        return "zstd" if has_zstd else "zlib"


def _compress_data(data: bytes, compressor: str) -> bytes:
    """データ圧縮"""
    if compressor == "zstd":
        try:
            import zstandard as zstd

            return zstd.ZstdCompressor(level=3).compress(data)
        except ImportError:
            compressor = "zlib"

    if compressor == "lz4":
        try:
            import lz4.frame

            return lz4.frame.compress(data)  # type: ignore
        except ImportError:
            compressor = "zlib"

    # fallback to zlib
    import zlib

    return zlib.compress(data, 6)


def load_checkpoint(model: BaseAlgorithm, path: str, light_mode: bool = False) -> None:
    """チェックポイント読み込み（軽量/通常両対応）"""
    if light_mode:
        # 軽量チェックポイントの読み込み
        import pickle

        try:
            import zstandard as zstd

            has_zstd = True
        except ImportError:
            has_zstd = False

        try:
            import lz4.frame

            has_lz4 = True
        except ImportError:
            has_lz4 = False

        # 圧縮形式の自動判定
        if path.endswith("_light.zip"):
            data = Path(path).read_bytes()

            # 圧縮形式判定（簡易的）
            if data.startswith(b"\x28\xb5\x2f\xfd"):  # Zstd magic
                decompressed = (
                    zstd.ZstdDecompressor().decompress(data)
                    if has_zstd
                    else zlib.decompress(data)
                )
            elif data.startswith(b"\x04\x22\x4d\x18"):  # LZ4 magic
                decompressed = (
                    lz4.frame.decompress(data) if has_lz4 else zlib.decompress(data)
                )
            else:
                decompressed = zlib.decompress(data)

            checkpoint_data = pickle.loads(decompressed)

            # モデルにロード
            model.policy.load_state_dict(checkpoint_data["policy"])
            if checkpoint_data.get("value_net") and hasattr(model, "value_net"):
                model.value_net.load_state_dict(checkpoint_data["value_net"])  # type: ignore
            if checkpoint_data.get("scaler"):
                model.scaler = checkpoint_data["scaler"]  # type: ignore

        else:
            # 通常チェックポイント
            model.load(path)
    else:
        # 通常チェックポイント
        model.load(path)


class TensorBoardCallback(BaseCallback):
    """TensorBoardログ用コールバック"""

    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        # TensorBoard logging is handled by logger configuration
        return True


class CheckpointCallback(BaseCallback):
    """Checkpoint callback supporting async saves and training state snapshots."""

    def __init__(
        self,
        save_freq: int,
        save_path: Optional[str],
        name_prefix: str = "checkpoint",
        verbose: int = 0,
        notifier: Optional[Any] = None,
        session_id: Optional[str] = None,
        light_mode: bool = False,
        training_manager: Optional[TrainingCheckpointManager] = None,
        stream_state_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path) if save_path else None
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix
        self.notifier = notifier
        self.session_id = session_id
        self.light_mode = light_mode
        self.training_manager = training_manager
        self.stream_state_provider = stream_state_provider

    def _on_step(self) -> bool:
        try:
            current_step = int(self.num_timesteps)
            saved = False

            if self.training_manager and self.training_manager.should_checkpoint(
                current_step
            ):
                stream_state = (
                    self.stream_state_provider() if self.stream_state_provider else {}
                )
                self.training_manager.save(
                    step=current_step,
                    model=self.model,
                    metrics=self._collect_metrics(),
                    extra={
                        "session_id": self.session_id,
                        "name_prefix": self.name_prefix,
                    },
                    stream_state=stream_state,
                )
                saved = True
            elif (
                self.save_path
                and self.save_freq > 0
                and self.n_calls % self.save_freq == 0
            ):
                checkpoint_path = self.save_path / f"{self.name_prefix}_{current_step}"
                save_checkpoint_async(
                    self.model,
                    str(checkpoint_path),
                    self.notifier,
                    light_mode=self.light_mode,
                )
                saved = True

            if saved:
                self._log_progress(current_step)

        except Exception as e:
            logging.error(f"Error in checkpoint callback: {e}")
            if self.notifier:
                self.notifier.send_error_notification("Checkpoint Error", str(e))

        return True

    def _collect_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "num_timesteps": int(self.num_timesteps),
            "callback_calls": int(self.n_calls),
        }

        infos = None
        if isinstance(self.locals, dict):
            infos = self.locals.get("infos")

        if infos:
            episode_rewards = [
                info["episode"]["r"]
                for info in infos
                if isinstance(info, dict) and "episode" in info
            ]
            episode_lengths = [
                info["episode"]["l"]
                for info in infos
                if isinstance(info, dict) and "episode" in info
            ]
            if episode_rewards:
                metrics["episode_reward_mean"] = float(np.mean(episode_rewards))
                metrics["episode_reward_max"] = float(np.max(episode_rewards))
                metrics["episode_reward_min"] = float(np.min(episode_rewards))
            if episode_lengths:
                metrics["episode_length_mean"] = float(np.mean(episode_lengths))
        return metrics

    def _log_progress(self, current_step: int) -> None:
        total_timesteps = getattr(self.model, "_total_timesteps", None) or current_step
        progress_percent = (
            (current_step / total_timesteps * 100) if total_timesteps else 0.0
        )
        progress_msg = (
            f"Step {current_step:,} / {total_timesteps:,} ({progress_percent:.1f}%)"
        )

        logging.info(progress_msg)

        if int(progress_percent) % 10 == 0 and self.notifier:
            self.notifier.send_custom_notification(
                f"📊 Training Progress ({self.session_id})",
                progress_msg,
                color=0x00FF00,
            )

        if self.verbose > 0:
            print(progress_msg)

        import gc

        gc.collect()


class SafetyCallback(BaseCallback):
    """安全策コールバック - トレード数が0のまま学習を停止"""

    def __init__(self, max_zero_trades: int = 10000, verbose: int = 0) -> None:
        """
        コンストラクタ

        Args:
            max_zero_trades (int): トレード数が0のまま許容する最大ステップ数
            verbose (int): 詳細ログレベル
        """
        super().__init__(verbose)
        self.max_zero_trades = max_zero_trades
        self.zero_trade_count = 0

    def _on_step(self) -> bool:
        """
        各ステップでトレード数をチェックし、0のまま続くと学習を停止

        Returns:
            bool: トレーニングを継続する場合はTrue
        """
        try:
            total_timesteps = getattr(self.model, "_total_timesteps", 1000000)
            progress_percent = (self.n_calls / total_timesteps) * 100
            progress_msg = (
                f"Step {self.n_calls:,} / {total_timesteps:,} ({progress_percent:.1f}%)"
            )
            if self.model.env:
                stats_list = self.model.env.get_attr("get_statistics")
                if stats_list and callable(stats_list[0]):
                    stats = stats_list[0]()
                    if isinstance(stats, dict):
                        total_trades = stats.get("total_trades", 0)

                        if total_trades == 0:
                            self.zero_trade_count += 1
                            if self.zero_trade_count >= self.max_zero_trades:
                                logging.warning(
                                    f"No trades for {self.max_zero_trades} steps, stopping training"
                                )
                                return False  # 学習停止
                            else:
                                self.zero_trade_count = 0
        except Exception as e:
            logging.error(f"Error in safety callback: {e}")

        return True


class PPOTrainer:
    """PPOトレーニングマネージャー"""

    def __init__(
        self,
        data_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_interval: int = 10000,
        checkpoint_dir: str = "models/checkpoints",
        streaming_pipeline: Optional["StreamingPipeline"] = None,
        stream_batch_size: int = 256,
    ) -> None:
        """PPOトレーニングマネージャー"""
        apply_cpu_tuning()

        if data_path is None and streaming_pipeline is None:
            raise ValueError("Either data_path or streaming_pipeline must be provided")

        self.streaming_pipeline = streaming_pipeline
        self.stream_batch_size = stream_batch_size
        self.data_path = Path(data_path) if data_path is not None else None

        base_config = self._get_default_config()
        if config:
            training_section = config.get("training")
            if training_section:
                base_config.update(training_section)
                base_config["training"] = training_section
            else:
                base_config.update(config)
        self.config = base_config

        self._setup_cpu_optimization()

        if "training" in self.config:
            self.config.update(self.config["training"])

        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.config["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(self.config["model_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.observability = setup_observability(
            f"ppo_trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            self.log_dir / "observability",
        )

        # Track initial system resources
        self._initial_memory_mb = 0.0
        self._peak_memory_mb = 0.0
        self._initial_cpu_percent = 0.0
        if HAS_PSUTIL:
            process = psutil.Process()
            self._initial_memory_mb = process.memory_info().rss / (1024 * 1024)
            self._initial_cpu_percent = process.cpu_percent(interval=0.1)

        if self.streaming_pipeline:
            self.streaming_pipeline.observability = self.observability

        if self.data_path is not None:
            self.df = self._load_data()
        elif self.streaming_pipeline is not None:
            self.df = self.streaming_pipeline.buffer.to_dataframe()
            if self.df.empty:
                logging.warning(
                    "Streaming pipeline buffer is empty during trainer initialization."
                )
        else:
            self.df = pd.DataFrame()

        self.env = self._create_env()

        checkpoint_cfg = self.config.get("checkpoint", {})
        self.training_checkpoint_config = TrainingCheckpointConfig(
            interval_steps=checkpoint_interval,
            keep_last=checkpoint_cfg.get("keep_last", 5),
            compress=checkpoint_cfg.get("compress", "zlib"),
            async_save=checkpoint_cfg.get("async_save", True),
            include_optimizer=checkpoint_cfg.get("include_optimizer", True),
            include_replay_buffer=checkpoint_cfg.get("include_replay_buffer", True),
            include_rng_state=checkpoint_cfg.get("include_rng_state", True),
        )
        self.training_checkpoint_manager = TrainingCheckpointManager(
            save_dir=str(self.checkpoint_dir),
            config=self.training_checkpoint_config,
            observability=self.observability,
        )
        self.resume_handler = ResumeHandler(
            self.training_checkpoint_manager,
            streaming_pipeline=self.streaming_pipeline,
        )

    def _setup_cpu_optimization(self) -> None:
        """CPU最適化設定"""
        from ..utils.perf.cpu_tune import auto_config_threads

        # 環境変数から設定取得
        num_processes = int(os.environ.get("PARALLEL_PROCESSES", "1"))
        pin_cores_str = os.environ.get("CPU_AFFINITY")
        pin_to_cores = (
            [int(x) for x in pin_cores_str.split(",")] if pin_cores_str else None
        )

        # 自動設定決定
        cpu_config = auto_config_threads(num_processes, pin_to_cores)

        # 環境変数設定
        for key, value in cpu_config.items():
            if key.startswith(("OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_")):
                os.environ[key] = str(value)

        # PyTorch設定
        torch.set_num_threads(cpu_config["torch_threads"])
        torch.backends.mkldnn.enabled = True  # type: ignore

        # ログ出力
        logging.info(
            f"CPU: phys={cpu_config['physical_cores']}, log={cpu_config['logical_cores']}, "
            f"procs={cpu_config['num_processes']}, pin={cpu_config['pin_to_cores']}, "
            f"torch={cpu_config['torch_threads']}, OMP={cpu_config['OMP_NUM_THREADS']}, "
            f"MKL={cpu_config['MKL_NUM_THREADS']}, OPENBLAS={cpu_config['OPENBLAS_NUM_THREADS']}"
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """
        デフォルトのPPOトレーニング設定を返します。

        Returns:
            dict: デフォルト設定の辞書
        """
        return {
            "total_timesteps": 200000,  # 本番用と同じ値に統一
            "eval_freq": 5000,
            "n_eval_episodes": 5,
            "batch_size": 64,
            "n_steps": 2048,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "log_dir": "./logs/",
            "model_dir": "./models/",
            "tensorboard_log": "./tensorboard/",
            "verbose": 1,
            "seed": 42,
            "stream": {"batch_size": 256},
            "checkpoint": {
                "keep_last": 5,
                "compress": "zlib",
                "async_save": True,
                "include_optimizer": True,
                "include_replay_buffer": True,
                "include_rng_state": True,
            },
        }

    def _stream_state_snapshot(self) -> Dict[str, Any]:
        if not self.streaming_pipeline:
            return {}

        buffer_df = self.streaming_pipeline.buffer.to_dataframe(
            last_n=self.streaming_pipeline.lookback_rows
        )
        stats = self.streaming_pipeline.stats()
        return {
            "buffer": buffer_df,
            "stats": {
                "rows": stats.buffer.rows,
                "capacity": stats.buffer.capacity,
                "total_rows_streamed": stats.total_rows_streamed,
                "last_batch_at": stats.last_batch_at.isoformat()
                if stats.last_batch_at
                else None,
                "last_batch_rows": stats.last_batch_rows,
            },
        }

    def _load_data(self) -> pd.DataFrame:
        """
        指定されたパスからトレーニングデータを読み込みます。
        ワイルドカードを含むパスにも対応し、複数のファイルを結合できます。

        Returns:
            pd.DataFrame: 読み込まれたデータフレーム

        Raises:
            FileNotFoundError: データファイルが見つからない場合
            ValueError: サポートされていないファイル形式または有効なデータファイルがない場合
        """
        if self.data_path is None:
            raise RuntimeError("Data path is required when loading static data")

        data_path = Path(self.data_path)

        # メモリ効率化: FeatureCacheチェック
        memory_config = self.config.get("memory", {})
        if memory_config.get("enable_cache", False):
            cache = FeatureCache(
                memory_config.get("cache_dir", "data/cache"),
                memory_config.get("cache_max_mb", 1000),
                memory_config.get("max_age_days", 7),
                memory_config.get("compressor", "zstd"),
            )
            params = {
                "data_path": str(data_path),
                "version": "v1",
                "downcast": memory_config.get("downcast", True),
            }
            cached = cache.get(str(data_path), params)
            if cached is not None:
                df = pickle.loads(cached)
                # メモリ使用量計算
                cached_size_mb = len(cached) / (1024 * 1024)
                logging.info(
                    f"[CACHE] Hit ({cached_size_mb:.1f} MB loaded) for {data_path}"
                )
                # ダウンキャスト適用
                if memory_config.get("downcast", True):
                    df = downcast_df(
                        df,
                        float_dtype=memory_config.get("float_dtype", "float32"),
                        int_dtype=memory_config.get("int_dtype", "int32"),
                    )
                return cast(pd.DataFrame, df)
            else:
                logging.info(f"[CACHE] Miss for {data_path}")

        # ワイルドカードが含まれる場合
        if "*" in str(data_path):
            # globパターンでファイルを検索
            import glob

            file_paths = glob.glob(str(data_path))

            if not file_paths:
                raise FileNotFoundError(f"No files found matching pattern: {data_path}")
            logging.info(f"Found {len(file_paths)} files matching pattern: {data_path}")

            # すべてのファイルを読み込んで結合
            dfs = []
            for file_path_str in file_paths:
                file_path = Path(file_path_str)
                if file_path.suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                elif file_path.suffix == ".csv":
                    df = pd.read_csv(file_path)
                else:
                    logging.warning(f"Skipping unsupported file: {file_path}")
                    continue

                dfs.append(df)
                logging.info(f"Loaded {file_path.name}: {len(df)} rows")
                print(f"Loaded {file_path.name}: {len(df)} rows")

            if not dfs:
                raise ValueError("No valid data files found")

            # データを結合
            df = pd.concat(dfs, ignore_index=True)

            # タイムスタンプでソート
            if "ts" in df.columns:
                df = df.sort_values("ts").reset_index(drop=True)

        else:
            # 単一ファイルの場合
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")

            if data_path.suffix == ".parquet":
                df = pd.read_parquet(data_path)
            elif data_path.suffix == ".csv":
                df = pd.read_csv(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
        logging.info(f"Total loaded data: {len(df)} rows, {len(df.columns)} columns")
        logging.info(f"Columns: {list(df.columns)}")
        print(f"Columns: {list(df.columns)}")

        # メモリ効率化: ダウンキャスト
        memory_config = self.config.get("memory", {})
        if memory_config.get("downcast", True):
            df = downcast_df(
                df,
                float_dtype=memory_config.get("float_dtype", "float32"),
                int_dtype=memory_config.get("int_dtype", "int32"),
            )
            logging.info(
                f"[MEMORY] Downcast applied: float->{memory_config.get('float_dtype', 'float32')}, int->{memory_config.get('int_dtype', 'int32')}"
            )

        # メモリ効率化: キャッシュ保存
        if memory_config.get("enable_cache", False):
            cache = FeatureCache(
                memory_config.get("cache_dir", "data/cache"),
                memory_config.get("cache_max_mb", 1000),
            )
            params = {
                "data_path": str(data_path),
                "version": "v1",
                "downcast": memory_config.get("downcast", True),
            }
            # メモリ使用量計算
            data_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            compressed = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_size_mb = len(compressed) / (1024 * 1024)
            cache.put(str(data_path), params, compressed)
            logging.info(
                f"[CACHE] Saved ({data_size_mb:.1f} MB -> {compressed_size_mb:.1f} MB compressed) for {data_path}"
            )

        return df

    def _create_env(self) -> Any:
        """
        トレーニング用のHeavyTradingEnv環境を作成し、Monitorでラップします。
        設定ファイルから取引手数料を読み込みます。

        Returns:
            Monitor: モニターでラップされた環境オブジェクト
        """
        # rl_config.jsonから手数料設定を読み込み
        config_path: Optional[str] = os.environ.get("RL_CONFIG_PATH")
        if config_path is None:
            # プロジェクトルートの絶対パスをデフォルトに
            config_path = str(
                Path(__file__).parent.parent.parent.parent / "rl_config.json"
            )
        config_path_obj = Path(config_path)
        transaction_cost = 0.001  # デフォルト
        if config_path_obj.exists():
            with open(config_path_obj, "r") as f:
                rl_config = json.load(f)
            fee_config = rl_config.get("fee_model", {})
            transaction_cost = fee_config.get("default_fee_rate", 0.001)
        env_config = {
            "reward_scaling": 1.0,
            "transaction_cost": transaction_cost,
            "max_position_size": 1.0,
            "risk_free_rate": 0.0,
        }

        base_df = (
            self.df if isinstance(self.df, pd.DataFrame) and not self.df.empty else None
        )
        stream_batch = self.config.get("stream", {}).get(
            "batch_size", self.stream_batch_size
        )
        env: Union[HeavyTradingEnv, Monitor[HeavyTradingEnv, Any]] = HeavyTradingEnv(
            base_df,
            env_config,
            streaming_pipeline=self.streaming_pipeline,
            stream_batch_size=stream_batch,
        )
        env = Monitor(env, str(self.log_dir / "monitor.csv"))

        return env

    def train(
        self, notifier: Optional[Any] = None, session_id: Optional[str] = None
    ) -> PPO:
        """
        設定に基づいてPPOモデルをトレーニングします。

        Args:
            notifier: 通知用のオプショナルなNotifierオブジェクト
            session_id: トレーニングセッションのID

        Returns:
            PPO: トレーニング済みのPPOモデル

        Raises:
            Exception: トレーニング中にエラーが発生した場合
        """
        # I/O最適化: ログバッファリングを設定
        buffer_handler = BufferingHandler(1000)  # 1000メッセージごとにフラッシュ
        buffer_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        buffer_handler.setFormatter(formatter)
        logging.getLogger().addHandler(buffer_handler)

        if hasattr(self, "observability") and self.observability:
            self.observability.log_event(
                "training_start",
                {
                    "total_timesteps": self.config.get("total_timesteps", 0),
                    "session_id": session_id,
                },
            )

        logging.info("Starting PPO training...")

        # Set global seed for reproducibility
        set_global_seed(self.config.get("seed"))

        # モデルの作成
        model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            ent_coef=self.config["ent_coef"],
            vf_coef=self.config["vf_coef"],
            max_grad_norm=self.config["max_grad_norm"],
            tensorboard_log=self.config["tensorboard_log"],
            verbose=self.config["verbose"],
            seed=self.config["seed"],
        )

        resume_state = self.resume_handler.resume(model)
        if resume_state:
            logging.info(
                "Resumed training from checkpoint at step %s", resume_state.step
            )
            if self.observability:
                self.observability.log_event(
                    "checkpoint_resume",
                    {
                        "step": resume_state.step,
                        "metrics": resume_state.metrics,
                    },
                )
            if resume_state.metrics:
                logging.info("Checkpoint metrics snapshot: %s", resume_state.metrics)

        # コールバックの設定
        eval_callback = DSREvaluationCallback(
            self.env,
            best_model_save_path=str(self.model_dir / "best_model"),
            log_path=str(self.log_dir / "eval"),
            eval_freq=self.config["eval_freq"],
            n_eval_episodes=self.config["n_eval_episodes"],
            deterministic=True,
            render=False,
            bootstrap_samples=1000,
            dsr_trials=50,
        )

        tensorboard_callback = TensorBoardCallback(eval_freq=self.config["eval_freq"])

        # チェックポイントコールバックの設定
        checkpoint_callback = CheckpointCallback(
            save_freq=self.checkpoint_interval,
            save_path=str(self.checkpoint_dir),
            name_prefix=f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            verbose=1,
            notifier=notifier,
            session_id=session_id,
            light_mode=self.config.get("training", {}).get("checkpoint_light", False),
            training_manager=self.training_checkpoint_manager,
            stream_state_provider=self._stream_state_snapshot,
        )

        # 安全策コールバックの設定
        safety_callback = SafetyCallback(max_zero_trades=1000, verbose=1)

        try:
            logging.info(
                f"Training started with total_timesteps: {self.config['total_timesteps']}"
            )
            model.learn(
                total_timesteps=self.config["total_timesteps"],
                callback=[
                    eval_callback,
                    tensorboard_callback,
                    checkpoint_callback,
                    safety_callback,
                ],
                progress_bar=True,
            )

            model.save(str(self.model_dir / "final_model"))
            logging.info(f"Model saved to {self.model_dir / 'final_model'}")

            memory_config = self.config.get("memory", {})
            if memory_config.get("enable_cache", False):
                cache = FeatureCache(
                    memory_config.get("cache_dir", "data/cache"),
                    memory_config.get("cache_max_mb", 1000),
                )
                stats = cache.get_stats()
                logging.info(
                    f"[CACHE] Final stats: {stats['hits']} hits, {stats['misses']} misses, "
                    f"{stats['hit_rate']:.1f}% hit rate, {stats['evictions']} evictions, "
                    f"{stats['compression_ratio']:.1f}% compression ratio"
                )

                health = cache.monitor_cache_health()
                if health["warnings"]:
                    for warning in health["warnings"]:
                        logging.warning(f"[CACHE] {warning}")
                else:
                    logging.info(
                        f"[CACHE] Health check passed - {health['size_mb']:.1f}MB used"
                    )

            if hasattr(self, "observability") and self.observability:
                final_timesteps = getattr(
                    model, "num_timesteps", self.config.get("total_timesteps", 0)
                )

                # Collect final system resource usage
                final_memory_mb = 0.0
                final_cpu_percent = 0.0
                if HAS_PSUTIL:
                    process = psutil.Process()
                    final_memory_mb = process.memory_info().rss / (1024 * 1024)
                    final_cpu_percent = process.cpu_percent(interval=0.1)
                    self._peak_memory_mb = max(self._peak_memory_mb, final_memory_mb)

                summary = {
                    "total_timesteps": self.config.get("total_timesteps", 0),
                    "final_timesteps": final_timesteps,
                    "resume_step": resume_state.step if resume_state else None,
                    "initial_memory_mb": self._initial_memory_mb,
                    "final_memory_mb": final_memory_mb,
                    "peak_memory_mb": self._peak_memory_mb,
                    "initial_cpu_percent": self._initial_cpu_percent,
                    "final_cpu_percent": final_cpu_percent,
                }
                self.observability.log_event("training_complete", summary)
                self.observability.record_metrics(
                    {
                        "training_total_timesteps": float(final_timesteps),
                        "training_peak_memory_mb": self._peak_memory_mb,
                        "training_final_memory_mb": final_memory_mb,
                    }
                )
                self.observability.export_artifact("training_summary", summary)

            return model

        except Exception as e:
            logging.exception(f"Training failed: {e}")
            if self.observability:
                self.observability.log_event("training_failed", {"error": repr(e)})
            if notifier:
                notifier.send_error_notification(
                    "Training Failed", f"Session {session_id}: {str(e)}"
                )
            raise
        finally:
            buffer_handler.flush()
            logging.getLogger().removeHandler(buffer_handler)
            if hasattr(self, "observability") and self.observability:
                self.observability.close()

    def evaluate(
        self, model_path: Optional[str] = None, n_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        指定されたモデルを評価し、統計情報を返します。

        Args:
            model_path (Optional[str]): 評価するモデルのパス。Noneの場合は最良モデルを使用。
            n_episodes (int): 評価エピソード数

        Returns:
            dict: 評価結果の統計情報（平均報酬など）
        """
        if model_path is None:
            model_path = str(self.model_dir / "best_model")

        # モデルの読み込み
        model = PPO.load(model_path)

        # 評価環境の作成
        eval_env = DummyVecEnv([lambda: self._create_env()])

        # 評価の実行
        episode_rewards = []
        episode_lengths: list[int] = []

        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                # obsがタプルの場合、最初の要素（観測データ）を使用
                predict_obs = obs[0] if isinstance(obs, tuple) else obs
                action, _ = model.predict(predict_obs, deterministic=True)
                obs, reward, done_vec, _ = eval_env.step(action)
                done = done_vec[0]
                episode_reward += reward[0]
                episode_length += 1

            episode_rewards.append(episode_reward)
            logging.info(
                f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}"
            )

        # 統計の計算
        stats = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "total_episodes": n_episodes,
        }

        # 結果の保存
        results_path = self.log_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(stats, f, indent=2)
        logging.info(f"Evaluation results saved to {results_path}")
        return stats

    def visualize_training(self) -> None:
        """
        monitor.csvログからトレーニング結果を可視化し、画像を保存します。
        """
        # モニターログの読み込み
        monitor_file = self.log_dir / "monitor.csv"
        if monitor_file.exists():
            # ヘッダー行数を自動判定
            with open(monitor_file, "r", encoding="utf-8") as f:
                header_lines = 0
                for line in f:
                    if line.startswith("#"):
                        header_lines += 1
                    else:
                        break
            monitor_df = pd.read_csv(monitor_file, skiprows=header_lines)

            # プロットの作成
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # リワードの推移
            axes[0, 0].plot(monitor_df["r"], alpha=0.7)
            axes[0, 0].set_title("Episode Rewards")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].grid(True)

            # エピソード長の推移
            axes[0, 1].plot(monitor_df["l"], alpha=0.7)
            axes[0, 1].set_title("Episode Lengths")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Length")
            axes[0, 1].grid(True)

            # リワードのヒストグラム
            axes[1, 0].hist(monitor_df["r"], bins=50, alpha=0.7)
            axes[1, 0].set_title("Reward Distribution")
            axes[1, 0].set_xlabel("Reward")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].grid(True)

            # 累積リワード
            axes[1, 1].plot(np.cumsum(monitor_df["r"]), alpha=0.7)
            axes[1, 1].set_title("Cumulative Rewards")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Cumulative Reward")
            axes[1, 1].grid(True)

            plt.tight_layout()
            plt.show()

            logging.info(
                f"Training visualization saved to {self.log_dir / 'training_visualization.png'}"
            )
            print(
                f"Training visualization saved to {self.log_dir / 'training_visualization.png'}"
            )


def optimize_hyperparameters(data_path: str, n_trials: int = 50) -> Dict[str, Any]:
    """
    Optunaによるハイパーパラメータ最適化
    :param data_path: トレーニングデータのパス
    :param n_trials: 試行回数
    """

    def objective(trial: Trial) -> float:
        """
        Optunaの目的関数。指定されたハイパーパラメータでモデルを評価します。

        Args:
            trial: OptunaのTrialオブジェクト

        Returns:
            float: 評価結果の平均報酬
        """
        # ハイパーパラメータのサンプリング
        config = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "n_epochs": trial.suggest_int("n_epochs", 5, 20),
            "gamma": trial.suggest_float("gamma", 0.9, 0.999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
            "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
            "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.7),
            "total_timesteps": 50000,  # 最適化時は短めに
            "eval_freq": 10000,
            "n_eval_episodes": 3,
            "log_dir": "./logs/optuna/",
            "model_dir": "./models/optuna/",
            "tensorboard_log": "./tensorboard/optuna/",
            "verbose": 0,
            "seed": 42,
        }

        # トレーナーの作成とトレーニング
        trainer = PPOTrainer(data_path, config)
        _ = trainer.train()

        # 評価
        eval_stats = trainer.evaluate(n_episodes=5)

        return cast(float, eval_stats["mean_reward"])

    # Optunaスタディの作成
    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(), pruner=MedianPruner()
    )

    # 最適化の実行
    study.optimize(objective, n_trials=n_trials)

    logging.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logging.info(f"  {key}: {value}")

    logging.info(f"Best reward: {study.best_value}")

    return study.best_params


def main() -> None:
    """
    メイン関数
    コマンドライン引数でモードを指定して実行
    例: python main.py --data ./data/train_features.parquet --mode train
    """

    from ztb.schema import DEFAULT_TRAINING_CONFIG
    from ztb.utils.cli_common import CLIFormatter, CLIValidator, create_standard_parser

    parser = create_standard_parser("PPO Training for Heavy Trading Environment")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to training data (parquet or csv)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_TRAINING_CONFIG.symbol,
        help=CLIFormatter.format_help("Trading symbol", DEFAULT_TRAINING_CONFIG.symbol),
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=DEFAULT_TRAINING_CONFIG.venue,
        help=CLIFormatter.format_help("Trading venue", DEFAULT_TRAINING_CONFIG.venue),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "optimize", "visualize"],
        default="train",
        help=CLIFormatter.format_help(
            "Operation mode", "train", ["train", "evaluate", "optimize", "visualize"]
        ),
    )
    parser.add_argument("--model-path", type=str, help="Path to model for evaluation")
    parser.add_argument(
        "--n-trials",
        type=lambda x: CLIValidator.validate_positive_int(x, "n-trials"),
        default=50,
        help=CLIFormatter.format_help("Number of optimization trials", 50),
    )
    parser.add_argument(
        "--n-episodes",
        type=lambda x: CLIValidator.validate_positive_int(x, "n-episodes"),
        default=10,
        help=CLIFormatter.format_help("Number of evaluation episodes", 10),
    )

    args = parser.parse_args()

    # トレーナーの作成
    trainer = PPOTrainer(args.data)

    if args.mode == "train":
        # まずハイパーパラメータ最適化を実施
        best_params = optimize_hyperparameters(args.data, args.n_trials)
        # total_timesteps を除外してから config を更新
        best_params.pop("total_timesteps", None)
        trainer.config.update(best_params)
        trainer.config["total_timesteps"] = 200000  # 本番トレーニング
        # 本番トレーニングを実施
        _ = trainer.train()
        trainer.evaluate(n_episodes=args.n_episodes)
    elif args.mode == "optimize":
        best_params = optimize_hyperparameters(args.data, args.n_trials)
        # 最適パラメータでの最終トレーニング
        trainer.config.update(best_params)
        trainer.config["total_timesteps"] = 200000  # 本番トレーニング
        _ = trainer.train()

    elif args.mode == "visualize":
        trainer.visualize_training()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
