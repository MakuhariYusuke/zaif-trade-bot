# PPO Training Script for Heavy Trading Environment
# 重特徴量取引環境でのPPO学習

import atexit
import csv
import json
import logging
import os
import pickle
import sys
import threading
import time
import zlib
from collections import defaultdict, deque
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import BufferingHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, Optional, Union, cast

if TYPE_CHECKING:
    from ztb.utils import DiscordNotifier

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch.optim.lr_scheduler
from numpy.typing import NDArray
from optuna import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from ztb.utils.data_utils import load_csv_data


class EarlyStoppingCallback(BaseCallback):
    """Early stopping callback based on evaluation metric."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor_metric: str = "eval_reward",
    ):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_metric = monitor_metric
        self.best_score = -float("inf")
        self.wait = 0
        self.stopped_epoch = 0

    def _on_step(self) -> bool:
        # Check if we have evaluation info
        if hasattr(self.locals.get("self", None), "last_eval_info"):
            eval_info = self.locals["self"].last_eval_info
            current_score = eval_info.get(self.monitor_metric, -float("inf"))

            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.wait = 0
            else:
                self.wait += 1

            if self.wait >= self.patience:
                self.logger.info(
                    f"Early stopping at step {self.num_timesteps}. Best {self.monitor_metric}: {self.best_score}"
                )
                return False

        return True


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False

try:
    import pynvml  # type: ignore[import]

    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore[assignment]
    HAS_PYNVML = False

try:
    import lz4.frame as lz4_frame

    HAS_LZ4 = True
except ImportError:
    lz4_frame = None
    HAS_LZ4 = False

try:
    import zstandard as zstd

    HAS_ZSTD = True
except ImportError:
    zstd = None
    HAS_ZSTD = False

# ローカルモジュールのインポート
# sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # プロジェクトルートを追加
try:
    from ztb.training import ResumeHandler  # type: ignore[attr-defined]
    from ztb.training import (
        TrainingCheckpointConfig,
        TrainingCheckpointManager,
    )
except ImportError:
    ResumeHandler = None
    TrainingCheckpointConfig = None
    TrainingCheckpointManager = None

from ztb.training.evaluation_callback import DSREvaluationCallback
from ztb.utils.seed_manager import set_global_seed

if TYPE_CHECKING:
    from ztb.data.streaming_pipeline import StreamingPipeline

from ztb.utils.observability import setup_observability

from ..utils.cache.feature_cache import FeatureCache
from ..utils.memory.dtypes import downcast_df
from ..utils.perf.cpu_tune import apply_cpu_tuning
from .environment import HeavyTradingEnv

if HAS_PYNVML:

    def _safe_nvml_shutdown() -> None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    atexit.register(_safe_nvml_shutdown)
else:

    def _safe_nvml_shutdown() -> None:  # pragma: no cover - fallback
        return


# チェックポイント用グローバルスレッドプール
_checkpoint_executor: Optional[futures.ThreadPoolExecutor] = None


_compression_stats: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=5))
_compression_lock = threading.Lock()


def _get_checkpoint_executor() -> futures.ThreadPoolExecutor:
    """チェックポイント保存用のスレッドプールを取得（再利用）"""
    global _checkpoint_executor
    if _checkpoint_executor is None or _checkpoint_executor._shutdown:
        import concurrent.futures

        _checkpoint_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="checkpoint"
        )
    return _checkpoint_executor


def _submit_to_checkpoint_pool(job: Callable[[], None]) -> futures.Future[Any]:
    """チェックポイントジョブをスレッドプールに投入"""
    executor = _get_checkpoint_executor()
    return executor.submit(job)


def _as_bool(value: Any, default: bool = False) -> bool:
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


@dataclass
class ResourceSnapshot:
    timestamp: float
    step: Optional[int]
    cpu_pct: float
    process_rss_mb: float
    system_mem_pct: float
    gpu_util_pct: Optional[float] = None
    gpu_mem_pct: Optional[float] = None

    def postfix(self) -> Dict[str, str]:
        postfix: Dict[str, str] = {
            "cpu%": f"{self.cpu_pct:.1f}",
            "rssMB": f"{self.process_rss_mb:.1f}",
            "mem%": f"{self.system_mem_pct:.1f}",
        }
        if self.gpu_util_pct is not None:
            postfix["gpu%"] = f"{self.gpu_util_pct:.1f}"
        if self.gpu_mem_pct is not None:
            postfix["gpu_mem%"] = f"{self.gpu_mem_pct:.1f}"
        return postfix


class ResourceMonitor:
    def __init__(
        self,
        *,
        enabled: bool,
        interval: float = 10.0,
        log_path: Optional[str] = None,
        alert_thresholds: Optional[Dict[str, float]] = None,
        include_gpu: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.enabled = bool(enabled and HAS_PSUTIL)
        self.interval = max(1.0, float(interval))
        self.logger = logger
        self.process = psutil.Process(os.getpid()) if self.enabled else None
        if self.process is not None:
            try:
                self.process.cpu_percent(interval=None)
            except Exception:
                pass
        self.log_path = Path(log_path) if log_path else None
        self._csv_header_written = False
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_header_written = self.log_path.exists()
        self.include_gpu = bool(include_gpu and HAS_PYNVML)
        self._gpu_handles: list[int] = []
        if self.include_gpu:
            try:
                count = pynvml.nvmlDeviceGetCount()
                self._gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)
                ]
                if not self._gpu_handles:
                    self.include_gpu = False
            except Exception:
                self.include_gpu = False
                if self.logger:
                    self.logger.debug("GPU monitoring disabled", exc_info=True)
        self.alert_thresholds = {
            key: float(value)
            for key, value in (alert_thresholds or {}).items()
            if value is not None
        }
        self.alert_cooldown = float(self.alert_thresholds.pop("cooldown", 60.0))
        self._last_alert_ts: Dict[str, float] = {}
        self._metric_to_attr = {
            "cpu_pct": "cpu_pct",
            "rss_mb": "process_rss_mb",
            "memory_pct": "system_mem_pct",
            "gpu_util_pct": "gpu_util_pct",
            "gpu_mem_pct": "gpu_mem_pct",
        }
        self._last_sample_ts = 0.0

    def maybe_sample(self, step: Optional[int] = None) -> Optional[ResourceSnapshot]:
        if not self.enabled or self.process is None:
            return None

        now = time.time()
        if now - self._last_sample_ts < self.interval:
            return None
        self._last_sample_ts = now

        system_cpu_pct = psutil.cpu_percent() if HAS_PSUTIL else 0.0
        rss_mb = self.process.memory_info().rss / 1024 / 1024
        system_mem_pct = psutil.virtual_memory().percent if HAS_PSUTIL else 0.0
        gpu_util_pct: Optional[float] = None
        gpu_mem_pct: Optional[float] = None

        if self.include_gpu and self._gpu_handles:
            util_total = 0.0
            mem_total = 0.0
            for handle in self._gpu_handles:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                except Exception:
                    continue
                util_total += float(getattr(util, "gpu", 0.0))
                if getattr(mem, "total", 0):
                    mem_total += (mem.used / mem.total) * 100.0
            gpu_count = len(self._gpu_handles)
            if gpu_count > 0:
                gpu_util_pct = util_total / gpu_count
                gpu_mem_pct = mem_total / gpu_count

        snapshot = ResourceSnapshot(
            timestamp=now,
            step=step,
            cpu_pct=system_cpu_pct,
            process_rss_mb=rss_mb,
            system_mem_pct=system_mem_pct,
            gpu_util_pct=gpu_util_pct,
            gpu_mem_pct=gpu_mem_pct,
        )

        self._write_log(snapshot)
        self._check_alerts(snapshot)
        return snapshot

    def _write_log(self, snapshot: ResourceSnapshot) -> None:
        if self.log_path is None:
            return
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not self._csv_header_written:
                writer.writerow(
                    [
                        "timestamp",
                        "step",
                        "cpu_pct",
                        "rss_mb",
                        "memory_pct",
                        "gpu_util_pct",
                        "gpu_mem_pct",
                    ]
                )
                self._csv_header_written = True
            writer.writerow(
                [
                    f"{snapshot.timestamp:.3f}",
                    snapshot.step if snapshot.step is not None else "",
                    f"{snapshot.cpu_pct:.2f}",
                    f"{snapshot.process_rss_mb:.2f}",
                    f"{snapshot.system_mem_pct:.2f}",
                    (
                        ""
                        if snapshot.gpu_util_pct is None
                        else f"{snapshot.gpu_util_pct:.2f}"
                    ),
                    (
                        ""
                        if snapshot.gpu_mem_pct is None
                        else f"{snapshot.gpu_mem_pct:.2f}"
                    ),
                ]
            )

    def _check_alerts(self, snapshot: ResourceSnapshot) -> None:
        if not self.alert_thresholds or self.logger is None:
            return
        now = snapshot.timestamp
        for metric, threshold in self.alert_thresholds.items():
            if threshold <= 0:
                continue
            attr = self._metric_to_attr.get(metric)
            if not attr:
                continue
            value = getattr(snapshot, attr, None)
            if value is None or value < threshold:
                continue
            last_alert = self._last_alert_ts.get(metric, 0.0)
            if now - last_alert < self.alert_cooldown:
                continue
            self._last_alert_ts[metric] = now
            self.logger.warning(
                "Resource alert: %s=%.2f exceeded threshold %.2f (step=%s)",
                metric,
                value,
                threshold,
                snapshot.step,
            )

    def close(self) -> None:
        """Placeholder for interface compatibility."""
        return


def _estimate_model_size(model: BaseAlgorithm) -> int:
    """モデルのサイズを推定"""
    try:
        # policyのパラメータ数をカウント
        param_count = sum(p.numel() for p in model.policy.parameters())
        # float32を想定してサイズ計算（4 bytes per param）
        estimated_size = param_count * 4

        # 追加のオーバーヘッド（optimizer状態など）
        estimated_size = int(estimated_size * 1.5)

        return estimated_size
    except Exception:
        # 推定失敗時はデフォルトサイズ
        return 500 * 1024 * 1024  # 500MB


def _create_incremental_checkpoint(
    current: Dict[str, Any], previous: Dict[str, Any]
) -> Dict[str, Any]:
    """増分チェックポイントを作成（前回からの差分のみ）"""
    incremental = {"timestamp": current["timestamp"], "step": current["step"]}

    # state_dictの差分を計算
    for key in ["policy", "value_net"]:
        if key in current and key in previous:
            current_state = current[key]
            previous_state = previous[key]

            if isinstance(current_state, dict) and isinstance(previous_state, dict):
                diff_state = {}
                for param_name, param_tensor in current_state.items():
                    if param_name in previous_state:
                        prev_tensor = previous_state[param_name]
                        if not torch.equal(param_tensor, prev_tensor):
                            diff_state[param_name] = param_tensor
                    else:
                        diff_state[param_name] = param_tensor

                if diff_state:  # 差分がある場合のみ保存
                    incremental[key] = diff_state
        elif key in current:
            incremental[key] = current[key]

    # その他のフィールド
    for key in ["scaler"]:
        if key in current and current[key] != previous.get(key):
            incremental[key] = current[key]

    return incremental


def _get_file_size_mb(path: str) -> float:
    """ファイルサイズをMB単位で取得"""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def save_checkpoint_async(
    model: BaseAlgorithm,
    path: str,
    notifier: Optional[Any] = None,
    light_mode: bool = False,
    compressor: str = "auto",
    incremental: bool = False,
    previous_checkpoint: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> futures.Future[None]:
    """最適化された非同期チェックポイント保存（圧縮・増分・リトライ対応）

    Returns:
        futures.Future[None]: 完了待ちに使用できるFuture
    """

    def _job() -> None:
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # 実際のデータサイズを取得して動的圧縮選択
                if compressor == "auto":
                    # モデルサイズを推定
                    model_size = _estimate_model_size(model)
                    selected_compressor = _select_checkpoint_compressor(model_size)
                else:
                    selected_compressor = compressor

                if light_mode:
                    # 軽量モード: policy + value_net + scaler の最小保存セット
                    checkpoint_data = {
                        "policy": model.policy.state_dict(),
                        "value_net": (
                            model.value_net.state_dict()
                            if hasattr(model, "value_net")
                            else None
                        ),
                        "scaler": getattr(model, "scaler", None),
                        "timestamp": datetime.now().isoformat(),
                        "step": getattr(model, "num_timesteps", 0),
                    }

                    if incremental and previous_checkpoint:
                        # 増分チェックポイント: 前回からの差分のみ保存
                        checkpoint_data = _create_incremental_checkpoint(
                            checkpoint_data, previous_checkpoint
                        )

                    # pickle化して圧縮
                    import pickle

                    data = pickle.dumps(
                        checkpoint_data, protocol=pickle.HIGHEST_PROTOCOL
                    )
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
                    f"[CKPT] saved: {final_path} (compressor: {selected_compressor}, "
                    f"incremental: {incremental}, size: {_get_file_size_mb(final_path):.1f}MB)"
                )
                if notifier:
                    notifier.send_custom_notification(
                        "💾 Checkpoint Saved",
                        f"Saved to {final_path} ({_get_file_size_mb(final_path):.1f}MB)",
                        color=0x0000FF,
                    )
                return  # 成功したら終了

            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    delay = retry_delay * (
                        2 ** (retry_count - 1)
                    )  # エクスポネンシャルバックオフ
                    logging.warning(
                        f"[CKPT] save failed (attempt {retry_count}/{max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logging.exception(
                        f"[CKPT] save failed permanently after {max_retries} retries: {e}"
                    )
                    if notifier:
                        notifier.send_error_notification(
                            "Checkpoint Save Error",
                            f"Failed to save {path} after {max_retries} retries: {str(e)}",
                        )

    # バックグラウンドスレッドプールで実行（再利用可能なプールを使用）
    return _submit_to_checkpoint_pool(_job)


def _select_checkpoint_compressor(data_size_bytes: int) -> str:
    """チェックポイント用圧縮方式選択（CheckpointManager._select_compressorと共通ロジック）"""
    candidates = []
    if HAS_ZSTD:
        candidates.append("zstd")
    if HAS_LZ4:
        candidates.append("lz4")
    if not candidates:
        return "zlib"

    with _compression_lock:
        estimates = {}
        for algo in candidates:
            stats = _compression_stats.get(algo)
            if stats:
                mean_throughput = sum(stats) / len(stats)
                if mean_throughput > 0:
                    estimates[algo] = data_size_bytes / mean_throughput
    if estimates:
        return min(estimates, key=lambda k: estimates[k])

    # Fallback to size-based heuristic when no statistics are available yet
    if data_size_bytes > 100 * 1024 * 1024:  # > 100MB
        return "lz4" if "lz4" in candidates else candidates[0]
    return "zstd" if "zstd" in candidates else candidates[0]


def _record_compression_stat(
    compressor: str, original_size: int, duration: float
) -> None:
    if duration <= 0:
        return
    throughput = original_size / duration
    with _compression_lock:
        _compression_stats[compressor].append(throughput)


def _compress_data(data: bytes, compressor: str) -> bytes:
    """データ圧縮"""
    start = time.perf_counter()
    if compressor == "zstd" and HAS_ZSTD:
        compressed = zstd.ZstdCompressor(level=3).compress(data)
    elif compressor == "lz4" and HAS_LZ4:
        compressed = lz4_frame.compress(data)  # type: ignore[attr-defined]
    else:
        import zlib

        compressed = zlib.compress(data, 6)
        compressor = "zlib"
    duration = time.perf_counter() - start
    _record_compression_stat(compressor, len(data), duration)
    return compressed


def load_checkpoint(model: BaseAlgorithm, path: str, light_mode: bool = False) -> None:
    """チェックポイント読み込み（軽量/通常両対応）"""
    if light_mode:
        # 軽量チェックポイントの読み込み
        import pickle

        # 圧縮形式の自動判定
        if path.endswith("_light.zip"):
            data = Path(path).read_bytes()

            # 圧縮形式判定（簡易的）
            if data.startswith(b"\x28\xb5\x2f\xfd"):  # Zstd magic
                decompressed = (
                    zstd.ZstdDecompressor().decompress(data)
                    if HAS_ZSTD
                    else zlib.decompress(data)
                )
            elif data.startswith(b"\x04\x22\x4d\x18"):  # LZ4 magic
                decompressed = (
                    lz4_frame.decompress(data) if HAS_LZ4 else zlib.decompress(data)
                )
            else:
                decompressed = zlib.decompress(data)

            checkpoint_data = pickle.loads(decompressed)

            # モデルにロード
            model.policy.load_state_dict(checkpoint_data["policy"])
            if checkpoint_data.get("value_net") and hasattr(model, "value_net"):
                model.value_net.load_state_dict(checkpoint_data["value_net"])  # type: ignore[attr-defined]
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

    def __init__(self, eval_freq: int = 1000, verbose: int = 0) -> None:
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
                                logging.info(progress_msg)
                                return False  # 学習停止
                            else:
                                self.zero_trade_count = 0
        except Exception as e:
            logging.error(f"Error in safety callback: {e}")

        return True


class PPOTrainer(object):
    """
    PPO Trainer for 1M timestep training with evaluation gates and memory optimization.
    """

    def __init__(
        self,
        data_path: Optional[str],
        config: Dict[str, Any],
        checkpoint_interval: int = 10000,
        checkpoint_dir: Optional[str] = None,
        streaming_pipeline: Optional["StreamingPipeline"] = None,
        stream_batch_size: int = 256,
        notifier: Optional["DiscordNotifier"] = None,
        max_features: Optional[int] = None,
    ):
        """
        Initialize PPO trainer.

        Args:
            data_path: Path to training data
            config: Training configuration
            checkpoint_interval: Steps between checkpoints
            checkpoint_dir: Directory for checkpoints
            streaming_pipeline: Optional streaming pipeline
            stream_batch_size: Batch size for streaming
        """
        self.data_path = Path(data_path) if data_path is not None else None
        self.config = config
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Initialize evaluation gates
        from ztb.training.ppo_trainer import PPOTrainer as GateTrainer

        self.gate_trainer = GateTrainer(
            eval_gates=None,  # Will be set up later
            checkpoint_interval=checkpoint_interval,
        )

        # Training state
        self.model: Optional[PPO] = None
        self.env: Optional[Any] = None
        self.session_id: Optional[str] = None
        self.notifier = notifier

        # Streaming pipeline
        self.streaming_pipeline = streaming_pipeline
        self.stream_batch_size = stream_batch_size
        self.max_features = max_features
        self.previous_checkpoint: Optional[Dict[str, Any]] = None
        self.use_incremental_checkpoints = True

        # Memory optimization
        self.rewards_history: deque[float] = deque(maxlen=50000)
        self.steps_history: deque[int] = deque(maxlen=50000)

        # Online statistics
        self.reward_sum = 0.0
        self.reward_count = 0
        self.reward_mean = 0.0
        self.reward_m2 = 0.0

        self.logger = logging.getLogger(__name__)
        self.logger.info("PPOTrainer initialized with memory optimization")

        monitor_enabled = _as_bool(self.config.get("resource_monitor_enabled", True))
        monitor_interval = float(self.config.get("resource_monitor_interval", 15.0))
        monitor_log_path = self.config.get("resource_monitor_log_path")
        monitor_alerts = self.config.get("resource_alert_thresholds")
        if not isinstance(monitor_alerts, dict):
            monitor_alerts = {}
        monitor_include_gpu = _as_bool(
            self.config.get("resource_monitor_include_gpu", True)
        )
        self._resource_monitor = ResourceMonitor(
            enabled=monitor_enabled,
            interval=monitor_interval,
            log_path=monitor_log_path,
            alert_thresholds=monitor_alerts,
            include_gpu=monitor_include_gpu,
            logger=self.logger,
        )

    def train(
        self,
        session_id: str,
        max_timesteps: Optional[int] = None,
        resume_model: Optional[PPO] = None,
    ) -> Optional[PPO]:
        """
        Train PPO model with evaluation gates.

        Args:
            session_id: Training session identifier
            max_timesteps: Maximum timesteps for this training session (default: from config)
            resume_model: Existing model to resume training from (default: None)

        Returns:
            Trained PPO model
        """
        self.session_id = session_id
        self.logger.info(f"Starting training session: {session_id}")

        progress_bar = None
        try:
            # Set up environment
            self._setup_environment()
            assert self.env is not None, "Environment not set up"

            # Create or resume model
            if resume_model is not None:
                self.model = resume_model
                self.logger.info("Resuming training from existing model")
            else:
                self.model = PPO(
                    "MlpPolicy",
                    self.env,
                    learning_rate=self.config.get("learning_rate", 3e-4),
                    n_steps=self.config.get("n_steps", 2048),
                    batch_size=self.config.get("batch_size", 64),
                    n_epochs=self.config.get("n_epochs", 10),
                    gamma=self.config.get("gamma", 0.99),
                    gae_lambda=self.config.get("gae_lambda", 0.95),
                    clip_range=self.config.get("clip_range", 0.2),
                    ent_coef=self.config.get("ent_coef", 0.0),
                    vf_coef=self.config.get("vf_coef", 0.5),
                    max_grad_norm=self.config.get("max_grad_norm", 0.5),
                    tensorboard_log=self.config.get("tensorboard_log"),
                    verbose=self.config.get("verbose", 1),
                    seed=self.config.get("seed", 42),
                )

            # Set up learning rate scheduler (Cosine Annealing)
            lr_scheduler_type = self.config.get("lr_scheduler", "cosine")
            if lr_scheduler_type == "cosine":
                # Cosine Annealing scheduler
                total_steps = (
                    total_timesteps // self.model.n_steps
                )  # Number of policy updates
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.model.policy.optimizer,
                    T_max=total_steps,
                    eta_min=self.config.get("lr_min", 1e-6),
                )
                self.logger.info(
                    f"Using Cosine Annealing LR scheduler: {total_steps} steps, min_lr={self.config.get('lr_min', 1e-6)}"
                )
            else:
                self.lr_scheduler = None
                self.logger.info("Using constant learning rate")

            # Set up early stopping
            self.early_stopping = EarlyStoppingCallback(
                patience=self.config.get("early_stopping_patience", 10),
                min_delta=self.config.get("early_stopping_min_delta", 0.001),
                monitor_metric="eval_sharpe_ratio",
            )
            from ztb.training.eval_gates import EvalGates

            eval_gates = EvalGates()
            eval_gates.enable_all()
            self.gate_trainer.eval_gates = eval_gates

            # Training loop with gates
            total_timesteps = (
                max_timesteps
                if max_timesteps is not None
                else self.config.get("total_timesteps", 1000000)
            )
            eval_freq = self.config.get("eval_freq", 10000)

            # Initialize progress bar
            progress_bar = tqdm(
                total=total_timesteps,
                desc=f"Training {session_id}",
                unit="steps",
                unit_scale=True,
                ncols=120,
            )

            for step in range(0, total_timesteps, eval_freq):
                remaining_steps = min(eval_freq, total_timesteps - step)

                # Train for eval_freq steps
                self.model.learn(
                    total_timesteps=remaining_steps,
                    reset_num_timesteps=False,
                    tb_log_name=f"tb_{session_id}",
                    callback=self.early_stopping,
                )

                # Step learning rate scheduler
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    self.logger.debug(f"Learning rate updated to: {current_lr}")

                # Update progress
                current_step = self.model.num_timesteps
                self._update_progress(current_step)
                progress_bar.update(remaining_steps)

                if self._resource_monitor.enabled:
                    snapshot = self._resource_monitor.maybe_sample(current_step)
                    if snapshot and progress_bar is not None:
                        progress_bar.set_postfix(snapshot.postfix(), refresh=False)

                # Check evaluation gates
                self._check_gates(current_step)

                # Save checkpoint after each eval_freq steps
                if self.checkpoint_dir:
                    self._save_checkpoint(current_step)

                # Check if training should halt
                if self.gate_trainer.halt_reason:
                    self.logger.warning(
                        f"Training halted: {self.gate_trainer.halt_reason}"
                    )
                    break

            self.logger.info(f"Training completed: {session_id}")
            return self.model

        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
        finally:
            if progress_bar is not None:
                progress_bar.close()
            if self._resource_monitor is not None:
                self._resource_monitor.close()

    def _setup_environment(self) -> None:
        """Set up training environment."""
        from ztb.trading.environment import HeavyTradingEnv as TradingEnvironment

        if self.streaming_pipeline:
            # Use streaming environment
            self.env = TradingEnvironment(
                config=self.config,
                streaming_pipeline=self.streaming_pipeline,
                stream_batch_size=self.stream_batch_size,
            )
        else:
            # Use static data environment

            self.env = TradingEnvironment(
                df=load_csv_data(self.data_path) if self.data_path else None,
                config=self.config,
            )

        # Wrap with Monitor for logging
        from stable_baselines3.common.monitor import Monitor

        log_dir = self.config.get("log_dir", "./logs")
        self.env = Monitor(self.env, log_dir)

        # Wrap with DummyVecEnv
        from stable_baselines3.common.vec_env import DummyVecEnv

        self.env = DummyVecEnv([lambda: self.env])

    def _update_progress(self, current_step: int) -> None:
        """Update training progress with memory-efficient statistics."""
        # Get latest reward from environment
        if hasattr(self.env, "get_attr") and self.env is not None:
            try:
                stats_list = self.env.get_attr("get_statistics")
                if stats_list and callable(stats_list[0]):
                    stats = stats_list[0]()
                    latest_reward = stats.get("total_reward", 0.0)

                    # Update history and statistics
                    self.rewards_history.append(latest_reward)
                    self.steps_history.append(current_step)

                    # Update online statistics using Welford's algorithm
                    self.reward_count += 1
                    delta = latest_reward - self.reward_mean
                    self.reward_mean += delta / self.reward_count
                    delta2 = latest_reward - self.reward_mean
                    self.reward_m2 += delta * delta2

                    # Update gate trainer
                    self.gate_trainer.update_progress(latest_reward, current_step)

            except Exception as e:
                self.logger.debug(f"Failed to update progress: {e}")

        # Send periodic Discord notification
        if self.notifier and current_step % 10000 == 0 and current_step > 0:
            progress_percent = (
                current_step / self.config.get("total_timesteps", 1000000)
            ) * 100
            self.notifier.send_notification(
                title="📊 Training Progress Update",
                message=f"Session {self.session_id}: {current_step:,} steps completed",
                color="info",
                fields={
                    "Progress": f"{progress_percent:.1f}%",
                    "Current Step": f"{current_step:,}",
                    "Total Steps": f"{self.config.get('total_timesteps', 1000000):,}",
                    "Session ID": self.session_id or "Unknown",
                },
            )

    def _check_gates(self, current_step: int) -> None:
        """Check evaluation gates."""
        if self.gate_trainer.eval_gates and self.gate_trainer.eval_gates.enabled:
            stats = self.gate_trainer.get_reward_stats()
            _ = self.gate_trainer.eval_gates.evaluate_all(
                rewards=list(self.rewards_history),
                steps=list(self.steps_history),
                final_eval_reward=(
                    self.rewards_history[-1] if self.rewards_history else 0.0
                ),
                reward_stats=stats,
            )

            # Update gate trainer state
            # self.gate_trainer._check_gates_and_halt_if_needed()

    def _save_checkpoint(self, current_step: int) -> None:
        """Save training checkpoint asynchronously."""
        self.logger.info(f"Attempting to save checkpoint at step {current_step}")
        self.logger.info(
            f"checkpoint_dir: {self.checkpoint_dir}, model: {self.model is not None}, session_id: {self.session_id}"
        )

        if not self.checkpoint_dir or not self.model or not self.session_id:
            self.logger.warning("Checkpoint save skipped due to missing requirements")
            return

        checkpoint_path = (
            self.checkpoint_dir / self.session_id / f"checkpoint_{current_step}"
        )
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # 非同期でモデルを保存
        model_path = str(checkpoint_path / "model")
        save_checkpoint_async(
            self.model,
            model_path,
            notifier=self.notifier,
            light_mode=False,  # 完全保存
            compressor="auto",
            incremental=self.use_incremental_checkpoints,
            previous_checkpoint=self.previous_checkpoint,
            max_retries=3,
        )

        # 現在のチェックポイントを保存（次回の増分用）
        if self.use_incremental_checkpoints:
            try:
                current_checkpoint = {
                    "policy": self.model.policy.state_dict(),
                    "scaler": getattr(self.model, "scaler", None),
                    "timestamp": datetime.now().isoformat(),
                    "step": current_step,
                }

                # value_netが存在する場合のみ追加
                try:
                    if (
                        hasattr(self.model, "value_net")
                        and self.model.value_net is not None
                    ):
                        current_checkpoint["value_net"] = (
                            self.model.value_net.state_dict()
                        )
                except AttributeError:
                    pass  # value_netがない場合は無視

                self.previous_checkpoint = current_checkpoint
            except Exception as e:
                self.logger.warning(
                    f"Failed to store checkpoint for incremental saving: {e}"
                )

        # トレーニング状態は同期的に保存（軽量なので）
        state = {
            "current_step": current_step,
            "rewards_history": list(self.rewards_history),
            "steps_history": list(self.steps_history),
            "reward_stats": self.gate_trainer.get_reward_stats(),
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
        }

        import json

        state_path = checkpoint_path / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Checkpoint save initiated: {checkpoint_path} (async)")

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return self.gate_trainer.get_training_status()


class PPOTrainerLegacy:
    """PPOトレーニングマネージャー"""

    def __init__(
        self,
        data_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_interval: int = 10000,
        checkpoint_dir: str = "models/checkpoints",
        streaming_pipeline: Optional["StreamingPipeline"] = None,
        stream_batch_size: int = 256,
        max_features: Optional[int] = None,
    ) -> None:
        """PPOトレーニングマネージャー"""
        apply_cpu_tuning()

        if data_path is None and streaming_pipeline is None:
            raise ValueError("Either data_path or streaming_pipeline must be provided")

        self.streaming_pipeline = streaming_pipeline
        self.stream_batch_size = stream_batch_size
        self.max_features = max_features
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
                "last_batch_at": (
                    stats.last_batch_at.isoformat() if stats.last_batch_at else None
                ),
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
                    df = load_csv_data(file_path)
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
                df = load_csv_data(data_path)
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
            "feature_set": self.config.get(
                "feature_set", "full"
            ),  # 特徴量セットをconfigから取得
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
            max_features=self.max_features,
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
            monitor_df = load_csv_data(monitor_file, skiprows=header_lines)

            # プロットの作成
            _, axes = plt.subplots(2, 2, figsize=(15, 10))

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
        _ = trainer.train(session_id="tune")

        # 評価
        # eval_stats = trainer.evaluate(n_episodes=5)
        # return cast(float, eval_stats["mean_reward"])
        return 0.0  # Placeholder

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

    import argparse

    parser = argparse.ArgumentParser(
        description="PPO Training for Heavy Trading Environment"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to training data (parquet or csv)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC",
        help="Trading symbol",
    )
    parser.add_argument(
        "--venue",
        type=str,
        default="binance",
        help="Trading venue",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "optimize", "visualize"],
        default="train",
        help="Operation mode",
    )
    parser.add_argument("--model-path", type=str, help="Path to model for evaluation")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )

    args = parser.parse_args()

    import json

    # Load config from scalping-config.json
    with open("scalping-config.json", "r") as f:
        full_config = json.load(f)
    config = full_config.get("environment", {})

    # トレーナーの作成
    trainer = PPOTrainer(
        data_path=args.data, config=config, checkpoint_dir="checkpoints/main"
    )

    if args.mode == "train":
        # 直接トレーニングを実施
        trainer.config["total_timesteps"] = 100000  # 本番トレーニング
        # 本番トレーニングを実施
        _ = trainer.train(session_id="main")
        # trainer.evaluate(n_episodes=args.n_episodes)
    elif args.mode == "optimize":
        # TODO: 最適化は未実装
        pass

    elif args.mode == "visualize":
        # trainer.visualize_training()
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
