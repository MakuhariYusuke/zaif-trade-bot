"""
Checkpoint management for experiments.

Provides async saving, compression, generation management, and auto-recovery.

Usage:
    from ztb.utils.checkpoint import CheckpointManager

    manager = CheckpointManager(save_dir="models/checkpoints", keep_last=5, compress="zstd")
    manager.save_async(model, step=1000)
"""

import logging
import pickle
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm

from ztb.types.common import ConfigDict
from ztb.utils.path_utils import ensure_dir

logger = logging.getLogger(__name__)

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
    zstd = None  # type: ignore
    HAS_ZSTD = False

import zlib


class CheckpointData(TypedDict):
    """Checkpoint data structure"""

    obj: Any
    step: int
    metadata: Dict[str, Any]
    timestamp: float
    training_context: Dict[str, Any]  # Enhanced metadata for training context


class TrainingStateCheckpointData(TypedDict):
    """Extended checkpoint data structure for complete training state"""

    # Model and training state
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    replay_buffer_state: Optional[Dict[str, Any]]

    # Training progress
    total_timesteps: int
    episode_count: int
    episode_rewards: List[float]
    episode_lengths: List[int]

    # Random state for reproducibility
    random_state: Tuple[Any, Any, Any]  # (random, numpy, torch) states

    # Training configuration
    config: ConfigDict

    # Metadata
    timestamp: float
    training_time: float
    version: str


class CheckpointManager:
    """Unified checkpoint manager with async saving and generation management"""

    def __init__(
        self,
        save_dir: str = "models/checkpoints",
        keep_last: int = 5,
        keep_every_nth: int = 10,
        compress: str = "zlib",
        max_queue_size: int = 10,
        differential: bool = False,
    ):
        self.save_dir = Path(save_dir)
        ensure_dir(self.save_dir)
        self.keep_last = keep_last
        self.keep_every_nth = keep_every_nth
        self.compress = compress
        self.max_queue_size = max_queue_size
        self.differential = differential

        # For differential checkpoints
        self.last_full_checkpoint: Optional[CheckpointData] = None
        self.last_checkpoint_path: Optional[str] = None

        # Async saving queue
        self.save_queue: Queue[Optional[Dict[str, Any]]] = Queue(maxsize=max_queue_size)
        self.worker_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.worker_thread.start()

        # Stats
        self.stats = {
            "saved_count": 0,
            "compressed_size_mb": 0.0,
            "total_save_time": 0.0,
            "differential_saves": 0,
        }

    def save_async(
        self,
        obj: Any,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
        training_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save checkpoint asynchronously"""
        if self.save_queue.full():
            print(f"Warning: Checkpoint queue full, dropping save at step {step}")
            return

        self.save_queue.put(
            {
                "obj": obj,
                "step": step,
                "metadata": metadata or {},
                "timestamp": time.time(),
                "training_context": training_context or {},
            }
        )

    def save_sync(
        self,
        obj: Any,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
        training_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save checkpoint synchronously (blocking)"""
        return self._save_checkpoint(obj, step, metadata or {}, training_context or {})

    def load_latest(self) -> Tuple[CheckpointData, int, Dict[str, Any]]:
        """Load the latest checkpoint with differential support"""
        checkpoints = list(self.save_dir.glob("checkpoint*.pkl*"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")

        # Sort by step number (handle both full and diff checkpoints)
        def get_step(path: Path) -> int:
            stem = path.stem
            if stem.startswith("checkpoint_diff_"):
                return int(stem.split("_")[2])
            else:
                return int(stem.split("_")[1])

        latest = max(checkpoints, key=get_step)

        # If loading a diff checkpoint, we need to find the base
        if "diff" in latest.name:
            # Find the most recent full checkpoint before this diff
            diff_step = get_step(latest)
            full_checkpoints = [
                cp
                for cp in checkpoints
                if "diff" not in cp.name and get_step(cp) <= diff_step
            ]
            if full_checkpoints:
                base_checkpoint = max(full_checkpoints, key=get_step)
                self.last_full_checkpoint = self._load_raw_checkpoint(
                    str(base_checkpoint)
                )

        return self._load_checkpoint(str(latest))

    def _load_raw_checkpoint(self, path: str) -> CheckpointData:
        """Load checkpoint without applying diffs"""
        try:
            with open(path, "rb") as f:
                compressed_data = f.read()
        except Exception as e:
            logger.error(f"Failed to read checkpoint file {path}: {e}")
            raise

        try:
            data = self._decompress_data(compressed_data)
        except Exception as e:
            logger.error(f"Failed to decompress checkpoint {path}: {e}")
            raise

        try:
            return cast(CheckpointData, pickle.loads(data))
        except Exception as e:
            logger.error(f"Failed to deserialize checkpoint {path}: {e}")
            raise

    def cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints based on keep_last and keep_every_nth"""
        checkpoints = sorted(
            self.save_dir.glob("checkpoint_*.pkl*"), key=lambda p: self._extract_step(p)
        )

        if len(checkpoints) <= self.keep_last:
            return

        # Keep last N
        to_keep = set(checkpoints[-self.keep_last :])

        # Keep every Nth
        for i, ckpt in enumerate(checkpoints[: -self.keep_last]):
            if (i + 1) % self.keep_every_nth == 0:
                to_keep.add(ckpt)

        # Remove others
        for ckpt in checkpoints:
            if ckpt not in to_keep:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt.name}")

    def _save_worker(self) -> None:
        """Background worker for async saves"""
        import gc
        import traceback

        from ztb.utils.notify import get_notifier

        notifier = get_notifier()
        logged_errors = set()  # Track logged errors to prevent spam

        while True:
            item = self.save_queue.get()
            if item is None:  # Poison pill
                break

            try:
                start_time = time.time()
                path = self._save_checkpoint(
                    item["obj"], item["step"], item["metadata"]
                )
                save_time = time.time() - start_time

                self.stats["saved_count"] += 1
                self.stats["total_save_time"] += save_time

                print(f"Async checkpoint saved: {path} ({save_time:.2f}s)")

                # Memory cleanup
                gc.collect()

                # Cleanup periodically
                if self.stats["saved_count"] % 10 == 0:
                    self.cleanup_old_checkpoints()

                self.save_queue.task_done()

            except Exception as e:
                error_msg = f"Error in checkpoint worker: {type(e).__name__}: {e}"
                error_trace = traceback.format_exc()

                # Log full error with traceback
                print(error_msg)
                print(error_trace)

                # Send to Discord if not already logged this error
                error_key = f"{type(e).__name__}:{str(e)}"
                if error_key not in logged_errors:
                    logged_errors.add(error_key)
                    if notifier:
                        notifier.send_notification(
                            title="Checkpoint Worker Error",
                            message=f"{error_msg}\n\n{error_trace[:1000]}...",  # Limit traceback length
                            color="error",
                        )

    def _save_checkpoint(
        self,
        obj: Any,
        step: int,
        metadata: Dict[str, Any],
        training_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Internal save method with differential support"""
        checkpoint_data: CheckpointData = {
            "obj": obj,
            "step": step,
            "metadata": metadata,
            "timestamp": time.time(),
            "training_context": training_context or {},
        }

        # Check if we should save differentially
        if (
            self.differential
            and self.last_full_checkpoint is not None
            and step % 10 != 0
        ):  # Save full every 10 steps
            # Create differential checkpoint
            diff_data = self._compute_diff(self.last_full_checkpoint, checkpoint_data)
            save_data = diff_data
            filename = f"checkpoint_diff_{step:08d}.pkl"
            self.stats["differential_saves"] += 1
        else:
            # Full checkpoint
            save_data = checkpoint_data
            filename = f"checkpoint_{step:08d}.pkl"
            self.last_full_checkpoint = checkpoint_data
            self.last_checkpoint_path = None  # Reset diff chain

        # Serialize
        data = pickle.dumps(save_data, protocol=pickle.HIGHEST_PROTOCOL)

        # Compress
        compressed_data = self._compress_data(data)

        # Save
        if self.compress == "zstd":
            filename += ".zst"
        elif self.compress == "lz4":
            filename += ".lz4"

        path = self.save_dir / filename
        self.last_checkpoint_path = str(path)

        with open(path, "wb") as f:
            f.write(compressed_data)

        # Update stats
        self.stats["compressed_size_mb"] += len(compressed_data) / 1024 / 1024

        return str(path)

    def _load_checkpoint(self, path: str) -> Tuple[CheckpointData, int, Dict[str, Any]]:
        """Internal load method with differential support"""
        try:
            with open(path, "rb") as f:
                compressed_data = f.read()
        except Exception as e:
            logger.error(f"Failed to read checkpoint file {path}: {e}")
            raise

        try:
            data = self._decompress_data(compressed_data)
        except Exception as e:
            logger.error(f"Failed to decompress checkpoint {path}: {e}")
            raise

        try:
            checkpoint_data = pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize checkpoint {path}: {e}")
            raise

        # If this is a differential checkpoint, apply it to base
        if isinstance(checkpoint_data, dict) and checkpoint_data.get("is_differential"):
            if self.last_full_checkpoint is None:
                raise ValueError(
                    "Cannot load differential checkpoint without base checkpoint"
                )
            checkpoint_data = self._apply_diff(
                self.last_full_checkpoint, checkpoint_data
            )

        return (
            checkpoint_data["obj"],
            checkpoint_data["step"],
            checkpoint_data["metadata"],
        )

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data based on compression setting"""
        compressor = self._select_compressor(len(data))
        return self._apply_compression(data, compressor)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data"""
        # Try different decompressors (data might have been saved with different compression)
        for decompressor in [
            self._decompress_zstd,
            self._decompress_lz4,
            self._decompress_zlib,
        ]:
            try:
                return decompressor(data)
            except Exception:
                continue
        raise ValueError("Could not decompress data")

    def _select_compressor(self, data_size: int) -> str:
        """Select compression algorithm"""
        if self.compress != "auto":
            return self.compress

        # Auto selection based on size
        if data_size > 100 * 1024 * 1024:  # > 100MB
            return "lz4" if HAS_LZ4 else "zlib"
        else:
            return "zstd" if HAS_ZSTD else "lz4" if HAS_LZ4 else "zlib"

    def _apply_compression(self, data: bytes, compressor: str) -> bytes:
        """Apply compression"""
        if compressor == "lz4" and HAS_LZ4 and lz4_frame:
            return cast(bytes, lz4_frame.compress(data))
        elif compressor == "zstd" and HAS_ZSTD and zstd:
            compressor_obj = zstd.ZstdCompressor()
            return cast(bytes, compressor_obj.compress(data))
        else:
            return zlib.compress(data, 6)

    def _decompress_lz4(self, data: bytes) -> bytes:
        if HAS_LZ4 and lz4_frame:
            return cast(bytes, lz4_frame.decompress(data))
        raise ImportError("lz4 not available")

    def _decompress_zstd(self, data: bytes) -> bytes:
        if HAS_ZSTD and zstd:
            decompressor = zstd.ZstdDecompressor()
            return cast(bytes, decompressor.decompress(data))
        raise ImportError("zstd not available")

    def _decompress_zlib(self, data: bytes) -> bytes:
        return zlib.decompress(data)

    def _extract_step(self, path: Path) -> int:
        """Extract step number from checkpoint filename"""
        name = path.stem  # checkpoint_00001234 or checkpoint_00001234.pkl
        # Handle both compressed and uncompressed filenames
        if name.endswith(".pkl"):
            name = name[:-4]  # Remove .pkl extension
        step_str = name.split("_")[1]
        # Remove any remaining extension
        if "." in step_str:
            step_str = step_str.split(".")[0]
        return int(step_str)

    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        return self.stats.copy()

    def shutdown(self) -> None:
        """Shutdown the manager"""
        self.save_queue.put(None)  # Poison pill
        self.worker_thread.join(timeout=5)

    def _compute_diff(
        self, old_data: CheckpointData, new_data: CheckpointData
    ) -> Dict[str, Any]:
        """Compute differential checkpoint data"""
        diff: Dict[str, Any] = {
            "step": new_data["step"],
            "timestamp": new_data["timestamp"],
        }

        # Compute diff for metadata
        if old_data.get("metadata") != new_data.get("metadata"):
            diff["metadata"] = new_data["metadata"]

        # For objects, if they are dicts, compute nested diff
        if isinstance(old_data.get("obj"), dict) and isinstance(
            new_data.get("obj"), dict
        ):
            diff["obj_diff"] = self._dict_diff(old_data["obj"], new_data["obj"])
            diff["is_differential"] = True
        else:
            # Fallback to full save if not dict
            diff["obj"] = new_data["obj"]
            diff["is_differential"] = False

        return diff

    def _dict_diff(
        self, old_dict: Dict[str, Any], new_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute diff between two dictionaries (recursive)"""
        diff = {}

        # Find changed/added keys
        for key, new_value in new_dict.items():
            if key not in old_dict:
                diff[key] = ("added", new_value)
            elif old_dict[key] != new_value:
                if isinstance(old_dict[key], dict) and isinstance(new_value, dict):
                    nested_diff = self._dict_diff(old_dict[key], new_value)
                    if nested_diff:
                        diff[key] = ("modified", nested_diff)
                else:
                    diff[key] = ("modified", new_value)

        # Find removed keys
        for key in old_dict:
            if key not in new_dict:
                diff[key] = ("removed", None)

        return diff

    def _apply_diff(
        self, base_data: CheckpointData, diff: Dict[str, Any]
    ) -> CheckpointData:
        """Apply differential data to base checkpoint"""
        result = base_data.copy()
        result["step"] = diff["step"]
        result["timestamp"] = diff["timestamp"]

        if "metadata" in diff:
            result["metadata"] = diff["metadata"]

        if "obj_diff" in diff:
            # Apply dict diff
            result["obj"] = self._apply_dict_diff(base_data["obj"], diff["obj_diff"])
        elif "obj" in diff:
            # Full object replacement
            result["obj"] = diff["obj"]

        return result

    def _apply_dict_diff(
        self, base_dict: Dict[str, Any], diff: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply dictionary diff to base dict (recursive)"""
        result = base_dict.copy()

        for key, (action, value) in diff.items():
            if action == "added" or action == "modified":
                if isinstance(value, dict) and action == "modified":
                    # Nested diff
                    if key in result and isinstance(result[key], dict):
                        result[key] = self._apply_dict_diff(result[key], value)
                    else:
                        result[key] = value
                else:
                    result[key] = value
            elif action == "removed":
                result.pop(key, None)

        return result


class HierarchicalCheckpointManager:
    """
    Hierarchical checkpoint manager for ML training.

    Manages three levels of checkpoints:
    - light: Frequent lightweight checkpoints (every 1k/5k steps)
    - full: Full model checkpoints (every 10k steps)
    - archive: Long-term archive checkpoints (every 50k steps)

    Recovery policy: On failure, resume from nearest full checkpoint.
    """

    def __init__(
        self,
        save_dir: str = "models/checkpoints",
        compress: str = "zstd",
        light_freq: Optional[List[int]] = None,
        full_freq: int = 10000,
        archive_freq: int = 50000,
    ):
        self.save_dir = Path(save_dir)
        ensure_dir(self.save_dir)
        self.compress = compress

        # Checkpoint frequencies
        self.light_freq = light_freq or [1000, 5000]  # Save at 1k and every 5k after
        self.full_freq = full_freq
        self.archive_freq = archive_freq

        # Keep policy: 3 generations for light/full, all for archive
        self.keep_light = 3
        self.keep_full = 3
        self.keep_archive = -1  # -1 means keep all

        # Async saving
        self.executor: Optional[ThreadPoolExecutor] = None
        self._init_executor()

    def _init_executor(self) -> None:
        """Initialize ThreadPoolExecutor for async saving"""
        import os

        max_workers = min(4, os.cpu_count() or 2)
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="checkpoint"
        )

    def should_save_light(self, step: int) -> bool:
        """Check if light checkpoint should be saved at this step"""
        if step in self.light_freq:
            return True
        if step > max(self.light_freq):
            return (step - max(self.light_freq)) % self.light_freq[1] == 0
        return False

    def should_save_full(self, step: int) -> bool:
        """Check if full checkpoint should be saved at this step"""
        return step % self.full_freq == 0

    def should_save_archive(self, step: int) -> bool:
        """Check if archive checkpoint should be saved at this step"""
        return step % self.archive_freq == 0

    def save_checkpoint(
        self,
        step: int,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        checkpoint_type: str = "auto",
    ) -> None:
        """
        Save checkpoint asynchronously.

        Args:
            step: Current training step
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            metrics: Training metrics
            checkpoint_type: 'light', 'full', 'archive', or 'auto'
        """
        if checkpoint_type == "auto":
            if self.should_save_archive(step):
                checkpoint_type = "archive"
            elif self.should_save_full(step):
                checkpoint_type = "full"
            elif self.should_save_light(step):
                checkpoint_type = "light"
            else:
                return  # No checkpoint needed

        # Submit async save
        if self.executor:
            self.executor.submit(
                self._save_checkpoint_sync,
                step,
                model_state,
                optimizer_state,
                metrics,
                checkpoint_type,
            )

    def _save_checkpoint_sync(
        self,
        step: int,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]],
        checkpoint_type: str,
    ) -> None:
        """Synchronous checkpoint saving"""
        try:
            import time

            start_time = time.time()

            # Prepare checkpoint data
            checkpoint_data = {
                "step": step,
                "model_state": model_state,
                "optimizer_state": optimizer_state,
                "metrics": metrics or {},
                "timestamp": datetime.now().isoformat(),
                "type": checkpoint_type,
            }

            # Create filename
            filename = f"checkpoint_{checkpoint_type}_{step:010d}.pkl"
            if self.compress == "zstd":
                filename += ".zst"
            elif self.compress == "lz4":
                filename += ".lz4"

            filepath = self.save_dir / filename

            # Serialize and compress
            data = pickle.dumps(checkpoint_data)
            uncompressed_size = len(data)

            if self.compress == "zstd" and HAS_ZSTD and zstd:
                compressor = zstd.ZstdCompressor()
                compressed_data = compressor.compress(data)
            elif self.compress == "lz4" and HAS_LZ4 and lz4_frame:
                compressed_data = lz4_frame.compress(data)
            else:
                compressed_data = zlib.compress(data, 6)

            # Save
            with open(filepath, "wb") as f:
                f.write(compressed_data)

            save_time = time.time() - start_time
            compression_ratio = (
                len(compressed_data) / uncompressed_size
                if uncompressed_size > 0
                else 1.0
            )

            print(
                f"Saved {checkpoint_type} checkpoint at step {step}: {filepath} "
                f"({save_time:.2f}s, {compression_ratio:.2f}x compression)"
            )

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(checkpoint_type)

        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def _cleanup_old_checkpoints(self, checkpoint_type: str) -> None:
        """Clean up old checkpoints according to retention policy"""
        if checkpoint_type == "light":
            # Keep specified number of light checkpoints
            light_checkpoints = list(self.save_dir.glob("checkpoint_light_*.pkl*"))
            if len(light_checkpoints) > self.keep_light:
                # Sort by step number (descending)
                light_checkpoints.sort(
                    key=lambda x: int(x.stem.split("_")[2]), reverse=True
                )
                for old_cp in light_checkpoints[self.keep_light :]:
                    old_cp.unlink()

        elif checkpoint_type == "full":
            # Keep specified number of full checkpoints
            full_checkpoints = list(self.save_dir.glob("checkpoint_full_*.pkl*"))
            if len(full_checkpoints) > self.keep_full:
                # Sort by step number (descending)
                full_checkpoints.sort(
                    key=lambda x: int(x.stem.split("_")[2]), reverse=True
                )
                for old_cp in full_checkpoints[self.keep_full :]:
                    old_cp.unlink()

        elif checkpoint_type == "archive":
            # Keep all archive checkpoints (no cleanup)
            pass

    def find_recovery_checkpoint(self) -> Optional[Path]:
        """
        Find the best checkpoint for recovery.
        Priority: latest full > latest archive > latest light
        """
        # Try full checkpoints first
        full_cps = list(self.save_dir.glob("checkpoint_full_*.pkl*"))
        if full_cps:
            return max(full_cps, key=lambda x: int(x.stem.split("_")[2]))

        # Then archive
        archive_cps = list(self.save_dir.glob("checkpoint_archive_*.pkl*"))
        if archive_cps:
            return max(archive_cps, key=lambda x: int(x.stem.split("_")[2]))

        # Finally light
        light_cps = list(self.save_dir.glob("checkpoint_light_*.pkl*"))
        if light_cps:
            return max(light_cps, key=lambda x: int(x.stem.split("_")[2]))

        return None

    def load_checkpoint(
        self, checkpoint_path: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from file.

        Args:
            checkpoint_path: Specific checkpoint path, or None to auto-find recovery checkpoint

        Returns:
            Checkpoint data dict, or None if not found
        """
        if checkpoint_path is None:
            checkpoint_path = self.find_recovery_checkpoint()

        if checkpoint_path is None or not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "rb") as f:
                compressed_data = f.read()

            # Decompress
            if checkpoint_path.suffix == ".zst" and HAS_ZSTD and zstd:
                decompressor = zstd.ZstdDecompressor()
                data = decompressor.decompress(compressed_data)
            elif checkpoint_path.suffix == ".lz4" and HAS_LZ4 and lz4_frame:
                data = lz4_frame.decompress(compressed_data)
            else:
                # Assume zlib
                data = zlib.decompress(compressed_data)

            # Deserialize
            checkpoint_data = pickle.loads(data)
            print(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint_data

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        light_count = len(list(self.save_dir.glob("checkpoint_light_*.pkl*")))
        full_count = len(list(self.save_dir.glob("checkpoint_full_*.pkl*")))
        archive_count = len(list(self.save_dir.glob("checkpoint_archive_*.pkl*")))

        return {
            "light_checkpoints": light_count,
            "full_checkpoints": full_count,
            "archive_checkpoints": archive_count,
            "total_checkpoints": light_count + full_count + archive_count,
        }

    def shutdown(self) -> None:
        """Shutdown the manager"""
        if self.executor:
            self.executor.shutdown(wait=True)


class TrainingStateManager:
    """Manager for saving and loading complete training state for resume functionality"""

    def __init__(
        self, save_dir: str = "models/training_states", compress: str = "zstd"
    ):
        self.save_dir = Path(save_dir)
        ensure_dir(self.save_dir)
        self.compress = compress

    def save_training_state(
        self,
        model: BaseAlgorithm,
        total_timesteps: int,
        episode_count: int = 0,
        episode_rewards: Optional[List[float]] = None,
        episode_lengths: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None,
        training_time: float = 0.0,
        filename: Optional[str] = None,
    ) -> str:
        """Save complete training state for later resumption"""

        # Capture random states for reproducibility
        random_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state()
            if torch.cuda.is_available()
            else torch.random.get_rng_state(),
        )

        # Extract model state
        model_state = {
            "policy": model.policy.state_dict(),
            "policy_kwargs": getattr(model, "policy_kwargs", {}),
        }

        # Extract optimizer state if available
        optimizer_state = {}
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            optimizer_state = model.policy.optimizer.state_dict()

        # Extract replay buffer state if available
        replay_buffer_state = None
        if hasattr(model, "replay_buffer") and model.replay_buffer is not None:
            try:
                replay_buffer_state = model.replay_buffer.__dict__.copy()
            except:
                logger.warning("Could not save replay buffer state")

        # Prepare training state data
        training_state: TrainingStateCheckpointData = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "replay_buffer_state": replay_buffer_state,
            "total_timesteps": total_timesteps,
            "episode_count": episode_count,
            "episode_rewards": episode_rewards or [],
            "episode_lengths": episode_lengths or [],
            "random_state": random_state,
            "config": config or {},
            "timestamp": time.time(),
            "training_time": training_time,
            "version": "1.0",
        }

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_state_{total_timesteps}_{timestamp}.pkl"

        filepath = self.save_dir / filename

        # Compress and save
        compressed_data = self._compress_data(training_state)
        with open(filepath, "wb") as f:
            f.write(compressed_data)

        logger.info(f"Saved training state to {filepath}")
        return str(filepath)

    def load_training_state(self, filepath: str) -> TrainingStateCheckpointData:
        """Load complete training state for resumption"""

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Training state file not found: {filepath}")

        # Load and decompress
        with open(filepath, "rb") as f:
            compressed_data = f.read()

        training_state = self._decompress_data(compressed_data)
        logger.info(f"Loaded training state from {filepath}")
        return training_state

    def restore_training_state(
        self, model: BaseAlgorithm, training_state: TrainingStateCheckpointData
    ) -> None:
        """Restore training state to model"""

        # Restore model state
        if "policy" in training_state["model_state"]:
            model.policy.load_state_dict(training_state["model_state"]["policy"])

        # Restore optimizer state
        if training_state["optimizer_state"] and hasattr(model.policy, "optimizer"):
            model.policy.optimizer.load_state_dict(training_state["optimizer_state"])

        # Restore replay buffer state (limited support)
        if (
            training_state["replay_buffer_state"]
            and hasattr(model, "replay_buffer")
            and model.replay_buffer is not None
        ):
            try:
                # This is a simplified restoration - full restoration would require
                # more complex buffer state management
                for key, value in training_state["replay_buffer_state"].items():
                    if hasattr(model.replay_buffer, key):
                        setattr(model.replay_buffer, key, value)
                logger.info("Restored replay buffer state")
            except Exception as e:
                logger.warning(f"Could not restore replay buffer state: {e}")

        # Restore random states
        random_state = training_state["random_state"]
        random.setstate(random_state[0])
        np.random.set_state(random_state[1])
        if torch.cuda.is_available():
            torch.set_rng_state(random_state[2])
        else:
            torch.random.set_rng_state(random_state[2])

        logger.info("Restored training state to model")

    def _compress_data(self, data: TrainingStateCheckpointData) -> bytes:
        """Compress training state data"""
        pickled_data = pickle.dumps(data)

        if self.compress == "zstd" and HAS_ZSTD:
            compressor = zstd.ZstdCompressor()
            return compressor.compress(pickled_data)
        elif self.compress == "lz4" and HAS_LZ4:
            return lz4_frame.compress(pickled_data)
        else:
            return zlib.compress(pickled_data)

    def _decompress_data(self, compressed_data: bytes) -> TrainingStateCheckpointData:
        """Decompress training state data"""
        if self.compress == "zstd" and HAS_ZSTD:
            decompressor = zstd.ZstdDecompressor()
            data = decompressor.decompress(compressed_data)
        elif self.compress == "lz4" and HAS_LZ4:
            data = lz4_frame.decompress(compressed_data)
        else:
            data = zlib.decompress(compressed_data)

        return pickle.loads(data)

    def list_training_states(self) -> List[Dict[str, Any]]:
        """List all saved training states with metadata"""
        states = []
        for filepath in self.save_dir.glob("training_state_*.pkl*"):
            try:
                # Quick load just metadata without full decompression
                with open(filepath, "rb") as f:
                    compressed_data = f.read()
                training_state = self._decompress_data(compressed_data)

                states.append(
                    {
                        "filepath": str(filepath),
                        "total_timesteps": training_state["total_timesteps"],
                        "episode_count": training_state["episode_count"],
                        "timestamp": training_state["timestamp"],
                        "training_time": training_state["training_time"],
                        "version": training_state["version"],
                    }
                )
            except Exception as e:
                logger.warning(f"Could not read training state {filepath}: {e}")

        return sorted(states, key=lambda x: x["timestamp"], reverse=True)

    def validate_resume_compatibility(
        self,
        training_state: TrainingStateCheckpointData,
        current_config: Dict[str, Any],
        data_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate compatibility between saved training state and current setup"""

        validation_results = {"compatible": True, "warnings": [], "errors": []}

        # Check version compatibility
        saved_version = training_state.get("version", "0.0")
        if saved_version != "1.0":
            validation_results["warnings"].append(
                f"Version mismatch: saved={saved_version}, current=1.0"
            )

        # Check configuration compatibility
        saved_config = training_state.get("config", {})

        # Check critical hyperparameters
        critical_params = ["learning_rate", "batch_size", "buffer_size", "gamma", "tau"]
        for param in critical_params:
            saved_val = self._get_nested_value(
                saved_config, f"training.sac_hyperparameters.{param}"
            )
            current_val = self._get_nested_value(
                current_config, f"training.sac_hyperparameters.{param}"
            )

            if (
                saved_val is not None
                and current_val is not None
                and saved_val != current_val
            ):
                validation_results["warnings"].append(
                    f"Hyperparameter mismatch for {param}: saved={saved_val}, current={current_val}"
                )

        # Check environment configuration
        saved_env = self._get_nested_value(saved_config, "training.environment_config")
        current_env = self._get_nested_value(
            current_config, "training.environment_config"
        )

        if saved_env and current_env:
            # Check critical environment parameters
            env_params = ["window_size", "fee", "leverage"]
            for param in env_params:
                if saved_env.get(param) != current_env.get(param):
                    validation_results["errors"].append(
                        f"Environment parameter mismatch for {param}: saved={saved_env.get(param)}, current={current_env.get(param)}"
                    )

        # Check data compatibility (if data_path provided)
        if data_path and os.path.exists(data_path):
            try:
                # This would require loading a small sample of data to check compatibility
                # For now, just check if data file exists and is readable
                with open(data_path, "r") as f:
                    # Try to read first few lines
                    lines = []
                    for i, line in enumerate(f):
                        if i >= 5:  # Check first 5 lines
                            break
                        lines.append(line.strip())

                if not lines:
                    validation_results["errors"].append("Data file appears to be empty")

            except Exception as e:
                validation_results["errors"].append(f"Cannot read data file: {e}")

        # Determine overall compatibility
        if validation_results["errors"]:
            validation_results["compatible"] = False

        return validation_results

    def _get_nested_value(self, config: ConfigDict, path: str) -> Any:
        """Get nested value from config using dot notation"""
        keys = path.split(".")
        value = config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
