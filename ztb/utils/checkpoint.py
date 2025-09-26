"""
Checkpoint management for experiments.

Provides async saving, compression, generation management, and auto-recovery.

Usage:
    from ztb.utils.checkpoint import CheckpointManager

    manager = CheckpointManager(save_dir="models/checkpoints", keep_last=5, compress="zstd")
    manager.save_async(model, step=1000)
"""

import asyncio
import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

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

import zlib


class CheckpointManager:
    """Unified checkpoint manager with async saving and generation management"""

    def __init__(self, save_dir: str = "models/checkpoints", keep_last: int = 5,
                 keep_every_nth: int = 10, compress: str = "zlib", max_queue_size: int = 10):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.keep_every_nth = keep_every_nth
        self.compress = compress
        self.max_queue_size = max_queue_size

        # Async saving queue
        self.save_queue: Queue[Dict[str, Any]] = Queue(maxsize=max_queue_size)
        self.worker_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.worker_thread.start()

        # Stats
        self.stats = {
            "saved_count": 0,
            "compressed_size_mb": 0.0,
            "total_save_time": 0.0
        }

    def save_async(self, obj: Any, step: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save checkpoint asynchronously"""
        if self.save_queue.full():
            print(f"Warning: Checkpoint queue full, dropping save at step {step}")
            return

        self.save_queue.put({
            "obj": obj,
            "step": step,
            "metadata": metadata or {},
            "timestamp": time.time()
        })

    def save_sync(self, obj: Any, step: int, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save checkpoint synchronously (blocking)"""
        return self._save_checkpoint(obj, step, metadata or {})

    def load_latest(self) -> tuple[Any, int, Dict[str, Any]]:
        """Load the latest checkpoint"""
        checkpoints = list(self.save_dir.glob("checkpoint_*.pkl*"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return self._load_checkpoint(str(latest))

    def cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints based on keep_last and keep_every_nth"""
        checkpoints = sorted(self.save_dir.glob("checkpoint_*.pkl*"),
                           key=lambda p: self._extract_step(p))

        if len(checkpoints) <= self.keep_last:
            return

        # Keep last N
        to_keep = set(checkpoints[-self.keep_last:])

        # Keep every Nth
        for i, ckpt in enumerate(checkpoints[:-self.keep_last]):
            if (i + 1) % self.keep_every_nth == 0:
                to_keep.add(ckpt)

        # Remove others
        for ckpt in checkpoints:
            if ckpt not in to_keep:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt.name}")

    def _save_worker(self) -> None:
        """Background worker for async saves"""
        while True:
            try:
                item = self.save_queue.get(timeout=1)
                if item is None:  # Poison pill
                    break

                start_time = time.time()
                path = self._save_checkpoint(item["obj"], item["step"], item["metadata"])
                save_time = time.time() - start_time

                self.stats["saved_count"] += 1
                self.stats["total_save_time"] += save_time

                print(f"Async checkpoint saved: {path} ({save_time:.2f}s)")

                # Cleanup periodically
                if self.stats["saved_count"] % 10 == 0:
                    self.cleanup_old_checkpoints()

                self.save_queue.task_done()

            except Exception as e:
                print(f"Error in checkpoint worker: {e}")

    def _save_checkpoint(self, obj: Any, step: int, metadata: Dict[str, Any]) -> str:
        """Internal save method"""
        checkpoint_data = {
            "obj": obj,
            "step": step,
            "metadata": metadata,
            "timestamp": time.time()
        }

        # Serialize
        data = pickle.dumps(checkpoint_data, protocol=pickle.HIGHEST_PROTOCOL)

        # Compress
        compressed_data = self._compress_data(data)

        # Save
        filename = f"checkpoint_{step:08d}.pkl"
        if self.compress == "zstd":
            filename += ".zst"
        elif self.compress == "lz4":
            filename += ".lz4"

        path = self.save_dir / filename

        with open(path, 'wb') as f:
            f.write(compressed_data)

        # Update stats
        self.stats["compressed_size_mb"] += len(compressed_data) / 1024 / 1024

        return str(path)

    def _load_checkpoint(self, path: str) -> tuple[Any, int, Dict[str, Any]]:
        """Internal load method"""
        with open(path, 'rb') as f:
            compressed_data = f.read()

        # Decompress
        data = self._decompress_data(compressed_data)

        # Deserialize
        checkpoint_data = pickle.loads(data)

        return checkpoint_data["obj"], checkpoint_data["step"], checkpoint_data["metadata"]

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data based on compression setting"""
        compressor = self._select_compressor(len(data))
        return self._apply_compression(data, compressor)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data"""
        # Try different decompressors (data might have been saved with different compression)
        for decompressor in [self._decompress_zstd, self._decompress_lz4, self._decompress_zlib]:
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
            return lz4_frame.compress(data)
        elif compressor == "zstd" and HAS_ZSTD and zstd:
            compressor_obj = zstd.ZstdCompressor()
            return compressor_obj.compress(data)
        else:
            return zlib.compress(data, 6)

    def _decompress_lz4(self, data: bytes) -> bytes:
        if HAS_LZ4 and lz4_frame:
            return lz4_frame.decompress(data)
        raise ImportError("lz4 not available")

    def _decompress_zstd(self, data: bytes) -> bytes:
        if HAS_ZSTD and zstd:
            decompressor = zstd.ZstdDecompressor()
            return decompressor.decompress(data)
        raise ImportError("zstd not available")

    def _decompress_zlib(self, data: bytes) -> bytes:
        return zlib.decompress(data)

    def _extract_step(self, path: Path) -> int:
        """Extract step number from checkpoint filename"""
        name = path.stem  # checkpoint_00001234
        return int(name.split('_')[1])

    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        return self.stats.copy()

    def shutdown(self) -> None:
        """Shutdown the manager"""
        self.save_queue.put(None)  # Poison pill
        self.worker_thread.join(timeout=5)


class HierarchicalCheckpointManager:
    """
    Hierarchical checkpoint manager for ML training.

    Manages three levels of checkpoints:
    - light: Frequent lightweight checkpoints (every 1k/5k steps)
    - full: Full model checkpoints (every 10k steps)
    - archive: Long-term archive checkpoints (every 50k steps)

    Recovery policy: On failure, resume from nearest full checkpoint.
    """

    def __init__(self, save_dir: str = "models/checkpoints", compress: str = "zstd",
                 light_freq: Optional[list] = None, full_freq: int = 10000, archive_freq: int = 50000):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
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

    def _init_executor(self):
        """Initialize ThreadPoolExecutor for async saving"""
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="checkpoint")

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

    def save_checkpoint(self, step: int, model_state: Dict[str, Any],
                       optimizer_state: Optional[Dict[str, Any]] = None,
                       metrics: Optional[Dict[str, Any]] = None,
                       checkpoint_type: str = "auto") -> None:
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
            self.executor.submit(self._save_checkpoint_sync, step, model_state,
                               optimizer_state, metrics, checkpoint_type)

    def _save_checkpoint_sync(self, step: int, model_state: Dict[str, Any],
                             optimizer_state: Optional[Dict[str, Any]],
                             metrics: Optional[Dict[str, Any]],
                             checkpoint_type: str) -> None:
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
                "type": checkpoint_type
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
            with open(filepath, 'wb') as f:
                f.write(compressed_data)

            save_time = time.time() - start_time
            compression_ratio = len(compressed_data) / uncompressed_size if uncompressed_size > 0 else 1.0

            print(f"Saved {checkpoint_type} checkpoint at step {step}: {filepath} "
                  f"({save_time:.2f}s, {compression_ratio:.2f}x compression)")

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
                light_checkpoints.sort(key=lambda x: int(x.stem.split('_')[2]), reverse=True)
                for old_cp in light_checkpoints[self.keep_light:]:
                    old_cp.unlink()

        elif checkpoint_type == "full":
            # Keep specified number of full checkpoints
            full_checkpoints = list(self.save_dir.glob("checkpoint_full_*.pkl*"))
            if len(full_checkpoints) > self.keep_full:
                # Sort by step number (descending)
                full_checkpoints.sort(key=lambda x: int(x.stem.split('_')[2]), reverse=True)
                for old_cp in full_checkpoints[self.keep_full:]:
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
            return max(full_cps, key=lambda x: int(x.stem.split('_')[2]))

        # Then archive
        archive_cps = list(self.save_dir.glob("checkpoint_archive_*.pkl*"))
        if archive_cps:
            return max(archive_cps, key=lambda x: int(x.stem.split('_')[2]))

        # Finally light
        light_cps = list(self.save_dir.glob("checkpoint_light_*.pkl*"))
        if light_cps:
            return max(light_cps, key=lambda x: int(x.stem.split('_')[2]))

        return None

    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
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
            with open(checkpoint_path, 'rb') as f:
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
            "total_checkpoints": light_count + full_count + archive_count
        }

    def shutdown(self) -> None:
        """Shutdown the manager"""
        if self.executor:
            self.executor.shutdown(wait=True)