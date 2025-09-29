"""Training checkpoint management utilities."""

from __future__ import annotations

import hashlib
import logging
import pickle
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from ztb.utils.observability import ObservabilityClient

logger = logging.getLogger(__name__)


@dataclass
class TrainingCheckpointConfig:
    """Configuration for training checkpoint behaviour."""

    interval_steps: int = 10_000
    keep_last: int = 5
    compress: str = "zstd"
    async_save: bool = True
    include_optimizer: bool = True
    include_replay_buffer: bool = True
    include_rng_state: bool = True


@dataclass
class TrainingCheckpointSnapshot:
    """In-memory representation of a checkpoint suitable for resuming."""

    step: int
    payload: Dict[str, Any]
    metadata: Dict[str, Any]

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.payload.get("metrics", {})  # type: ignore[no-any-return]


class TrainingCheckpointManager:
    """High-level manager for saving and restoring long-running training state."""

    SCHEMA_VERSION = 1

    def __init__(self, save_dir: str, config: Optional[TrainingCheckpointConfig] = None, observability: Optional[ObservabilityClient] = None) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or TrainingCheckpointConfig()
        self.observability = observability
        self.correlation_id = observability.correlation_id if observability else None
        self._manager = CheckpointManager(
            save_dir=str(self.save_dir),
            keep_last=self.config.keep_last,
            compress=self.config.compress,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def should_checkpoint(self, step: int) -> bool:
        return step > 0 and step % self.config.interval_steps == 0

    def save(
        self,
        *,
        step: int,
        model: BaseAlgorithm,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        stream_state: Optional[Dict[str, Any]] = None,
        async_save: Optional[bool] = None,
    ) -> None:
        payload = self._build_payload(
            model=model,
            metrics=metrics or {},
            extra=extra or {},
            stream_state=stream_state or {},
        )
        metadata = {
            "timestamp": payload["timestamp"],
            "schema_version": payload["schema_version"],
            "correlation_id": self.correlation_id,
            "metrics": {
                key: value
                for key, value in (metrics or {}).items()
                if isinstance(value, (int, float))
            },
        }
        use_async = self.config.async_save if async_save is None else async_save
        if use_async:
            self._manager.save_async(payload, step, metadata)
        else:
            self._manager.save_sync(payload, step, metadata)

        if self.observability:
            self.observability.log_event(
                "checkpoint_save",
                {
                    "step": step,
                    "async": use_async,
                    "metrics": metrics or {},
                },
            )
            self.observability.record_metrics({
                "checkpoint_step": float(step),
            })

    def load_latest(self) -> Optional[TrainingCheckpointSnapshot]:
        try:
            payload, step, metadata = self._manager.load_latest()
        except FileNotFoundError:
            if self.observability:
                self.observability.log_event('checkpoint_load_missing')
            return None

        self._validate_payload(payload)  # type: ignore[arg-type]
        snapshot = TrainingCheckpointSnapshot(step=step, payload=payload, metadata=metadata)  # type: ignore[arg-type]
        if self.observability:
            self.observability.log_event(
                'checkpoint_load',
                {
                    'step': step,
                    'metrics': snapshot.metrics,
                },
            )
        return snapshot

    def apply_snapshot(self, model: BaseAlgorithm, snapshot: TrainingCheckpointSnapshot) -> None:
        payload = snapshot.payload
        model.policy.load_state_dict(payload["policy_state"], strict=False)

        if self.config.include_optimizer and payload.get("optimizer_state"):
            optimizer = getattr(model.policy, "optimizer", None)
            if optimizer is not None:
                optimizer.load_state_dict(payload["optimizer_state"])
            else:
                logger.warning("Optimizer state present in checkpoint but optimizer not found on model")

        if self.config.include_replay_buffer:
            self._restore_buffer(model, payload.get("buffer_kind"), payload.get("buffer_bytes"))

        if self.config.include_rng_state:
            self._restore_rng_state(payload.get("rng_state"))

        if self.observability:
            self.observability.log_event(
                'checkpoint_apply',
                {
                    'step': snapshot.step,
                    'metrics': snapshot.metrics,
                },
            )

    def shutdown(self) -> None:
        self._manager.shutdown()
        if self.observability:
            self.observability.log_event('checkpoint_manager_shutdown')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_payload(
        self,
        *,
        model: BaseAlgorithm,
        metrics: Dict[str, Any],
        extra: Dict[str, Any],
        stream_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "policy_state": model.policy.state_dict(),
            "optimizer_state": None,
            "buffer_kind": None,
            "buffer_bytes": None,
            "metrics": metrics,
            "extra": extra,
            "stream_state": stream_state,
            "rng_state": None,
        }

        if self.config.include_optimizer:
            optimizer = getattr(model.policy, "optimizer", None)
            if optimizer is not None:
                try:
                    payload["optimizer_state"] = optimizer.state_dict()
                except Exception:
                    logger.exception("Failed to serialize optimizer state; continuing without it")

        if self.config.include_replay_buffer:
            kind, buffer_bytes = self._capture_buffer(model)
            payload["buffer_kind"] = kind
            payload["buffer_bytes"] = buffer_bytes

        if self.config.include_rng_state:
            payload["rng_state"] = self._collect_rng_state()

        payload["checksum"] = self._compute_checksum(payload)
        return payload

    def _compute_checksum(self, payload: Dict[str, Any]) -> str:
        payload_copy = {k: v for k, v in payload.items() if k != "checksum"}
        blob = pickle.dumps(payload_copy, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(blob).hexdigest()

    def _validate_payload(self, payload: Dict[str, Any]) -> None:
        expected = payload.get("checksum")
        if not expected:
            raise ValueError("Checkpoint payload missing checksum")
        try:
            actual = self._compute_checksum(payload)
        except Exception as exc:
            logger.warning("Failed to recompute checkpoint checksum: %s", exc)
            if self.observability:
                self.observability.log_event('checkpoint_checksum_error', {'error': repr(exc)})
            return
        if actual != expected:
            # Compute additional diagnostic information
            payload_size = len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            payload_copy_size = len(pickle.dumps({k: v for k, v in payload.items() if k != "checksum"}, protocol=pickle.HIGHEST_PROTOCOL))
            diagnostics = {
                'expected': expected,
                'actual': actual,
                'payload_size_bytes': payload_size,
                'payload_without_checksum_size_bytes': payload_copy_size,
                'schema_version': payload.get('schema_version'),
                'timestamp': payload.get('timestamp')
            }
            logger.warning("Checkpoint checksum mismatch: %s", diagnostics)
            if self.observability:
                self.observability.log_event('checkpoint_checksum_mismatch', diagnostics)
            return
        schema = payload.get("schema_version")
        if schema != self.SCHEMA_VERSION:
            raise ValueError(f"Unsupported checkpoint schema version: {schema}")

    def _capture_buffer(self, model: BaseAlgorithm) -> Tuple[Optional[str], Optional[bytes]]:
        if not self.config.include_replay_buffer:
            return None, None

        if hasattr(model, "save_replay_buffer"):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                model.save_replay_buffer(str(tmp_path))
                data = tmp_path.read_bytes()
                return "replay_buffer_file", data
            except Exception:
                logger.debug("Failed to save replay buffer using save_replay_buffer", exc_info=True)
            finally:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

        for attr in ("replay_buffer", "rollout_buffer"):
            buffer_obj = getattr(model, attr, None)
            if buffer_obj is not None:
                try:
                    data = pickle.dumps(buffer_obj, protocol=pickle.HIGHEST_PROTOCOL)
                    return f"{attr}_pickle", data
                except Exception:
                    logger.debug("Buffer %s could not be pickled", attr, exc_info=True)
        return None, None

    def _restore_buffer(self, model: BaseAlgorithm, kind: Optional[str], data: Optional[bytes]) -> None:
        if not kind or not data:
            return

        try:
            if kind == "replay_buffer_file" and hasattr(model, "load_replay_buffer"):
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    tmp_path.write_bytes(data)
                    model.load_replay_buffer(str(tmp_path))
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
            elif kind == "replay_buffer_pickle":
                setattr(model, "replay_buffer", pickle.loads(data))
            elif kind == "rollout_buffer_pickle":
                setattr(model, "rollout_buffer", pickle.loads(data))
            else:
                setattr(model, kind.replace("_pickle", ""), pickle.loads(data))
        except Exception:
            logger.exception("Failed to restore buffer kind %s", kind)

    def _collect_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }
        try:
            import torch

            state["torch"] = torch.get_rng_state()
            if torch.cuda.is_available():
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            logger.debug("Torch RNG state could not be captured", exc_info=True)
        return state

    def _restore_rng_state(self, rng_state: Optional[Dict[str, Any]]) -> None:
        if not rng_state:
            return
        try:
            python_state = rng_state.get("python")
            if python_state:
                random.setstate(python_state)
            numpy_state = rng_state.get("numpy")
            if numpy_state is not None:
                np.random.set_state(numpy_state)
            import torch

            torch_state = rng_state.get("torch")
            if torch_state is not None:
                torch.set_rng_state(torch_state)
            cuda_state = rng_state.get("torch_cuda")
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)
        except Exception:
            logger.warning("Failed to restore RNG state", exc_info=True)
