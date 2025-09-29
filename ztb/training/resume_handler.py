"""Helpers for resuming training sessions from checkpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

from .checkpoint_manager import TrainingCheckpointManager, TrainingCheckpointSnapshot

if TYPE_CHECKING:
    from ztb.data.streaming_pipeline import StreamingPipeline

logger = logging.getLogger(__name__)


@dataclass
class ResumeState:
    """Summary of a successful resume operation."""

    step: int
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    streaming_state: Optional[Dict[str, Any]] = None


class ResumeHandler:
    """Coordinate checkpoint restoration and streaming pipeline recovery."""

    def __init__(
        self,
        checkpoint_manager: TrainingCheckpointManager,
        *,
        streaming_pipeline: Optional["StreamingPipeline"] = None,
    ) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.streaming_pipeline = streaming_pipeline

    def resume(
        self,
        model: BaseAlgorithm,
        *,
        apply_snapshot: Optional[Callable[[TrainingCheckpointSnapshot], None]] = None,
    ) -> Optional[ResumeState]:
        snapshot = self.checkpoint_manager.load_latest()
        if snapshot is None:
            return None

        if apply_snapshot:
            apply_snapshot(snapshot)
        else:
            self.checkpoint_manager.apply_snapshot(model, snapshot)

        stream_state = snapshot.payload.get("stream_state")
        self._restore_streaming_state(stream_state)

        return ResumeState(
            step=snapshot.step,
            metrics=snapshot.metrics,
            metadata=snapshot.metadata,
            streaming_state=stream_state,
        )

    def _restore_streaming_state(self, stream_state: Optional[Dict[str, Any]]) -> None:
        if not self.streaming_pipeline or not stream_state:
            return

        buffer_df = stream_state.get("buffer")
        if buffer_df is not None:
            df = (
                buffer_df
                if isinstance(buffer_df, pd.DataFrame)
                else pd.DataFrame(buffer_df)
            )
            self.streaming_pipeline.buffer.clear()
            if not df.empty:
                self.streaming_pipeline.buffer.extend(df)

        stats = self.streaming_pipeline.stats()
        logger.info(
            "Streaming pipeline restored with %s rows (capacity %s)",
            stats.buffer.rows,
            stats.buffer.capacity,
        )
