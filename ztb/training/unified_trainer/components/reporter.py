"""
Compatibility shim for TrainingReporter.

Use ztb.training.unified_trainer.reporting.TrainingReporter as the source of truth.
"""

from typing import Any, Dict, Optional

from ztb.training.unified_trainer.reporting import TrainingReporter as _TrainingReporter
from ztb.types.common import ConfigDict


class TrainingReporter(_TrainingReporter):
    """Compatibility wrapper to preserve legacy method signatures."""

    def __init__(self, logger: Any) -> None:
        super().__init__(logger)
        self._last_config: ConfigDict = {}
        self._last_stats: Dict[str, Any] = {}
        self._last_success: bool = True

    def log_training_start(
        self,
        algorithm: str,
        config: Optional[ConfigDict] = None,
        total_timesteps: Optional[int] = None,
    ) -> None:
        cfg: ConfigDict = config or {}
        if total_timesteps is not None and isinstance(cfg, dict):
            training = dict(cfg.get("training", {}))
            training.setdefault("total_timesteps", total_timesteps)
            cfg = dict(cfg)
            cfg["training"] = training
        self._last_config = cfg
        super().log_training_start(algorithm, cfg)

    def log_training_complete(
        self,
        final_metrics: Dict[str, Any],
        training_time: float,
        model_path: Optional[str] = None,
    ) -> None:
        stats = {
            "final_metrics": final_metrics,
            "training_time": training_time,
            "model_path": model_path,
        }
        self._last_stats = stats
        self._last_success = True
        super().log_training_complete(True, stats)

    def generate_training_report(self) -> Dict[str, Any]:
        """Legacy entrypoint built from the most recent start/complete calls."""
        return self.generate_report(self._last_config, self._last_stats, self._last_success)
