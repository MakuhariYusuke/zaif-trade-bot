"""
Gradient Probe Guard for detecting zero-gradient issues during training.

This module provides functionality to detect when gradient probes (especially for SELL actions)
are stuck at zero, which indicates a training failure. When detected, it automatically:
1. Halts training
2. Saves diagnostics (replay buffer, manifests, model state)
3. Creates an archive of all relevant data for debugging
"""

import json
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Protocol

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ztb.training.constants import DEFAULT_TOTAL_TIMESTEPS_SAC
from ztb.utils.file_utils import safe_json_dump
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GradProbeConfig:
    """Configuration for gradient probe guard."""

    # Detection thresholds
    zero_threshold: float = 1e-8  # Threshold for considering gradient as zero
    consecutive_zeros: int = 5  # Number of consecutive zero checks before triggering
    check_interval: int = 1000  # Check every N steps

    # Action-specific monitoring
    monitor_actions: List[str] = field(default_factory=lambda: ["SELL", "BUY", "HOLD"])
    critical_actions: List[str] = field(
        default_factory=lambda: ["SELL"]
    )  # Actions that trigger halt

    # Archive settings
    save_replay_buffer: bool = True
    save_model_state: bool = True
    save_diagnostics: bool = True
    archive_dir: str = "grad_probe_archives"

    # Additional diagnostics
    save_tensorboard_events: bool = True
    save_environment_state: bool = True


@dataclass
class GradProbeStats:
    """Statistics for gradient probe monitoring."""

    step: int
    timestamp: float
    action_grads: Dict[str, float] = field(default_factory=dict)
    grad_norms: Dict[str, float] = field(default_factory=dict)
    is_zero: Dict[str, bool] = field(default_factory=dict)
    consecutive_zero_count: Dict[str, int] = field(default_factory=dict)


class ModelProtocol(Protocol):
    """Protocol for model that supports gradient probe access."""

    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        ...

    def save(self, path: str) -> None:
        """Save model to path."""
        ...


class GradProbeGuard(BaseCallback):
    """
    Callback for monitoring gradient probes and auto-halting on zero-gradient issues.

    This callback tracks gradient values for specific actions (especially SELL) and
    automatically halts training when gradients are stuck at zero for too long.

    Example:
        guard = GradProbeGuard(
            config=GradProbeConfig(
                zero_threshold=1e-8,
                consecutive_zeros=5,
                critical_actions=["SELL"]
            ),
            checkpoint_dir="checkpoints"
        )
        model.learn(total_timesteps=DEFAULT_TOTAL_TIMESTEPS_SAC, callback=guard)
    """

    def __init__(
        self,
        config: Optional[GradProbeConfig] = None,
        checkpoint_dir: str = "checkpoints",
        session_id: Optional[str] = None,
        verbose: int = 1,
    ) -> None:
        # BaseCallback implementations in some test stubs may not accept
        # arguments or may be a bare object; guard the call to avoid
        # TypeError during tests in minimal environments.
        try:
            super().__init__(verbose)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                # Last-resort: ignore if super init isn't available in stub
                pass

        self.config = config or GradProbeConfig()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.session_id = session_id or f"session_{datetime.now():%Y%m%d_%H%M%S}"

        # Archive directory
        self.archive_root = Path(self.config.archive_dir)
        self.archive_root.mkdir(exist_ok=True, parents=True)

        # Monitoring state
        self.history: Deque[GradProbeStats] = deque(maxlen=1000)
        self.consecutive_zeros: Dict[str, int] = dict.fromkeys(self.config.monitor_actions, 0)
        self.last_check_step = 0
        self.halt_triggered = False
        self.halt_reason: Optional[str] = None

    def _init_callback(self) -> None:
        """Initialize callback (called by SB3)."""
        super()._init_callback()
        logger.info(
            f"GradProbeGuard initialized with config: "
            f"zero_threshold={self.config.zero_threshold}, "
            f"consecutive_zeros={self.config.consecutive_zeros}, "
            f"check_interval={self.config.check_interval}"
        )

    def _on_step(self) -> bool:
        """
        Check gradient probes on each step.

        Returns:
            False if training should halt, True otherwise.
        """
        if self.halt_triggered:
            return False

        # Check only at intervals
        if self.num_timesteps - self.last_check_step < self.config.check_interval:
            return True

        self.last_check_step = self.num_timesteps

        # Extract gradient information
        stats = self._extract_grad_stats()
        if stats is None:
            return True

        self.history.append(stats)

        # Check for zero-gradient issues
        if self._check_zero_gradients(stats):
            self._handle_zero_gradient_halt(stats)
            return False

        return True

    def _extract_grad_stats(self) -> Optional[GradProbeStats]:
        """
        Extract gradient statistics from model.

        Returns:
            GradProbeStats if successful, None otherwise.
        """
        if not hasattr(self.model, "policy"):
            logger.warning("Model does not have policy attribute")
            return None

        policy = self.model.policy
        stats = GradProbeStats(step=self.num_timesteps, timestamp=time.time())

        try:
            # Extract gradients from policy network
            # For PPO, gradients are stored in policy.mlp_extractor or policy.action_net
            if hasattr(policy, "action_net"):
                action_net = policy.action_net

                # Get gradients for each action head
                for action_name in self.config.monitor_actions:
                    grad_norm = self._get_action_grad_norm(action_net, action_name)
                    if grad_norm is not None:
                        stats.action_grads[action_name] = grad_norm
                        stats.grad_norms[action_name] = grad_norm
                        stats.is_zero[action_name] = (
                            grad_norm < self.config.zero_threshold
                        )

                        # Update consecutive zero count
                        if stats.is_zero[action_name]:
                            self.consecutive_zeros[action_name] += 1
                        else:
                            self.consecutive_zeros[action_name] = 0

                        stats.consecutive_zero_count[
                            action_name
                        ] = self.consecutive_zeros[action_name]

            return stats

        except Exception as e:
            logger.warning(f"Failed to extract gradient stats: {e}")
            return None

    def _get_action_grad_norm(
        self, action_net: Any, action_name: str
    ) -> Optional[float]:
        """
        Get gradient norm for specific action.

        Args:
            action_net: Action network from policy
            action_name: Name of action (e.g., "SELL", "BUY", "HOLD")

        Returns:
            Gradient norm if available, None otherwise.
        """
        try:
            # For discrete action spaces, action_net typically has shape (features, n_actions)
            # We compute the norm of gradients for the specific action column
            if hasattr(action_net, "weight") and hasattr(action_net.weight, "grad"):
                grad = action_net.weight.grad
                if grad is not None:
                    # Map action name to index (this is environment-specific)
                    action_idx = {"BUY": 0, "HOLD": 1, "SELL": 2}.get(action_name)
                    if action_idx is not None and grad.shape[-1] > action_idx:
                        action_grad = grad[:, action_idx]
                        return float(np.linalg.norm(action_grad.cpu().detach().numpy()))

            return None

        except Exception as e:
            logger.debug(f"Failed to get grad norm for {action_name}: {e}")
            return None

    def _check_zero_gradients(self, stats: GradProbeStats) -> bool:
        """
        Check if any critical action has zero gradients for too long.

        Args:
            stats: Current gradient statistics

        Returns:
            True if halt should be triggered, False otherwise.
        """
        for action in self.config.critical_actions:
            if action in stats.consecutive_zero_count:
                count = stats.consecutive_zero_count[action]
                if count >= self.config.consecutive_zeros:
                    logger.error(
                        f"ZERO GRADIENT DETECTED: {action} action has zero gradients "
                        f"for {count} consecutive checks (threshold: {self.config.consecutive_zeros})"
                    )
                    self.halt_reason = (
                        f"{action} action gradient stuck at zero for {count} checks "
                        f"(step {stats.step})"
                    )
                    return True

        return False

    def _handle_zero_gradient_halt(self, stats: GradProbeStats) -> None:
        """
        Handle halt triggered by zero gradients.

        This saves all diagnostics and creates an archive for debugging.

        Args:
            stats: Current gradient statistics
        """
        self.halt_triggered = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"grad_zero_{self.session_id}_{timestamp}"
        archive_dir = self.archive_root / archive_name
        archive_dir.mkdir(exist_ok=True, parents=True)

        logger.error(f"🛑 TRAINING HALTED: {self.halt_reason}")
        logger.info(f"📦 Saving diagnostics to: {archive_dir}")

        # Save manifest
        manifest = self._create_manifest(stats, timestamp)
        manifest_path = archive_dir / "manifest.json"
        safe_json_dump(manifest, str(manifest_path), indent=2)
        logger.info(f"✅ Manifest saved: {manifest_path}")

        # Save diagnostics
        if self.config.save_diagnostics:
            self._save_diagnostics(archive_dir, stats)

        # Save model state
        if self.config.save_model_state:
            self._save_model_state(archive_dir)

        # Save replay buffer
        if self.config.save_replay_buffer:
            self._save_replay_buffer(archive_dir)

        # Save TensorBoard events
        if self.config.save_tensorboard_events:
            self._save_tensorboard_events(archive_dir)

        logger.info(f"📦 Archive complete: {archive_dir}")
        logger.error(f"🛑 Training halted due to: {self.halt_reason}")

    def _create_manifest(self, stats: GradProbeStats, timestamp: str) -> Dict[str, Any]:
        """Create manifest for archive."""
        return {
            "session_id": self.session_id,
            "timestamp": timestamp,
            "halt_reason": self.halt_reason,
            "halt_step": stats.step,
            "config": {
                "zero_threshold": self.config.zero_threshold,
                "consecutive_zeros": self.config.consecutive_zeros,
                "check_interval": self.config.check_interval,
                "monitor_actions": self.config.monitor_actions,
                "critical_actions": self.config.critical_actions,
            },
            "final_stats": {
                "step": stats.step,
                "action_grads": stats.action_grads,
                "grad_norms": stats.grad_norms,
                "is_zero": stats.is_zero,
                "consecutive_zero_count": stats.consecutive_zero_count,
            },
            "history_length": len(self.history),
        }

    def _save_diagnostics(self, archive_dir: Path, stats: GradProbeStats) -> None:
        """Save detailed diagnostics."""
        diagnostics_dir = archive_dir / "diagnostics"
        diagnostics_dir.mkdir(exist_ok=True)

        # Save gradient history
        history_data = [
            {
                "step": s.step,
                "timestamp": s.timestamp,
                "action_grads": s.action_grads,
                "grad_norms": s.grad_norms,
                "is_zero": s.is_zero,
                "consecutive_zero_count": s.consecutive_zero_count,
            }
            for s in self.history
        ]
        history_path = diagnostics_dir / "gradient_history.json"
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
        logger.info(f"✅ Gradient history saved: {history_path}")

        # Save current stats
        stats_path = diagnostics_dir / "final_stats.json"
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "step": stats.step,
                    "timestamp": stats.timestamp,
                    "action_grads": stats.action_grads,
                    "grad_norms": stats.grad_norms,
                    "is_zero": stats.is_zero,
                    "consecutive_zero_count": stats.consecutive_zero_count,
                },
                f,
                indent=2,
            )
        logger.info(f"✅ Final stats saved: {stats_path}")

    def _save_model_state(self, archive_dir: Path) -> None:
        """Save model state."""
        model_dir = archive_dir / "model"
        model_dir.mkdir(exist_ok=True)

        try:
            model_path = model_dir / "model.zip"
            self.model.save(str(model_path))
            logger.info(f"✅ Model saved: {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")

    def _save_replay_buffer(self, archive_dir: Path) -> None:
        """Save replay buffer if available."""
        if not hasattr(self.model, "replay_buffer") or self.model.replay_buffer is None:
            logger.warning("⚠️  Replay buffer not available")
            return

        replay_dir = archive_dir / "replay_buffer"
        replay_dir.mkdir(exist_ok=True)

        try:
            replay_path = replay_dir / "replay_buffer.pkl"
            self.model.replay_buffer.save(str(replay_path))
            logger.info(f"✅ Replay buffer saved: {replay_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save replay buffer: {e}")

    def _save_tensorboard_events(self, archive_dir: Path) -> None:
        """Copy TensorBoard events if available."""
        # Look for tensorboard logs in checkpoint directory
        tb_dir = self.checkpoint_dir.parent / "logs"
        if not tb_dir.exists():
            logger.warning("⚠️  TensorBoard logs not found")
            return

        events_dir = archive_dir / "tensorboard_events"
        events_dir.mkdir(exist_ok=True)

        try:
            # Copy all event files
            for event_file in tb_dir.rglob("events.out.tfevents.*"):
                shutil.copy2(event_file, events_dir / event_file.name)
            logger.info(f"✅ TensorBoard events saved: {events_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to save TensorBoard events: {e}")

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of gradient probe statistics."""
        if not self.history:
            return {"status": "no_data"}

        latest = self.history[-1]
        return {
            "step": latest.step,
            "timestamp": latest.timestamp,
            "action_grads": latest.action_grads,
            "consecutive_zeros": self.consecutive_zeros.copy(),
            "halt_triggered": self.halt_triggered,
            "halt_reason": self.halt_reason,
            "history_length": len(self.history),
        }
