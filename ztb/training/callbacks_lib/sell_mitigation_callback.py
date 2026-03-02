"""
SELL Bias Mitigation Callback.

This callback monitors and logs statistics from various SELL bias mitigation
components during training, including:
- Lagrange constraints
- Gradient probes
- PAN (Per-Action Advantage Normalization)
- Target Entropy Controller
- Stratified Sampler
"""

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ztb.training.experiments.entropy_temperature import TargetEntropyController
from ztb.training.optimization.adv_norm import PerActionAdvantageNormalizer
from ztb.training.optimization.lagrange_constraint import LagrangeConstraint
from ztb.training.optimization.stratified_sampler import StratifiedSampler
from ztb.training.utils.grad_probes import SELLGradientProbe
from ztb.training.utils.weights import ActionWeightCalculator

class SELLBiasMitigationCallback(BaseCallback):
    """Callback for SELL bias mitigation during training."""

    def __init__(
        self,
        lagrange: LagrangeConstraint | None = None,
        probe: SELLGradientProbe | None = None,
        weight_calc: ActionWeightCalculator | None = None,
        pan_normalizer: PerActionAdvantageNormalizer | None = None,
        entropy_controller: TargetEntropyController | None = None,
        stratified_sampler: StratifiedSampler | None = None,
        verbose: int = 0,
    ):
        """
        Initialize SELL bias mitigation callback.

        Args:
            lagrange: Lagrange constraint for minimum action rate
            probe: Gradient probe for monitoring and failsafe
            weight_calc: Action weight calculator
            pan_normalizer: Per-Action Advantage Normalizer
            entropy_controller: Target Entropy Controller
            stratified_sampler: Stratified mini-batch sampler
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.lagrange = lagrange
        self.probe = probe
        self.weight_calc = weight_calc
        self.pan_normalizer = pan_normalizer
        self.entropy_controller = entropy_controller
        self.stratified_sampler = stratified_sampler
        self.step_count = 0

    def _on_step(self) -> bool:
        """
        Called at each step. Returns False to stop training.

        Returns:
            bool: True to continue training, False to stop
        """
        self.step_count += 1

        # Log Lagrange statistics
        if self.lagrange is not None:
            stats = self.lagrange.get_statistics()
            for key, value in stats.items():
                self.logger.record(f"lagrange/{key}", value)

        # Log probe statistics
        if self.probe is not None:
            stats = self.probe.get_statistics()
            for key, value in stats.items():
                self.logger.record(f"probe/{key}", value)

        # Log PAN statistics
        if self.pan_normalizer is not None:
            stats = self.pan_normalizer.get_statistics()
            for key, value in stats.items():
                self.logger.record(f"pan/{key}", value)

        # Log Target Entropy statistics
        if self.entropy_controller is not None:
            stats = self.entropy_controller.get_statistics()
            for key, value in stats.items():
                self.logger.record(f"entropy/{key}", value)

        # Log Stratified Sampler statistics
        if self.stratified_sampler is not None:
            sampler_stats: dict[str, Any] = self.stratified_sampler.get_statistics()
            # Log bucket distribution
            if "bucket_counts" in sampler_stats:
                bucket_counts = sampler_stats["bucket_counts"]
                if isinstance(bucket_counts, np.ndarray):
                    for regime in range(3):
                        for action in range(3):
                            self.logger.record(
                                f"stratified/bucket_r{regime}_a{action}",
                                int(bucket_counts[regime, action]),
                            )

        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout.

        Note: Lagrange constraint is now integrated into CustomPPO,
        so this method is primarily for future extensions.
        """
        pass
