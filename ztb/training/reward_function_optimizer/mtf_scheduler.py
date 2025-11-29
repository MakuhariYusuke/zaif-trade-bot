"""
MTF Scheduler: periodically runs MTFOptimizer and applies the best candidate to the MTFWeightManager.
This is a lightweight helper for integration into training loops or for periodic CI use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from time import sleep
from typing import Optional

from ztb.training.reward_function_optimizer.mtf_optimizer import (
    CandidateConfig,
    MTFOptimizer,
)


@dataclass
class MTFSchedulerConfig:
    base_config: str
    out_dir: str = "config/v448/mtf_candidates"
    candidates: int = 10
    per_seed: int = 3
    timesteps: int = 2000
    strategy: str = "random"
    seed: int = 42


class MTFScheduler:
    def __init__(self, manager, config: MTFSchedulerConfig) -> None:
        self.manager = manager
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._optimizer = MTFOptimizer(
            base_config_path=config.base_config,
            out_dir=config.out_dir,
            candidates=config.candidates,
            per_seed=config.per_seed,
            timesteps=config.timesteps,
            strategy=config.strategy,
            seed=config.seed,
        )

    def run_once(
        self, dry_run: bool = True, apply: bool = True
    ) -> Optional[CandidateConfig]:
        # Run optimizer and optionally apply selected candidate to manager
        best_candidate, best_score = self._optimizer.run(dry_run=dry_run)
        if best_candidate is None:
            self.logger.warning("MTFOptimizer returned no candidate")
            return None
        self.logger.info(
            f"Scheduler found best candidate: {best_candidate.candidate_id}"
        )
        if apply and self.manager is not None and best_candidate is not None:
            self._optimizer.apply_candidate_to_manager(best_candidate, self.manager)
            self.logger.info(
                f"Applied candidate {best_candidate.candidate_id} to MTFWeightManager"
            )
        return best_candidate

    def create_stage_change_callback(
        self, stage_filter: Optional[list] = None, dry_run: bool = True
    ):
        """Create a callback to be used by BalanceCurriculumManager stage change listeners.

        The returned callback can be registered with `BalanceCurriculumManager.add_stage_change_listener`.
        If stage_filter is provided, it will only apply when the new stage matches one of the entries.
        """

        def _cb(**kwargs):
            stage = kwargs.get("stage")
            emergency = kwargs.get("emergency", False)
            if stage_filter and stage not in stage_filter:
                return
            self.logger.info(
                f"Stage changed to {stage} (emergency={emergency}), running optimizer callback"
            )
            try:
                self.run_once(dry_run=dry_run, apply=True)
            except Exception:
                self.logger.exception(
                    "Failed to run optimizer callback on stage change"
                )

        return _cb

    def run_periodic(self, interval_seconds: int = 3600, dry_run: bool = True) -> None:
        # Infinite loop: run and apply periodically - intended for manual deployment
        self.logger.info("Starting MTFScheduler periodic loop")
        try:
            while True:
                self.run_once(dry_run=dry_run, apply=True)
                sleep(interval_seconds)
        except KeyboardInterrupt:
            self.logger.info("MTFScheduler stopped by KeyboardInterrupt")
