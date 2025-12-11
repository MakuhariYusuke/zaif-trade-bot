"""
MTF Scheduler: periodically runs MTFOptimizer and applies the best candidate to the MTFWeightManager.
This is a lightweight helper for integration into training loops or for periodic CI use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from time import sleep
from typing import Callable, Optional

from ztb.training.reward_function_optimizer.mtf_optimizer import MTFOptimizer
from ztb.types.common import CandidateConfig, StageChangeEvent


@dataclass
class MTFSchedulerConfig:
    base_config: str
    out_dir: str = "config/v448/mtf_candidates"
    candidates: int = 10
    per_seed: int = 3
    timesteps: int = 2000
    strategy: str = "random"
    seed: int = 42
    gate_composite_score: Optional[float] = None
    gate_min_reports: Optional[int] = None


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
            # Apply gating checks if configured
            gate_ok = True
            if (
                self.config.gate_composite_score is not None
                and getattr(best_score, "composite_score", 0.0)
                < self.config.gate_composite_score
            ):
                self.logger.warning(
                    "Candidate %s rejected by composite_score gate: %s < %s",
                    best_candidate.candidate_id,
                    getattr(best_score, "composite_score", 0.0),
                    self.config.gate_composite_score,
                )
                gate_ok = False
            if (
                self.config.gate_min_reports is not None
                and getattr(best_score, "report_count", 0)
                < self.config.gate_min_reports
            ):
                self.logger.warning(
                    "Candidate %s rejected by report_count gate: %s < %s",
                    best_candidate.candidate_id,
                    getattr(best_score, "report_count", 0),
                    self.config.gate_min_reports,
                )
                gate_ok = False
            if not gate_ok:
                self.logger.info("Not applying candidate due to gating conditions")
                return best_candidate
            ok = self._optimizer.apply_candidate_to_manager(
                best_candidate, self.manager
            )
            if ok:
                self.logger.info(
                    f"Applied candidate {best_candidate.candidate_id} to MTFWeightManager"
                )
                # Persist telemetry for applied candidate
                try:
                    import json
                    from pathlib import Path

                    rpt_dir = Path("reports")
                    rpt_dir.mkdir(parents=True, exist_ok=True)
                    cid, ts = self.manager.get_last_applied_info()
                    applied_path = rpt_dir / f"applied_candidate_{cid}.json"
                    cfg = json.loads(
                        Path(best_candidate.config_path).read_text(encoding="utf-8")
                    )
                    from ztb.types.common import AppliedCandidateTelemetry

                    applied_data: AppliedCandidateTelemetry = {
                        "candidate_id": cid,
                        "applied_at": ts,
                        "weights": cfg.get("multi_timeframe", {}).get(
                            "feature_weights", {}
                        ),
                        "composite_score": getattr(best_score, "composite_score", None),
                        "mean_sharpe": getattr(best_score, "mean_sharpe", None),
                        "mean_total_return": getattr(
                            best_score, "mean_total_return", None
                        ),
                    }
                    applied_path.write_text(
                        json.dumps(applied_data, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception:
                    self.logger.exception("Failed to persist applied candidate info")
            else:
                self.logger.warning(
                    f"MTFWeightManager rejected candidate {best_candidate.candidate_id} (set_weights returned False)"
                )
        return best_candidate

    def create_stage_change_callback(
        self, stage_filter: Optional[list] = None, dry_run: bool = True
    ) -> Callable[[StageChangeEvent], None]:
        """Create a callback to be used by BalanceCurriculumManager stage change listeners.

        The returned callback can be registered with `BalanceCurriculumManager.add_stage_change_listener`.
        If stage_filter is provided, it will only apply when the new stage matches one of the entries.
        """

        def _cb(**kwargs):
            # event uses `new_stage` to indicate the stage progressed to
            stage = kwargs.get("new_stage") or kwargs.get("stage")
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
