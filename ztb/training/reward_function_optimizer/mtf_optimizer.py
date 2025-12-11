"""
MTFOptimizer MVP: Basic candidate generation and evaluation using ab_test_runner.

This module is intentionally lightweight and avoids importing heavy libs at module import time.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ztb.training.reward_function_optimizer.candidate_evaluator import (
    evaluate_candidate,
)
from ztb.types.common import CandidateConfig, CandidateScore

# CandidateConfig and CandidateScore dataclasses are defined in ztb.types.common


class MTFOptimizer:
    """A minimal MTF optimizer for Layer 6.

    - propose_candidates: returns candidate configs by perturbing feature_weights
    - evaluate_candidates: runs ab_test_runner for candidates (dry_run optional)
    - select_best: selects candidate by composite score
    """

    def __init__(
        self,
        base_config_path: str,
        out_dir: str | Path = "config/v448/mtf_candidates",
        candidates: int = 10,
        per_seed: int = 3,
        timesteps: int = 2000,
        strategy: str = "random",
        seed: int = 42,
    ) -> None:
        self.base_config_path = Path(base_config_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates = candidates
        self.per_seed = per_seed
        self.timesteps = timesteps
        self.strategy = strategy
        self.seed = seed
        random.seed(self.seed)

    def _load_base_config(self) -> Dict:
        return json.loads(self.base_config_path.read_text(encoding="utf-8"))

    def propose_candidates(self) -> List[CandidateConfig]:
        base = self._load_base_config()
        mtf = base.get("multi_timeframe", {})
        weights = mtf.get("feature_weights", {})
        # Validate base config contains expected fields to avoid KeyErrors later
        if not isinstance(weights, dict) or not weights:
            raise ValueError(
                f"Invalid base_config: 'multi_timeframe.feature_weights' missing or empty in {self.base_config_path}"
            )
        # Expected keys: '1min', '5min', '15min'
        keys = list(weights.keys())
        candidates: List[CandidateConfig] = []
        for i in range(self.candidates):
            # perturb weights within +/- 0.1 but keep positive and sum normalized
            perturbed = {}
            total = 0.0
            for k in keys:
                base_val = float(weights.get(k, 0.0))
                delta = random.uniform(-0.1, 0.1) * base_val
                val = max(0.0, base_val + delta)
                perturbed[k] = val
                total += val
            if total <= 0:
                # fallback: use base weights
                perturbed = weights.copy()
            else:
                # normalize
                for k in keys:
                    perturbed[k] = float(perturbed[k] / total)
                # Adjust for rounding error by ensuring sum is 1.0
                # Compute the sum and explicitly set the largest weight to account for rounding residual
                largest = max(perturbed.keys(), key=lambda kk: perturbed[kk])
                other_sum = sum(v for k, v in perturbed.items() if k != largest)
                perturbed[largest] = round(1.0 - other_sum, 9)
                # finally round to stable precision (6 decimals)
                for k in keys:
                    perturbed[k] = round(perturbed[k], 9)
            # write candidate config - set unique model_name per candidate
            cfg = base.copy()
            # set model_name unique per candidate to be able to identify reports
            base_model_name = cfg.get("training", {}).get("model_name", "mtf_candidate")
            cfg["multi_timeframe"]["feature_weights"] = perturbed
            cfg["training"]["model_name"] = f"{base_model_name}_candidate_{i}"
            candidate_file = self.out_dir / f"mtf_candidate_{i}.json"
            candidate_file.write_text(
                json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            candidates.append(
                CandidateConfig(
                    config_path=str(candidate_file), candidate_id=f"mtf_candidate_{i}"
                )
            )
        return candidates

    def evaluate_candidates(
        self, candidates: List[CandidateConfig], dry_run: bool = False
    ) -> List[CandidateScore]:
        results: List[CandidateScore] = []
        # If dry_run, return placeholder scores
        if dry_run:
            for c in candidates:
                results.append(
                    CandidateScore(
                        candidate_id=c.candidate_id,
                        mean_sharpe=0.0,
                        mean_total_return=0.0,
                        composite_score=0.0,
                        report_count=0,
                        run_artifacts=[],
                    )
                )
            return results

        for c in candidates:
            # Run ab_test_runner for candidate
            cmd = [
                sys.executable,
                "tools/ab_test_runner.py",
                "--configs",
                c.config_path,
                "--seeds",
                str(self.per_seed),
                "--timesteps",
                str(self.timesteps),
            ]
            try:
                metrics = evaluate_candidate(
                    c.config_path,
                    seeds=self.per_seed,
                    timesteps=self.timesteps,
                    dry_run=False,
                )
                # Enforce minimum report_count to trust metrics
                rc = int(metrics.get("report_count", 0) or 0)
                if rc < int(self.per_seed):
                    # treat as invalid candidate (failed to produce enough reports)
                    raise RuntimeError("Not enough reports produced for candidate")
                mean_sharpe = metrics.get("mean_sharpe", 0.0)
                mean_return = metrics.get("mean_total_return", 0.0)
                composite = self._composite_score(mean_sharpe, mean_return)
                results.append(
                    CandidateScore(
                        candidate_id=c.candidate_id,
                        mean_sharpe=mean_sharpe,
                        mean_total_return=mean_return,
                        composite_score=composite,
                        report_count=rc,
                        run_artifacts=metrics.get("run_artifacts", []),
                    )
                )
            except Exception:
                results.append(CandidateScore(candidate_id=c.candidate_id))
        return results

    def _composite_score(self, sharpe: float, ret: float) -> float:
        # Simple composite: weighted by sharpe and return (norm)
        return round((0.6 * sharpe + 0.4 * min(1.0, ret)), 4)

    def select_best(self, scores: List[CandidateScore]) -> CandidateScore:
        best = max(scores, key=lambda s: s.composite_score)
        return best

    def run(
        self, dry_run: bool = False
    ) -> tuple[Optional[CandidateConfig], Optional[CandidateScore]]:
        candidates = self.propose_candidates()
        scores = self.evaluate_candidates(candidates, dry_run=dry_run)
        best_score = self.select_best(scores)
        # Find matching candidate config by id
        best_candidate = next(
            (c for c in candidates if c.candidate_id == best_score.candidate_id), None
        )
        return best_candidate, best_score

    def apply_candidate_to_manager(self, candidate: CandidateConfig, manager) -> bool:
        """Apply candidate weights parsed from candidate config into an MTFWeightManager instance.

        Args:
            candidate: CandidateConfig returned from propose_candidates
            manager: MTFWeightManager or similar object with set_weights(dict) method
        """
        import logging

        logger = logging.getLogger(self.__class__.__name__)
        cfg = json.loads(Path(candidate.config_path).read_text(encoding="utf-8"))
        fw = cfg.get("multi_timeframe", {}).get("feature_weights", {})
        logger.info(
            f"Applying candidate {candidate.candidate_id} to manager (weights={fw})"
        )
        if hasattr(manager, "set_weights"):
            try:
                # Attach candidate id for telemetry if supported by manager
                try:
                    payload = dict(fw)
                    payload["_candidate_id"] = candidate.candidate_id
                except Exception:
                    payload = fw
                try:
                    ok = manager.set_weights(payload)
                    if ok is False:
                        logger.warning(
                            "manager.set_weights returned False for candidate %s",
                            candidate.candidate_id,
                        )
                    return bool(ok)
                except Exception:
                    logger.exception(
                        "manager.set_weights raised an exception for candidate %s",
                        candidate.candidate_id,
                    )
            except Exception:
                logger.exception("Failed to apply weights to manager")
        else:
            # fallback: try to set attributes if manager exposes `_weights`
            try:
                manager._weights = fw
                return True
            except Exception:
                # silently ignore if manager doesn't support weight set
                pass
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--candidates", type=int, default=10)
    args = parser.parse_args()

    m = MTFOptimizer(base_config_path=args.config, candidates=args.candidates)
    best_candidate, best_score = m.run(dry_run=args.dry_run)
    if best_candidate is not None and best_score is not None:
        print(
            f"Best candidate: {best_candidate.candidate_id} composite={best_score.composite_score} sharpe={best_score.mean_sharpe} return={best_score.mean_total_return}"
        )
    else:
        print("No best candidate found")
