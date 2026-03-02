"""
MTFOptimizer MVP: Basic candidate generation and evaluation using ab_test_runner.

This module is intentionally lightweight and avoids importing heavy libs at module import time.
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from pathlib import Path

from ztb.io.json_io import read_json_object, write_json
from ztb.training.reward_function_optimizer.candidate_evaluator import (
    CandidateEvaluationResult,
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

    def _load_base_config(self) -> dict[str, object]:
        return read_json_object(self.base_config_path)

    @staticmethod
    def _extract_feature_weights(config: dict[str, object]) -> dict[str, float]:
        multi_timeframe = config.get("multi_timeframe")
        if not isinstance(multi_timeframe, dict):
            raise ValueError(
                "Invalid base_config: missing 'multi_timeframe' section"
            )

        raw_weights = multi_timeframe.get("feature_weights")
        if not isinstance(raw_weights, dict) or not raw_weights:
            raise ValueError(
                "Invalid base_config: 'multi_timeframe.feature_weights' missing or empty"
            )

        normalized_weights: dict[str, float] = {}
        for key, value in raw_weights.items():
            normalized_weights[str(key)] = float(value)
        return normalized_weights

    @staticmethod
    def _extract_model_name(config: dict[str, object]) -> str:
        training = config.get("training")
        if not isinstance(training, dict):
            return "mtf_candidate"
        model_name = training.get("model_name")
        if isinstance(model_name, str) and model_name:
            return model_name
        return "mtf_candidate"

    def propose_candidates(self) -> list[CandidateConfig]:
        base = self._load_base_config()
        weights = self._extract_feature_weights(base)
        base_model_name = self._extract_model_name(base)
        keys = list(weights.keys())

        candidates: list[CandidateConfig] = []
        for i in range(self.candidates):
            # Perturb weights within +/- 0.1 but keep positive and sum normalized.
            perturbed: dict[str, float] = {}
            total = 0.0
            for key in keys:
                base_val = weights.get(key, 0.0)
                delta = random.uniform(-0.1, 0.1) * base_val
                value = max(0.0, base_val + delta)
                perturbed[key] = value
                total += value

            if total <= 0:
                perturbed = dict(weights)
            else:
                for key in keys:
                    perturbed[key] = float(perturbed[key] / total)
                largest_key = max(perturbed.keys(), key=lambda k: perturbed[k])
                other_sum = sum(value for k, value in perturbed.items() if k != largest_key)
                perturbed[largest_key] = round(1.0 - other_sum, 9)
                for key in keys:
                    perturbed[key] = round(perturbed[key], 9)

            # Deep copy is required to avoid mutating base nested sections across candidates.
            config_copy = deepcopy(base)
            multi_timeframe = config_copy.get("multi_timeframe")
            training = config_copy.get("training")
            if not isinstance(multi_timeframe, dict) or not isinstance(training, dict):
                raise ValueError(
                    "Invalid base_config: 'multi_timeframe' or 'training' section missing"
                )
            multi_timeframe["feature_weights"] = perturbed
            training["model_name"] = f"{base_model_name}_candidate_{i}"

            candidate_file = self.out_dir / f"mtf_candidate_{i}.json"
            write_json(candidate_file, config_copy, indent=2, ensure_ascii=False)
            candidates.append(
                CandidateConfig(
                    config_path=str(candidate_file), candidate_id=f"mtf_candidate_{i}"
                )
            )

        return candidates

    def evaluate_candidates(
        self, candidates: list[CandidateConfig], dry_run: bool = False
    ) -> list[CandidateScore]:
        results: list[CandidateScore] = []

        if dry_run:
            for candidate in candidates:
                results.append(
                    CandidateScore(
                        candidate_id=candidate.candidate_id,
                        mean_sharpe=0.0,
                        mean_total_return=0.0,
                        composite_score=0.0,
                        report_count=0,
                        run_artifacts=[],
                    )
                )
            return results

        for candidate in candidates:
            try:
                metrics: CandidateEvaluationResult = evaluate_candidate(
                    candidate.config_path,
                    seeds=self.per_seed,
                    timesteps=self.timesteps,
                    dry_run=False,
                )
                report_count = int(metrics["report_count"])
                if report_count < int(self.per_seed):
                    raise RuntimeError("Not enough reports produced for candidate")

                mean_sharpe = float(metrics["mean_sharpe"])
                mean_return = float(metrics["mean_total_return"])
                composite = self._composite_score(mean_sharpe, mean_return)
                results.append(
                    CandidateScore(
                        candidate_id=candidate.candidate_id,
                        mean_sharpe=mean_sharpe,
                        mean_total_return=mean_return,
                        composite_score=composite,
                        report_count=report_count,
                        run_artifacts=list(metrics["run_artifacts"]),
                    )
                )
            except Exception:
                results.append(CandidateScore(candidate_id=candidate.candidate_id))

        return results

    def _composite_score(self, sharpe: float, ret: float) -> float:
        # Simple composite: weighted by sharpe and return (norm)
        return round((0.6 * sharpe + 0.4 * min(1.0, ret)), 4)

    def select_best(self, scores: list[CandidateScore]) -> CandidateScore:
        return max(scores, key=lambda score: score.composite_score)

    def run(
        self, dry_run: bool = False
    ) -> tuple[CandidateConfig | None, CandidateScore | None]:
        candidates = self.propose_candidates()
        scores = self.evaluate_candidates(candidates, dry_run=dry_run)
        best_score = self.select_best(scores)
        best_candidate = next(
            (candidate for candidate in candidates if candidate.candidate_id == best_score.candidate_id),
            None,
        )
        return best_candidate, best_score

    def apply_candidate_to_manager(self, candidate: CandidateConfig, manager: object) -> bool:
        """Apply candidate weights parsed from candidate config into an MTFWeightManager instance.

        Args:
            candidate: CandidateConfig returned from propose_candidates
            manager: MTFWeightManager or similar object with set_weights(dict) method
        """
        logger = logging.getLogger(self.__class__.__name__)
        payload = read_json_object(Path(candidate.config_path))
        multi_timeframe = payload.get("multi_timeframe")
        feature_weights = (
            multi_timeframe.get("feature_weights")
            if isinstance(multi_timeframe, dict)
            else {}
        )
        if not isinstance(feature_weights, dict):
            feature_weights = {}

        logger.info(
            "Applying candidate %s to manager (weights=%s)",
            candidate.candidate_id,
            feature_weights,
        )
        if hasattr(manager, "set_weights"):
            try:
                update_payload = dict(feature_weights)
                update_payload["_candidate_id"] = candidate.candidate_id
                try:
                    ok = manager.set_weights(update_payload)
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
                manager._weights = feature_weights  # type: ignore[attr-defined]
                return True
            except Exception:
                pass
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--candidates", type=int, default=10)
    args = parser.parse_args()

    optimizer = MTFOptimizer(base_config_path=args.config, candidates=args.candidates)
    best_candidate, best_score = optimizer.run(dry_run=args.dry_run)
    if best_candidate is not None and best_score is not None:
        print(
            f"Best candidate: {best_candidate.candidate_id} composite={best_score.composite_score} sharpe={best_score.mean_sharpe} return={best_score.mean_total_return}"
        )
    else:
        print("No best candidate found")
