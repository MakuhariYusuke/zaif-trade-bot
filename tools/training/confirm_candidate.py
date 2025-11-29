#!/usr/bin/env python3
"""Two-stage candidate confirmation helper:
1. Quick prefilter: candidate propose + quick AB-run (short timesteps)
2. Long-run verify: re-run top-N candidates with longer timesteps
3. Apply the winner to MTFWeightManager if --apply set

This is an opinionated helper for CI and local use, not for production use in training loops.
"""
from __future__ import annotations

import argparse
import logging

from ztb.trading.environment.components.reward.mtf_weight_manager import (
    MTFWeightManager,
)
from ztb.training.reward_function_optimizer.mtf_optimizer import MTFOptimizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default="config/v448/mtf_candidates")
    p.add_argument("--candidates", type=int, default=10)
    p.add_argument("--prefilter-seeds", type=int, default=1)
    p.add_argument("--prefilter-timesteps", type=int, default=500)
    p.add_argument("--verify-seeds", type=int, default=3)
    p.add_argument("--verify-timesteps", type=int, default=2000)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    logger = logging.getLogger("confirm_candidate")
    opt = MTFOptimizer(
        base_config_path=args.config,
        out_dir=args.out_dir,
        candidates=args.candidates,
        per_seed=args.prefilter_seeds,
        timesteps=args.prefilter_timesteps,
    )
    logger.info("Proposing candidates (quick prefilter)")
    candidates = opt.propose_candidates()
    scores = opt.evaluate_candidates(candidates, dry_run=False)
    # sort by composite score
    ranked = sorted(
        zip(candidates, scores), key=lambda cs: cs[1].composite_score, reverse=True
    )
    top = ranked[: args.top_n]
    logger.info(
        "Top candidates from quick prefilter: %s", [s[0].candidate_id for s in top]
    )

    # verify with longer runs
    logger.info("Longer verify for top candidates")
    for idx, (c, sc) in enumerate(top):
        logger.info("Verifying: %s", c.candidate_id)
    # Recreate optimizer with longer verify settings for evaluation
    verify_opt = MTFOptimizer(
        base_config_path=args.config,
        out_dir=args.out_dir,
        candidates=args.candidates,
        per_seed=args.verify_seeds,
        timesteps=args.verify_timesteps,
    )
    verified_scores = []
    for c, _ in top:
        sc = verify_opt.evaluate_candidates([c], dry_run=False)[0]
        verified_scores.append((c, sc))

    winner = max(verified_scores, key=lambda kv: kv[1].composite_score)
    winner_candidate, winner_score = winner
    logger.info(
        "Winner: %s composite=%s sharpe=%s ret=%s",
        winner_candidate.candidate_id,
        winner_score.composite_score,
        winner_score.mean_sharpe,
        winner_score.mean_total_return,
    )

    if args.apply:
        mgr = MTFWeightManager(config={})
        opt.apply_candidate_to_manager(winner_candidate, mgr)
        cid, ts = mgr.get_last_applied_info()
        logger.info("Applied candidate: %s at %s", cid, ts)
    else:
        logger.info("--apply not set; not applying candidate to manager")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
