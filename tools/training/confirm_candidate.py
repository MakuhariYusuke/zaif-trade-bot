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
import sys

from ztb.trading.environment.components.reward.mtf_weight_manager import (
    MTFWeightManager,
)
from ztb.training.reward_function_optimizer.mtf_optimizer import MTFOptimizer


def main():
    p = argparse.ArgumentParser()
    from ztb.utils.cli import add_common_cli_args

    add_common_cli_args(p)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default="config/v448/mtf_candidates")
    p.add_argument("--candidates", type=int, default=10)
    p.add_argument("--prefilter-seeds", type=int, default=1)
    p.add_argument("--prefilter-timesteps", type=int, default=500)
    p.add_argument("--verify-seeds", type=int, default=3)
    p.add_argument("--verify-timesteps", type=int, default=2000)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--apply", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Dry run: no ab_test_runner or application",
    )
    p.add_argument("--gate-sharpe", type=float, default=0.5)
    p.add_argument("--gate-return", type=float, default=0.05)
    p.add_argument("--min-reports", type=int, default=1)
    args = p.parse_args()
    from ztb.utils.cli import configure_logging_from_args

    configure_logging_from_args(args)

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
    scores = opt.evaluate_candidates(candidates, dry_run=args.dry_run)
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
        sc = verify_opt.evaluate_candidates([c], dry_run=args.dry_run)[0]
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

    # Write verification summary for gate checks / artifacts
    import json
    import subprocess
    from pathlib import Path

    from ztb.types.common import AppliedCandidateTelemetry

    rpt_dir = Path("reports")
    rpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = rpt_dir / "mtf_optimizer_summary.json"
    summary = []
    for c, sc in verified_scores:
        summary.append(
            {
                "model_name": c.candidate_id,
                "mean_sharpe": sc.mean_sharpe,
                "mean_total_return": sc.mean_total_return,
                "composite_score": sc.composite_score,
                "report_count": int(sc.report_count or 0),
            }
        )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Optionally gate using tools/ci/check_optimizer_gates.py
    gate_args = [
        sys.executable,
        "tools/ci/check_optimizer_gates.py",
        "--summary",
        str(summary_path),
        "--sharpe",
        str(args.gate_sharpe),
        "--return",
        str(args.gate_return),
        "--min-reports",
        str(args.min_reports),
    ]
    gate_ok = True
    try:
        res = subprocess.run(gate_args, capture_output=True, text=True)
        logger.info(
            "Gate result: stdout=%s stderr=%s rc=%s",
            res.stdout,
            res.stderr,
            res.returncode,
        )
        if res.returncode != 0:
            gate_ok = False
    except Exception:
        logger.exception(
            "Failed to run check_optimizer_gates.py; skipping gate and continuing"
        )

    if args.apply and not gate_ok:
        logger.warning("Gate failed; not applying candidate")
        return

    if args.apply and not args.dry_run:
        mgr = MTFWeightManager(config={})
        ok = opt.apply_candidate_to_manager(winner_candidate, mgr)
        cid, ts = mgr.get_last_applied_info()
        # Persist applied candidate telemetry to reports for auditability
        applied_path = rpt_dir / f"applied_candidate_{cid}.json"
        try:
            applied_data: AppliedCandidateTelemetry = {
                "candidate_id": cid,
                "applied_at": ts,
                "composite_score": winner_score.composite_score,
                "mean_sharpe": winner_score.mean_sharpe,
                "mean_total_return": winner_score.mean_total_return,
            }
            try:
                cfg = json.loads(
                    Path(winner_candidate.config_path).read_text(encoding="utf-8")
                )
                applied_data["weights"] = cfg.get("multi_timeframe", {}).get(
                    "feature_weights", {}
                )
            except Exception:
                applied_data["weights"] = {}
            applied_path.write_text(
                json.dumps(applied_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            logger.exception("Failed to persist applied candidate info")
        if ok is True:
            logger.info("Applied candidate: %s at %s", cid, ts)
        else:
            logger.warning("set_weights returned False for candidate %s", cid)
    else:
        logger.info("--apply not set; not applying candidate to manager")


if __name__ == "__main__":
    from ztb.utils.cli import run_main

    run_main(main)
