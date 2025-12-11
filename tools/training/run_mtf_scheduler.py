#!/usr/bin/env python3
"""
Run MTF scheduler tool quickly from the command line.
"""
import argparse
from pathlib import Path

from ztb.trading.environment.components.reward.mtf_weight_manager import (
    MTFWeightManager,
)
from ztb.training.reward_function_optimizer.mtf_scheduler import (
    MTFScheduler,
    MTFSchedulerConfig,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="base config for MTF optimizer")
    parser.add_argument(
        "--dry-run", action="store_true", help="dry run optimizer without real training"
    )
    parser.add_argument(
        "--out",
        help="out dir for candidate configs",
        default="config/v448/mtf_candidates",
    )
    parser.add_argument(
        "--gate-composite-score",
        type=float,
        default=None,
        help="Optional composite score gate (skip apply if candidate is below)",
    )
    parser.add_argument(
        "--gate-min-reports",
        type=int,
        default=None,
        help="Optional minimum report count gate before applying a candidate",
    )
    from ztb.utils.cli import add_common_cli_args

    add_common_cli_args(parser)
    args = parser.parse_args()
    from ztb.utils.cli import configure_logging_from_args

    configure_logging_from_args(args)
    cfg = MTFSchedulerConfig(
        base_config=str(Path(args.config)),
        out_dir=args.out,
        gate_composite_score=args.gate_composite_score,
        gate_min_reports=args.gate_min_reports,
    )
    mgr = MTFWeightManager(config={})
    scheduler = MTFScheduler(mgr, cfg)
    res = scheduler.run_once(dry_run=args.dry_run, apply=not args.dry_run)
    if res:
        print(f"Applied candidate: {res.candidate_id}")
    else:
        print("No candidate applied")


if __name__ == "__main__":
    from ztb.utils.cli import run_main

    run_main(main)
