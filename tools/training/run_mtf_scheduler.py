#!/usr/bin/env python3
"""
Run MTF scheduler tool quickly from the command line.
"""
import argparse
import logging
import sys
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = MTFSchedulerConfig(base_config=str(Path(args.config)), out_dir=args.out)
    mgr = MTFWeightManager(config={})
    scheduler = MTFScheduler(mgr, cfg)
    res = scheduler.run_once(dry_run=args.dry_run, apply=not args.dry_run)
    if res:
        print(f"Applied candidate: {res.candidate_id}")
    else:
        print("No candidate applied")


if __name__ == "__main__":
    sys.exit(main())
