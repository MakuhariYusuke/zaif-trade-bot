#!/usr/bin/env python3
"""Improved 5000-step SAC training entrypoint."""

from train_sac_5000step_common import Train5000Profile, run_training


if __name__ == "__main__":
    run_training(
        Train5000Profile(
            name="improved",
            title="IMPROVED 5000-STEP TRAINING SUMMARY",
            model_path="models/sac_improved_5000step_final.zip",
            stats_path="analysis/training_stats_5000step_improved.json",
            checkpoint_dir="models/checkpoints_5000step_improved/",
            checkpoint_prefix="sac_improved_5000step",
            threshold=0.05,
            no_action_penalty=-0.0001,
            learning_rate=3e-4,
            learning_starts=50,
            batch_size=32,
            ent_coef=0.5,
            net_arch=(64, 64),
            improvements=(
                "Lowered action thresholds (0.05 instead of 0.1)",
                "Added small penalty for inaction (-0.0001)",
                "Increased entropy coefficient (0.5) for more exploration",
                "Earlier learning start (50 steps)",
            ),
        )
    )
