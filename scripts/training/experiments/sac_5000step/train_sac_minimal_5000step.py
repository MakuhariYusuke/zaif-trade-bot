#!/usr/bin/env python3
"""Minimal 5000-step SAC training entrypoint."""

from train_sac_5000step_common import Train5000Profile, run_training


if __name__ == "__main__":
    run_training(
        Train5000Profile(
            name="minimal",
            title="MINIMAL 5000-STEP TRAINING SUMMARY",
            model_path="models/sac_minimal_5000step_final.zip",
            stats_path="analysis/training_stats_5000step_minimal.json",
            checkpoint_dir="models/checkpoints_5000step_minimal/",
            checkpoint_prefix="sac_minimal_5000step",
            threshold=0.10,
            learning_rate=3e-4,
            learning_starts=100,
            batch_size=32,
            ent_coef=0.1,
            net_arch=(64, 64),
        )
    )
