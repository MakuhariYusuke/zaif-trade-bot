#!/usr/bin/env python3
"""Aggressive 5000-step SAC training entrypoint."""

from train_sac_5000step_common import Train5000Profile, run_training


if __name__ == "__main__":
    run_training(
        Train5000Profile(
            name="aggressive",
            title="AGGRESSIVE 5000-STEP TRAINING SUMMARY",
            model_path="models/sac_aggressive_5000step_final.zip",
            stats_path="analysis/training_stats_5000step_aggressive.json",
            checkpoint_dir="models/checkpoints_5000step_aggressive/",
            checkpoint_prefix="sac_aggressive_5000step",
            threshold=0.02,
            no_action_penalty=-0.001,
            action_bonus=0.0005,
            learning_rate=5e-4,
            learning_starts=50,
            batch_size=32,
            ent_coef=0.5,
            net_arch=(64, 64),
            improvements=(
                "Very low action threshold (0.02)",
                "Strong inactivity penalty (-0.001)",
                "Action execution bonus (+0.0005)",
                "Aggressive learning rate (5e-4)",
            ),
        )
    )
