"""
Phase 4: Execution-Aware Training.

This script trains a SAC model directly in a "Realistic" environment.
It utilizes the enhanced UnifiedTrainer to handle configuration and evaluation.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from ztb.training.unified_trainer.trainer import UnifiedTrainer


def run_execution_aware_training():
    print("========================================================")
    print("🚀 Phase 4: Execution-Aware Training (v450)")
    print("========================================================")

    # Define the Realistic Execution Model Config
    realistic_execution_model = {
        "base_slippage": 0.0005,  # 0.05% base slippage
        "atr_slippage_factor": 0.5,  # Slippage increases with volatility
        "base_latency_ms": 50.0,  # 50ms latency
        "latency_jitter_ms": 20.0,  # +/- 20ms jitter
    }

    # Configuration
    config = {
        "model_name": "sac_v450_execution_aware",
        "training": {
            "algorithm": "sac",
            "total_timesteps": 20000,  # Short run for verification (increase for full training)
            "log_interval": 1000,
            "checkpoint_interval": 5000,
            "checkpoint_dir": os.path.join(
                project_root, "models", "checkpoints", "phase4"
            ),
            # Data Configuration
            "data_config": {
                "data_path": os.path.join(
                    project_root, "data", "range_medium_featured.csv"
                ),
            },
            # Environment Configuration (TRAINING - REALISTIC)
            "environment": {
                "config": {
                    "initial_portfolio_value": 100000.0,
                    "transaction_cost": 0.001,  # 0.1% fee
                    "slippage": 0.0,  # Handled by execution_model, but base param kept for compat
                    "execution_model": realistic_execution_model,  # ENABLED FOR TRAINING
                    "feature_set": "full",
                    "reward_scaling": 1.0,
                    "use_continuous_actions": True,
                    "action_space_type": "continuous",
                    "adaptive_threshold_mode": True,
                }
            },
            # SAC Hyperparameters
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
            },
        },
        # Evaluation Configuration (New Feature!)
        "evaluation": {
            "enabled": True,
            "eval_freq": 2000,
            "n_eval_episodes": 5,
            "data_path": os.path.join(
                project_root, "data", "range_medium_featured.csv"
            ),  # Use same data for now (in real usage, use validation set)
            # Evaluation Environment Overrides
            # We want to evaluate on the SAME realistic conditions to track progress
            "overrides": {
                "execution_model": realistic_execution_model,
                "transaction_cost": 0.001,
            },
        },
    }

    # Initialize Trainer
    trainer = UnifiedTrainer(config=config, force=True)  # Overwrite existing models

    # Run Training
    success = trainer.train()

    if success:
        print("\n✅ Phase 4 Training Completed Successfully!")
        print(
            "Check tensorboard logs for 'eval/mean_reward' to see performance in realistic environment."
        )
    else:
        print("\n❌ Phase 4 Training Failed.")


if __name__ == "__main__":
    run_execution_aware_training()
