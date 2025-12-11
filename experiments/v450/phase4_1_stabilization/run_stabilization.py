"""
Phase 4.1: Stabilization Training.

This script trains a SAC model with stabilized hyperparameters and enhanced bankruptcy/drawdown penalties.
It builds upon the Phase 4 execution-aware training but aims for stability over raw performance.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from ztb.training.unified_trainer.trainer import UnifiedTrainer


def run_stabilization_training():
    print("========================================================")
    print("🚀 Phase 4.1: Stabilization Training (v450)")
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
        "model_name": "sac_v450_stable",
        "training": {
            "algorithm": "sac",
            "total_timesteps": 50000,  # Increased for stability verification
            "log_interval": 1000,
            "checkpoint_interval": 2000,  # More frequent checkpoints
            "checkpoint_dir": os.path.join(
                project_root, "models", "checkpoints", "phase4_1"
            ),
            # Hyperparameters for Stabilization (Nested in sac_hyperparameters)
            "sac_hyperparameters": {
                "learning_rate": 1e-4,  # Reduced from 3e-4
                "batch_size": 512,  # Increased from 256 (default)
                "buffer_size": 100000,
                "learning_starts": 1000,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
            },
            # Data Configuration
            "data_config": {
                # Use the expanded dataset
                "data_path": os.path.join(
                    project_root, "data", "btc_jpy_1m_dataset.csv"
                ),
            },
            # Environment Configuration (TRAINING - REALISTIC)
            "environment": {
                "config": {
                    "initial_portfolio_value": 200000.0,  # 200k JPY
                    "max_position_size": 1.0,  # 1.0 BTC (but limited by funds)
                    "transaction_cost": 0.001,  # 0.1% fee
                    "reward_scaling": 1.0,
                    "timeframe": "1m",
                    "feature_set": "full",
                    "use_continuous_actions": True,  # Required for SAC
                    "action_space_type": "continuous",
                    # Execution Model (Always enabled for Phase 4)
                    "execution_model": realistic_execution_model,
                    # Bankruptcy & Drawdown Settings (New)
                    "bankruptcy_threshold": 2000.0,
                    "bankruptcy_penalty": 1000.0,
                    "drawdown_penalty_threshold": 0.20,  # 20% drawdown triggers penalty
                    "drawdown_penalty_factor": 0.1,  # Penalty multiplier
                }
            },
            # Evaluation Configuration
            "evaluation": {
                "enabled": True,
                "eval_freq": 2000,
                "n_eval_episodes": 3,
                "deterministic": True,
                "config": {
                    # Evaluation uses the same realistic settings
                    "execution_model": realistic_execution_model,
                    "initial_portfolio_value": 200000.0,
                    "bankruptcy_threshold": 2000.0,
                },
            },
        },
    }

    # Initialize Trainer
    trainer = UnifiedTrainer(config)

    # Start Training
    try:
        trainer.train()
        print("\n✅ Phase 4.1 Stabilization Training Completed Successfully!")
    except Exception as e:
        print(f"\n❌ Training Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_stabilization_training()
