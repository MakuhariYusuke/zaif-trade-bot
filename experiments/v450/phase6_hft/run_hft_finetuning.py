import sys
from pathlib import Path

# Import torch first to avoid DLL issues
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from ztb.training.unified_trainer import UnifiedTrainer


def run_hft_finetuning():
    # Base configuration derived from Phase 5
    config = {
        "model_name": "sac_v450_phase6_hft",
        "training": {
            "algorithm": "sac",
            "total_timesteps": 10000,  # Reduced to 10k steps for efficiency
            "log_interval": 100,
            "checkpoint_interval": 2000,  # Frequent checkpoints
            "checkpoint_dir": "checkpoints/v450/phase6",
            "sac_hyperparameters": {
                "learning_rate": 5e-5,  # Slightly higher LR for faster adaptation
                "batch_size": 2048,  # Larger batch for stability with more data
                "buffer_size": 200000,  # Increased buffer
                "learning_starts": 0,  # Start learning immediately
                "tau": 0.005,
                "gamma": 0.80,  # Gamma 0.80: Revert to Iter 9 (Scalping focus)
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": 0.05,  # Entropy 0.05: Revert to Iter 9 (High exploration)
            },
            "environment": {
                "config": {
                    "initial_portfolio_value": 200000.0,
                    "max_position_size": 1.0,
                    "transaction_cost": 0.0005,  # 0.05% Fee
                    "reward_scaling": 1.0,
                    "timeframe": "1m",
                    "feature_set": "full",
                    "use_continuous_actions": True,
                    "action_space_type": "continuous",
                    "curriculum_stage": "trading_focused",  # Force trading focused reward
                    "reward_settings": {
                        "behavior": {
                            "trading_focused": {
                                "hold_penalty_rate": 0.0,
                                "trading_bonus_multiplier": 1.0,  # Enable bonus
                                "balance_penalty": 0.0,  # Disable balance penalty (No forced HOLD)
                                "fee_penalty": 0.0,  # Disable explicit fee penalty (PnL handles it)
                            }
                        }
                    },
                    "execution_model": {
                        "base_slippage": 0.0001,  # Minimal slippage
                        "atr_slippage_factor": 0.1,
                        "base_latency_ms": 10.0,
                        "latency_jitter_ms": 5.0,
                    },
                    "bankruptcy_threshold": 2000.0,
                    "bankruptcy_penalty": 1000.0,
                    "drawdown_penalty_threshold": 0.05,  # Tighter drawdown tolerance
                    "drawdown_penalty_factor": 0.5,
                },
                "feature_set": "full",
            },
            "data_config": {
                "data_path": str(project_root / "data" / "btc_jpy_1m_dataset.csv"),
            },
            "evaluation": {
                "enabled": True,
                "eval_freq": 2000,
                "n_eval_episodes": 3,
                "deterministic": True,
                "config": {
                    "transaction_cost": 0.0000,  # Eval with same low fee
                    "initial_portfolio_value": 200000.0,
                },
            },
        },
    }

    # Initialize trainer
    trainer = UnifiedTrainer(config)

    # Setup trainer (initialize algorithm trainer)
    # UnifiedTrainer.run() calls _execute_training() which creates algorithm_trainer.
    # But we need to inject the pretrained model BEFORE training starts.
    # We can manually trigger the algorithm creation.

    # Create algorithm trainer manually
    from ztb.training.unified_trainer.algorithms import create_algorithm_trainer

    trainer.algorithm_trainer = create_algorithm_trainer("sac", config, trainer.logger)

    # Load the Phase 5 Stage 4 model
    prev_model_path = project_root / "models" / "sac_v450_phase5_stage4_pnl_focused.zip"
    print(f"Loading Phase 5 model from: {prev_model_path}")

    # Load model with custom objects to override hyperparameters
    from stable_baselines3 import SAC

    # Prepare custom objects for hyperparameter override
    # Note: We exclude ent_coef here because changing it from 'auto' to float causes structure mismatch during load.
    # We will patch it manually after loading.
    custom_objects = {
        "learning_rate": config["training"]["sac_hyperparameters"]["learning_rate"],
        "gamma": config["training"]["sac_hyperparameters"]["gamma"],
        "batch_size": config["training"]["sac_hyperparameters"]["batch_size"],
    }

    print(f"Overriding hyperparameters (except ent_coef): {custom_objects}")

    # Load the model
    # We don't pass env here; trainer.train() will set the new environment.
    model = SAC.load(str(prev_model_path), custom_objects=custom_objects)

    # Manually patch entropy coefficient to fixed value (0.05)
    new_ent_coef = config["training"]["sac_hyperparameters"]["ent_coef"]
    print(f"Manually patching ent_coef to fixed value: {new_ent_coef}")

    model.ent_coef_optimizer = None  # Disable auto-tuning
    model.ent_coef_tensor = torch.tensor(float(new_ent_coef)).to(model.device)
    model.ent_coef = new_ent_coef

    trainer.algorithm_trainer.model = model

    print("Model loaded successfully with new hyperparameters.")

    print("Starting Phase 6 HFT Fine-tuning...")
    # We call train() on the algorithm trainer directly
    trainer.algorithm_trainer.train(
        total_timesteps=config["training"]["total_timesteps"]
    )

    # Save final model
    save_path = project_root / "models" / "sac_v450_phase6_hft.zip"
    trainer.algorithm_trainer.model.save(str(save_path))
    print(f"Phase 6 model saved to {save_path}")


if __name__ == "__main__":
    run_hft_finetuning()
