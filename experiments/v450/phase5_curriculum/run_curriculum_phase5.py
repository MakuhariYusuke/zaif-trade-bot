import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from ztb.training.unified_trainer import UnifiedTrainer


def run_curriculum_phase5():
    base_config_path = (
        project_root
        / "experiments"
        / "v450"
        / "phase4_1_stabilization"
        / "run_stabilization.py"
    )
    # We need to extract the config dict from the python script or just define it here.
    # Since run_stabilization.py is a script, not a json, we can't load it easily.
    # I will copy the config structure from run_stabilization.py and adapt it.

    # Base configuration (Phase 4.1 settings)
    base_config = {
        "model_name": "sac_v450_phase5_curriculum",
        "training": {
            "algorithm": "sac",
            "total_timesteps": 0,  # Will be set per stage
            "log_interval": 100,
            "checkpoint_interval": 1000,
            "checkpoint_dir": "",  # Will be set per stage
            "sac_hyperparameters": {
                "learning_rate": 1e-4,
                "batch_size": 512,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
            },
            "environment": {
                "config": {
                    "initial_portfolio_value": 200000.0,
                    "max_position_size": 1.0,
                    "transaction_cost": 0.001,
                    "reward_scaling": 1.0,
                    "timeframe": "1m",
                    "feature_set": "full",
                    "use_continuous_actions": True,
                    "action_space_type": "continuous",
                    "execution_model": {
                        "base_slippage": 0.0005,
                        "atr_slippage_factor": 0.5,
                        "base_latency_ms": 50.0,
                        "latency_jitter_ms": 20.0,
                    },
                    "bankruptcy_threshold": 2000.0,
                    "bankruptcy_penalty": 1000.0,
                    "drawdown_penalty_threshold": 0.2,
                    "drawdown_penalty_factor": 0.1,
                    "curriculum_stage": "simple",  # Default, will be overridden
                },
                "feature_set": "full",
            },
            "data_config": {
                "data_path": str(project_root / "data" / "btc_jpy_1m_dataset.csv"),
            },
            "evaluation": {
                "enabled": True,
                "eval_freq": 1000,
                "n_eval_episodes": 3,
                "deterministic": True,
                "config": {
                    "execution_model": {
                        "base_slippage": 0.0005,
                        "atr_slippage_factor": 0.5,
                        "base_latency_ms": 50.0,
                        "latency_jitter_ms": 20.0,
                    },
                    "initial_portfolio_value": 200000.0,
                    "bankruptcy_threshold": 2000.0,
                },
            },
        },
    }

    # Define Curriculum Stages
    stages = [
        # {
        #     "name": "stage1_discovery",
        #     "steps": 5000,
        #     "curriculum_stage": "action_discovery",
        #     "description": "Encourage action taking, ignore costs."
        # },
        # {
        #     "name": "stage2_forced_balance",
        #     "steps": 5000, # Cumulative 10000
        #     "curriculum_stage": "forced_balance",
        #     "description": "Enforce BUY/SELL balance."
        # },
        # {
        #     "name": "stage3_balanced_transition",
        #     "steps": 10000, # Cumulative 20000
        #     "curriculum_stage": "balanced_transition",
        #     "description": "Introduce PnL with balance constraints."
        # },
        {
            "name": "stage4_pnl_focused",
            "steps": 30000,  # Cumulative 50000
            "curriculum_stage": "pnl_focused",
            "description": "Full PnL optimization.",
        }
    ]

    base_checkpoint_dir = project_root / "models" / "checkpoints" / "phase5_curriculum"
    os.makedirs(base_checkpoint_dir, exist_ok=True)

    # Manually set previous checkpoint from Stage 3
    # Note: Update the timestamp in the filename if it changes
    previous_checkpoint_path = str(
        project_root
        / "models/checkpoints/phase5_curriculum/stage3_balanced_transition/training_state_10000_20251207_185943.pkl"
    )

    for i, stage in enumerate(stages):
        print(
            f"\n=== Starting Curriculum Stage {i+4}: {stage['name']} ==="
        )  # Adjusted index for display
        print(f"Goal: {stage['description']}")
        print(f"Steps: {stage['steps']} (Stage specific)")

        stage_checkpoint_dir = base_checkpoint_dir / stage["name"]
        os.makedirs(stage_checkpoint_dir, exist_ok=True)

        # Prepare Config
        config = base_config.copy()
        config["model_name"] = f"sac_v450_phase5_{stage['name']}"
        config["training"]["total_timesteps"] = stage["steps"]
        config["training"]["checkpoint_dir"] = str(stage_checkpoint_dir)
        config["checkpoint_dir"] = str(
            stage_checkpoint_dir
        )  # Set top-level for UnifiedTrainer resume logic
        config["training"]["environment"]["config"]["curriculum_stage"] = stage[
            "curriculum_stage"
        ]

        # Resume logic
        resume = False
        if previous_checkpoint_path and os.path.exists(previous_checkpoint_path):
            print(
                f"Resuming from previous stage checkpoint: {previous_checkpoint_path}"
            )

            # Use SACTrainer's built-in resume logic
            config["training"]["resume_from"] = previous_checkpoint_path

            # We DO NOT set resume=True for UnifiedTrainer, because it uses incompatible logic.
            # SACTrainer will handle it.
            resume = False
        else:
            print(
                f"Warning: Previous checkpoint not found at {previous_checkpoint_path}. Starting from scratch."
            )

        # We need to pass 'resume' to UnifiedTrainer constructor or config?
        # UnifiedTrainer(config, resume=True)

        trainer = UnifiedTrainer(config, resume=resume)
        success = trainer.train()

        if not success:
            print(f"Stage {stage['name']} failed! Stopping curriculum.")
            break

        # Find the latest checkpoint from this stage to pass to next stage
        # We can use TrainingCheckpointManager to find it
        from ztb.training.checkpoint.checkpoint_manager import TrainingCheckpointManager

        ckpt_manager = TrainingCheckpointManager(save_dir=str(stage_checkpoint_dir))
        latest_snapshot = ckpt_manager.load_latest()

        if latest_snapshot:
            # Manually find the file path since snapshot doesn't contain it
            import re

            checkpoints = list(stage_checkpoint_dir.glob("checkpoint_*.pkl*"))
            if checkpoints:

                def get_step(p):
                    match = re.search(r"checkpoint_(\d+)", p.name)
                    if match:
                        return int(match.group(1))
                    return -1

                latest_checkpoint_path = max(checkpoints, key=get_step)
                previous_checkpoint_path = str(latest_checkpoint_path)
                print(
                    f"Stage {stage['name']} completed. Latest checkpoint: {previous_checkpoint_path}"
                )
            else:
                print(
                    f"Warning: Snapshot loaded but file not found in {stage_checkpoint_dir}"
                )
                previous_checkpoint_path = None
        else:
            print(
                f"Warning: No checkpoint found after stage {stage['name']}. Next stage will start from scratch (if any)."
            )
            previous_checkpoint_path = None


if __name__ == "__main__":
    run_curriculum_phase5()
