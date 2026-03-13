"""
Short training script for reward function improvements (v378, v379, v380)
Run all three reward configurations with short 5k timesteps for action distribution analysis
"""
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.utils.training_utils import display_training_complete

CONFIGS = [
    {
        "name": "v378_scale_short",
        "path": "configs/training/ppo_reward_v378_scale_short.json",
        "description": "Scale-adjusted (HOLD penalty 4x, profit 3x, trading bonus 3x) - SHORT RUN",
    },
    {
        "name": "v379_dynamic_short",
        "path": "configs/training/ppo_reward_v379_dynamic_short.json",
        "description": "Dynamic market-adaptive (v378 + volatility/trend scaling) - SHORT RUN",
    },
    {
        "name": "v380_aggressive_short",
        "path": "configs/training/ppo_reward_v380_aggressive_short.json",
        "description": "Aggressive anti-HOLD (HOLD penalty 10x, profit 5x, trading bonus 6x) - SHORT RUN",
    },
]

def run_training(config_info):
    """Run training for a single config."""
    config_path = project_root / config_info["path"]

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    session_id = config["session_id"]

    print("=" * 80)
    print(f"Training: {config_info['name']} (SHORT RUN)")
    print("=" * 80)
    print(f"Description: {config_info['description']}")
    print(f"Session ID: {session_id}")
    print(f"Total timesteps: {config['training']['total_timesteps']}")
    print(f"Features: {config.get('curated_features_list', 'default')}")
    print(f"Random start: {config['environment']['random_start']}")
    print("\nReward settings:")
    for key, value in config["environment"]["reward_settings"].items():
        print(f"  {key}: {value}")
    print("=" * 80 + "\n")

    # Import trainer
    from ztb.training.core.ppo_trainer import PPOTrainer

    # Create trainer
    print("Creating trainer...")
    trainer = PPOTrainer(
        data_path=str(config["data_path"]),
        config=config,
        checkpoint_dir=f"checkpoints/{session_id}",
    )

    # Train
    print("Starting training...")
    start_time = time.time()
    trainer.train(session_id=session_id)
    training_time = time.time() - start_time

    print("\n" + "=" * 80)
    display_training_complete({"session_id": session_id}, training_time)
    print("=" * 80 + "\n")

def main():
    print("=" * 80)
    print("Reward Function Improvements Training - SHORT RUNS")
    print("=" * 80)
    print("Purpose: Generate TensorBoard action distribution metrics")
    print(f"Configs to train: {len(CONFIGS)}")
    for i, cfg in enumerate(CONFIGS, 1):
        print(f"{i}. {cfg['name']}: {cfg['description']}")
    print("=" * 80 + "\n")

    for i, config_info in enumerate(CONFIGS, 1):
        print(f"\n{'#' * 80}")
        print(f"# Training {i}/{len(CONFIGS)}: {config_info['name']}")
        print(f"{'#' * 80}\n")

        try:
            run_training(config_info)
        except KeyboardInterrupt:
            print(f"\n⚠️ Training interrupted by user for: {config_info['name']}")
            response = input("Continue with next config? (y/n): ").strip().lower()
            if response != "y":
                print("Training sequence cancelled.")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ Training failed for: {config_info['name']}")
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

            response = input("Continue with next config? (y/n): ").strip().lower()
            if response != "y":
                print("Training sequence cancelled.")
                sys.exit(1)

    print("\n" + "=" * 80)
    print("ALL SHORT TRAINING COMPLETED!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check TensorBoard logs for pan_action_counts/* metrics")
    print("2. Run inspect_tensorboard_tags.py to verify action distribution data")
    print("3. Compare HOLD/BUY/SELL ratios across v378/v379/v380")
    print("4. Proceed to risk metrics and backtesting")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Training sequence failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
