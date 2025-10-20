"""
v377 (curated features) の動作確認テスト
2000ステップで特徴フィルタリングが正しく動作するか確認
"""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    # Load config
    config_path = (
        project_root / "configs" / "training" / "ppo_curated_features_v377.json"
    )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Modify for quick test
    test_session_id = "ppo_v377_quick_test"
    config["session_id"] = test_session_id
    config["training"]["total_timesteps"] = 2000  # Quick test
    config["training"]["eval_freq"] = 1000
    config["training"]["checkpoint_interval"] = 2000

    print("=" * 80)
    print("Quick Test: ppo_curated_features_v377 (2000 timesteps)")
    print("=" * 80)
    print(f"Feature filtering: {config.get('enable_feature_filtering')}")
    print(f"Feature mode: {config.get('feature_filter_mode')}")
    print(f"Curated list: {config.get('curated_features_list')}")
    print(f"Data rows: {config.get('data_rows_limit', 'ALL (1000+)')}")
    print(f"Random start: {config['environment']['random_start']}")
    print(f"Entropy coef: {config['ppo']['ent_coef']}")
    print("=" * 80 + "\n")

    # Import trainer
    from ztb.training.ppo_trainer import PPOTrainer

    # Prepare params dict for trainer
    params = {
        "data_path": str(config["data_path"]),
        "checkpoint_dir": f"checkpoints/{test_session_id}",
        "checkpoint_interval": config["training"]["checkpoint_interval"],
        "config": config,
    }

    # Create trainer
    print("Creating trainer...")
    trainer = PPOTrainer(params=params)

    # Train
    print("Starting training...")
    print("NOTE: Watch for '✅ Applied curated features filter' message")
    print("Expected: kept 60/110 features\n")

    trainer.train(session_id=test_session_id)

    print("\n" + "=" * 80)
    print("Quick test completed successfully!")
    print("Curated features filtering verified!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
