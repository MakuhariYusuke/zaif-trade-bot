#!/usr/bin/env python3
"""
Minimal AB Experiment Runner - Bypass import issues
"""

import sys
import os

# Disable scipy imports temporarily
os.environ['SKIP_SCIPY_IMPORT'] = '1'

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

print("="*70)
print("Minimal AB Reward Experiment Runner")
print("="*70)

# Import only what's needed
from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.reward_config_schema import load_reward_config

# Load reward config
reward_config_path = project_root / "configs/rewards/stage1_basic.yaml"
print(f"\nLoading reward config: {reward_config_path}")
from ztb.training.reward_config_schema import RewardConfigSchema
reward_dict = RewardConfigSchema.load_and_validate(reward_config_path)
behavior_opt = reward_dict.pop("behavior_optimization", None)
print("✅ Reward config loaded (validated)")

# Create minimal config
config = {
    "seed": 42,
    "experiment_name": "test_stage1_seed42",
    "training": {
        "algorithm": "SAC",
        "model_name": "ab_reward_stage1_s42",
        "output_dir": str(project_root / "results/ab_rewards/stage1_basic/seed_42"),
        "total_timesteps": 5000,
        "eval_freq": 5000,
        "n_eval_episodes": 3,
        "log_interval": 100,
        "environment": {
            "use_continuous_actions": True,
            "action_space_type": "continuous",
            "initial_portfolio_value": 100000.0,
            "transaction_cost": 0.0,
            "reward_settings": reward_dict
        },

        "sac_hyperparameters": {
            "learning_rate": 0.0003,
            "buffer_size": 50000,
            "learning_starts": 500,
            "batch_size": 128,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "target_update_interval": 1,
            "target_entropy": "auto"
        },

        "data_config": {
            "data_path": str(project_root / "data" / "btc_jpy_1m_v451.csv"),
            "window_size": 60
        },
        "environment": {
            "use_continuous_actions": True,
            "action_space_type": "continuous",
            "initial_portfolio_value": 1000000.0,
            "transaction_cost": 0.0
        },
        "walk_forward": {
            "enabled": True,
            "n_splits": 4,
            "train_size": 0.6,
            "validation_size": 0.2,
            "test_size": 0.2
        }
    }
}

print("\n🚀 Starting training...")
print("="*70)

try:
    trainer = UnifiedTrainer(config)
    print("✅ Trainer initialized")
    
    success = trainer.run()
    
    if success:
        print("\n✅ Training completed successfully!")
        
        # Get report
        report = trainer.get_training_report()
        
        # Extract metrics
        training_stats = report.get("training_stats", {})
        print(f"\nTraining Stats Keys: {list(training_stats.keys())}")
        
        # Save minimal report
        import json
        output_file = project_root / "results/ab_rewards/test_result.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "experiment": config["experiment_name"],
                "success": True,
                "training_stats": training_stats
            }, f, indent=2, default=str)
        
        print(f"\n📊 Results saved to: {output_file}")
    else:
        print("\n❌ Training failed")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ Experiment complete!")
print("="*70)
