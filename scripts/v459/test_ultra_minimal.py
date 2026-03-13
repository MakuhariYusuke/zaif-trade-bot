#!/usr/bin/env python3
"""
Ultra-minimal test - No Walk-Forward, minimal timesteps
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Avoid heavy optional imports during minimal tests
os.environ.setdefault("SKIP_HEAVY_IMPORTS", "1")
os.environ.setdefault("ZTB_SKIP_SCIPY", "1")
os.environ.setdefault("ZTB_SKIP_SKLEARN", "1")
os.environ.setdefault("ZTB_SAFE_DATETIME", "1")
os.environ.setdefault(
    "ZTB_SIGINT_POLICY", "ignore" if os.name == "nt" else "default"
)

print("="*70)
print("Ultra-Minimal Training Test")
print("="*70)

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.reward_config_schema import load_reward_config

# Load reward config
reward_config_path = project_root / "configs/rewards/stage1_basic.yaml"
print(f"\nLoading: {reward_config_path.name}")
from ztb.training.reward_config_schema import RewardConfigSchema
reward_dict = RewardConfigSchema.load_and_validate(reward_config_path)
print("✅ Config loaded (validated)")

# Ultra-minimal configuration
config = {
    "seed": 42,
    "training": {
        "algorithm": "SAC",
        "model_name": "test_minimal",
        "output_dir": str(project_root / "results/test_minimal"),
        "total_timesteps": 2000,  # Very small
        "eval_freq": 2000,  # No eval during training
        "n_eval_episodes": 1,
        "log_interval": 500,
        "sac_hyperparameters": {
            "learning_rate": 0.0003,
            "buffer_size": 10000,  # Very small
            "learning_starts": 100,  # Start fast
            "batch_size": 64,  # Small batch
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
        },
        "data_config": {
            "data_path": str(project_root / "data" / "btc_jpy_1m_v451.csv"),
            "window_size": 60
        },
        "environment": {
            "use_continuous_actions": True,
            "action_space_type": "continuous",
            "reward_settings": reward_dict
        },
        "walk_forward": {
            "enabled": False  # DISABLED
        }
    }
}

print(f"\n📋 Configuration:")
print(f"   Timesteps: 2000")
print(f"   Walk-Forward: DISABLED")
print(f"   Buffer: 10000")
print(f"   Batch: 64")

print("\n🚀 Starting...")
print("="*70)

import time
start = time.time()

try:
    trainer = UnifiedTrainer(config)
    success = trainer.run()
    
    elapsed = time.time() - start
    
    if success:
        print(f"\n✅ SUCCESS in {elapsed:.1f}s!")
        
        report = trainer.get_training_report()
        stats = report.get("training_stats", {})
        
        print(f"\n📊 Stats:")
        print(f"   Total timesteps: {stats.get('total_timesteps', 'N/A')}")
        print(f"   Training time: {stats.get('training_time', 'N/A')}s")
        print(f"   Final reward: {stats.get('final_reward', 'N/A')}")
        
        # Save
        import json
        output = project_root / "results/test_minimal/result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump({"success": True, "stats": stats}, f, indent=2, default=str)
        print(f"\n💾 Saved: {output}")
        
    else:
        print(f"\n❌ FAILED after {elapsed:.1f}s")
        sys.exit(1)

except KeyboardInterrupt:
    print("\n⚠️ KeyboardInterrupt caught!")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ Complete!")
print("="*70)
