#!/usr/bin/env python3
"""
Debug training with detailed exception tracking
"""

import sys
import os
from pathlib import Path
import traceback
import signal

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Skip heavy imports
os.environ.setdefault("SKIP_HEAVY_IMPORTS", "1")
os.environ.setdefault("ZTB_SKIP_SCIPY", "1")
os.environ.setdefault("ZTB_SKIP_SKLEARN", "1")
os.environ.setdefault("ZTB_SAFE_DATETIME", "1")

print("="*70)
print("Debug Training Test - Exception Tracking")
print("="*70)

# Check for signal handlers
def signal_handler(signum, frame):
    print(f"\n🚨 Signal received: {signum}")
    print(f"   Frame: {frame}")
    traceback.print_stack(frame)
    sys.exit(1)

# Install signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print("\n✅ Signal handlers installed")

try:
    print("\n📦 Importing modules...")
    from ztb.training.unified_trainer.trainer import UnifiedTrainer
    from ztb.training.reward_config_schema import load_reward_config
    print("✅ Imports successful")
    
    # Load config
    print("\n📋 Loading configuration...")
    reward_config_path = project_root / "configs/rewards/stage1_basic.yaml"
    reward_settings = load_reward_config(reward_config_path)
    print("✅ Config loaded")
    
    # Create minimal config
    config = {
        "seed": 42,
        "training": {
            "algorithm": "SAC",
            "model_name": "debug_test",
            "output_dir": str(project_root / "results/debug_test"),
            "total_timesteps": 1000,  # Even smaller
            "eval_freq": 1000,
            "n_eval_episodes": 1,
            "log_interval": 100,
            "reward_settings": reward_settings,
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 5000,
                "learning_starts": 50,
                "batch_size": 32,
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
            },
            "walk_forward": {
                "enabled": False
            }
        }
    }
    
    print(f"\n🎯 Config: 1000 timesteps, batch_size=32, learning_starts=50")
    print("\n🚀 Creating trainer...")
    
    trainer = UnifiedTrainer(config)
    print("✅ Trainer created")
    
    print("\n▶️  Starting training...")
    print("    (Watching for interrupts...)")
    
    success = trainer.run()
    
    if success:
        print("\n✅ SUCCESS!")
    else:
        print("\n❌ FAILED (returned False)")
        
except KeyboardInterrupt as e:
    print(f"\n❌ KeyboardInterrupt caught in main:")
    print(f"   Type: {type(e)}")
    print(f"   Args: {e.args}")
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Exception caught in main:")
    print(f"   Type: {type(e)}")
    print(f"   Message: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("Complete!")
print("="*70)
