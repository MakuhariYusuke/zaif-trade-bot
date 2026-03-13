#!/usr/bin/env python3
"""
Simple single experiment test for debugging
最小構成でのテスト実行
"""

import sys
import os
from pathlib import Path

# Project root setup
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

from ztb.training.unified_trainer.trainer import UnifiedTrainer

# 最小構成のconfig
config = {
    "training": {
        "algorithm": "SAC",
        "total_timesteps": 5000,  # 極小テスト
        "eval_freq": 1000,
        "n_eval_episodes": 1,
        "log_interval": 100,
        
        "sac_hyperparameters": {
            "learning_rate": 0.0003,
            "buffer_size": 5000,
            "learning_starts": 100,
            "batch_size": 64,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto"
        },
        
        "data_config": {
            "data_path": str(project_root / "data" / "btc_jpy_1m_v451.csv"),
            "window_size": 60,
            "train_end_index": 10000,  # 小さいデータセット
        },
        
        "environment": {
            "use_continuous_actions": True,  # ✅ CRITICAL: SAC requires continuous
            "action_space_type": "continuous",
            "config": {
                "initial_portfolio_value": 1000000.0,
                "transaction_cost": 0.0,
                "max_position_size": 1.0,
            }
        },
        
        "model_name": "test_single_sac",
        "output_dir": str(project_root / "results" / "test_single")
    }
}

def main():
    print("="*70)
    print("Single Experiment Test - Minimal Configuration")
    print("="*70)
    print(f"Algorithm: {config['training']['algorithm']}")
    print(f"Timesteps: {config['training']['total_timesteps']}")
    print(f"Use Continuous Actions: {config['training']['environment']['use_continuous_actions']}")
    print(f"Data: {config['training']['data_config']['data_path']}")
    print("="*70)
    
    try:
        trainer = UnifiedTrainer(config)
        print("\n✅ Trainer initialized successfully")
        
        result = trainer.run()
        print("\n✅ Training completed successfully")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
