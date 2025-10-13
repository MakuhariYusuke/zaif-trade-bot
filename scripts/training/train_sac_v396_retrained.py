"""
SAC v396 Retrained Script

設計:
- max_position_size: 1.0 (100% position)
- initial_balance: 200,000円
- reward_settings: v397i style with balance_penalty
- curriculum_stage: balanced_transition
- target_ratios: [0.5, 0.25, 0.25] (HOLD 50%, BUY 25%, SELL 25%)
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ['MPLBACKEND'] = 'Agg'

from scripts.optimization.train_with_config import main as train_main

def main():
    config_path = project_root / "configs" / "sac_v396_retrained.json"
    print("=" * 80)
    print("SAC v396 Retrained - Max Position 100% with Balance Penalty")
    print("=" * 80)
    print(f"Config: {config_path}")
    print("  max_position_size: 1.0 (100%)")
    print("  total_timesteps: 100,000")
    print("  curriculum_stage: balanced_transition")
    print("  target_ratios: [0.5, 0.25, 0.25]")
    print("  balance_penalty: 1500.0")
    print("  balance_penalty_tolerance: 0.05")
    print("  inactivity_penalty_rate: 0.00005")
    print("=" * 80)

    # 訓練スクリプトを使用
    sys.argv = ["train_with_config.py", "--config", str(config_path)]
    train_main()

if __name__ == "__main__":
    main()