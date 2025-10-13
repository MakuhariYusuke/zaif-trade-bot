"""
SAC v397e Training Script - Reward Redesign

Changes from v397d:
1. Threshold: 0.20 -> 0.15 (BUY sensitivity increase)
2. Reward Scale: 1000.0 -> 100.0 (PnL component reduction)
3. Trade Bonus: 0.05 -> 0.5 (10x increase for signal strength)
4. Inactivity Penalty: 0.002 -> 0.02 (10x increase to reduce zero rewards)
5. Per-Trade Cost: 0.1 (new, to discourage excessive trading)

Expected Improvements:
- BUY/SELL balance (target: 20-30% each)
- Zero rewards reduction (target: <50%)
- Trade count control (target: 100-500)
- Reward signal strength (target: >10% positive rewards)
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ['MPLBACKEND'] = 'Agg'

from scripts.optimization.train_with_config import main as train_main


def main():
    config_path = project_root / "configs" / "sac_v397e_reward_redesign.json"
    
    print("=" * 80)
    print("SAC v397e Training - Reward Redesign")
    print("=" * 80)
    print(f"Config: {config_path}")
    print()
    print("🔧 Changes from v397d:")
    print("  1. Threshold: 0.20 -> 0.15 (BUY sensitivity increase)")
    print("  2. Reward Scale: 1000.0 -> 100.0 (PnL component reduction)")
    print("  3. Trade Bonus: 0.05 -> 0.5 (10x increase)")
    print("  4. Inactivity Penalty: 0.002 -> 0.02 (10x increase)")
    print("  5. Per-Trade Cost: 0.1 (new, to discourage excessive trading)")
    print()
    print("📊 Expected Improvements:")
    print("  - BUY/SELL balance: 20-30% each (was SELL 42.8% vs BUY 9.5%)")
    print("  - Zero rewards: <50% (was 94.6%)")
    print("  - Trade count: 100-500 (was 2,615)")
    print("  - Reward signal: >10% positive (was 0.68%)")
    print("=" * 80)
    print()
    
    sys.argv = ["train_with_config.py", "--config", str(config_path)]
    train_main()


if __name__ == "__main__":
    main()
