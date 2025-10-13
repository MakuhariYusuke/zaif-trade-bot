"""
SAC v397c_fixed_scale Training Script
max_position_size=1.0を維持しつつRewardCalculatorの正規化を活用

重要な変更点:
1. reward_scale: 200.0 → 1000.0 (RewardCalculatorのデフォルト設計に合わせる)
   - pnl_ratio正規化により、max_position_sizeに依存しない
   - 0.1% profit = reward 10.0 の設計
2. ボーナス/ペナルティを適切に調整
   - inactivity_penalty_rate: 0.001 → 0.01 (×10)
   - opportunity_cost_rate: 0.0005 → 0.005 (×10)
   - trade_execution_bonus_rate: 0.05 → 0.05 (維持)
3. クリッピング範囲: [-2.0, 2.0] → [-10.0, 10.0] (×5)
4. continuous_to_discrete_threshold: 0.15 → 0.25 (適度に)

スケール設計哲学:
- PnL (0.1% price change, position=1.0) ≈ 1,000円 → pnl_ratio=0.01 → reward = 10.0
- Trade Bonus (position_change > 0.01) = 0.05 (PnLの0.5%)
- Inactivity Penalty (3 steps) = -0.03 (PnLの0.3%)
- Opportunity Cost (position=1.0) = -0.005 (PnLの0.05%)
→ バランス: PnL : Bonus : Penalty : Cost ≈ 10.0 : 0.05 : 0.03 : 0.005
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ['MPLBACKEND'] = 'Agg'

from scripts.optimization.train_with_config import main

if __name__ == "__main__":
    config_path = project_root / "configs" / "sac_v397c_fixed_scale.json"
    
    print("=" * 80)
    print("SAC v397c_fixed_scale Training")
    print("=" * 80)
    print(f"Config: {config_path}")
    print("\n重要な設計:")
    print("  max_position_size: 1.0 (維持)")
    print("  reward_scale: 1000.0 (RewardCalculatorのpnl_ratio正規化を活用)")
    print("  ボーナス/ペナルティ: PnLの0.5-3%程度に調整")
    print("\n期待されるバランス:")
    print("  PnL : Bonus : Penalty : Cost ≈ 10.0 : 0.05 : 0.03 : 0.005")
    print("=" * 80)
    
    sys.argv = ["train_with_config.py", "--config", str(config_path)]
    main()
