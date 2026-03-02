"""
v394f (超激辛版 - Ultra High Entropy) を訓練
ent_coef = 0.2 (v394dの20倍) で探索を最大限維持
"""

import json

from ztb.training.unified_trainer import UnifiedTrainer

def main():
    config_path = "configs/ppo_v394f_ultra_entropy.json"

    print("=" * 80)
    print("🔥🔥🔥 v394f Training (ULTRA HIGH ENTROPY: ent_coef=0.2) 🔥🔥🔥")
    print("=" * 80)
    print()
    print(f"Config: {config_path}")
    print()

    print("🚀 Key Changes from v394d:")
    print("  - ent_coef: 0.01 → 0.2 (20x) 🔥🔥🔥")
    print("  - Same aggressive reward settings as v394d")
    print("  - Goal: Maintain entropy >1.0 throughout training")
    print("  - Target: HOLD ratio <70% at 100k timesteps")
    print()

    print("📊 v394d Results (for comparison):")
    print("  - Initial (2k steps): HOLD 50% ✅")
    print("  - Final (100k steps): HOLD 89% 🚨")
    print("  - Final entropy: 0.61 (too low)")
    print()

    print("🎯 v394f Expected Results:")
    print("  - Initial: HOLD ~50-60%")
    print("  - Final: HOLD <70% (maintained!)")
    print("  - Final entropy: >1.0 (exploration maintained)")
    print()
    print("🚀 Starting training...")
    print()

    # 設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    trainer = UnifiedTrainer(config)
    result = trainer.train()

    print(f"\n{'='*80}")
    print("=== Training Complete ===")
    print(f"{'='*80}")
    if result:
        print("✅ v394f training finished!")
        print()
        print("Next steps:")
        print("  1. Analyze final Action distribution")
        print("  2. Run Stochastic backtest (deterministic=False)")
        print("  3. Compare with v394d results")

if __name__ == "__main__":
    main()
