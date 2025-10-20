"""
v394e (高エントロピー版) を訓練
"""

import json

from ztb.training.unified_trainer import UnifiedTrainer


def main():
    config_path = "configs/ppo_v394e_high_entropy.json"

    print("=== v394e Training (High Entropy: ent_coef=0.05) ===")
    print(f"Config: {config_path}\n")

    # 設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("🔥 Key Changes from v394a:")
    print("  - ent_coef: 0.01 → 0.05 (5x)")
    print("  - Trade rewards: v394d level")
    print("  - Expected: Maintain exploration throughout training\n")

    trainer = UnifiedTrainer(config)
    result = trainer.train()

    print("\n=== Training Complete ===")
    if result:
        print("Training finished")


if __name__ == "__main__":
    main()
