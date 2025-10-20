"""
v394a (HOLD罰則強化版) を訓練
"""

import json

from ztb.training.unified_trainer import UnifiedTrainer


def main():
    config_path = "configs/ppo_v394a_hold_penalty.json"

    print("=== v394a Training (HOLD Penalty x5) ===")
    print(f"Config: {config_path}\n")

    # 設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    trainer = UnifiedTrainer(config)
    result = trainer.train()

    print("\n=== Training Complete ===")
    if result:
        print("Training finished")


if __name__ == "__main__":
    main()
