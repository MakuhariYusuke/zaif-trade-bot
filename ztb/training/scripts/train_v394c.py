"""
v394c (バランス調整版) を訓練
"""

import json

from ztb.training.unified_trainer import UnifiedTrainer


def main():
    config_path = "configs/ppo_v394c_balanced.json"

    print("=== v394c Training (Balanced) ===")
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
