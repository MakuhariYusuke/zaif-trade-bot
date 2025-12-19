"""
v394c (バランス調整版) を訓練
"""

from pathlib import Path

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.file_utils import safe_json_load


def main():
    config_path = "configs/ppo_v394c_balanced.json"

    print("=== v394c Training (Balanced) ===")
    print(f"Config: {config_path}\n")

    # 設定ファイル読み込み
    config = safe_json_load(Path(config_path))

    trainer = UnifiedTrainer(config)
    result = trainer.train()

    print("\n=== Training Complete ===")
    if result:
        print("Training finished")


if __name__ == "__main__":
    main()
