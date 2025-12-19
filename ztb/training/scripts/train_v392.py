"""v392訓練スクリプト - random_startバグ修正後"""

# ruff: noqa: E402
import os

os.environ["MPLBACKEND"] = "Agg"  # matplotlibバックエンド設定

import json
import logging

logging.basicConfig(level=logging.INFO)

from ztb.training.unified_trainer import UnifiedTrainer

if __name__ == "__main__":
    config_path = "configs/ppo_profitable_v392_bugfix.json"

    print(f"🚀 Starting v392 training with config: {config_path}")
    print("🔧 Bug fixes applied:")
    print("   ✅ random_start: position argument fix")
    print("   ✅ Random buffer: dynamic calculation (10% of data)")
    print("\n訓練を開始します...\n")

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    trainer = UnifiedTrainer(config)
    result = trainer.train()

    if result:
        print("\n✅ Training completed!")
        print(f"   Model saved: {result.get('model_path')}")
    else:
        print("\n❌ Training failed or was cancelled")
