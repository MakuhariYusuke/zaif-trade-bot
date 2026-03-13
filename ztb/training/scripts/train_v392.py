"""v392訓練スクリプト - random_startバグ修正後"""

# ruff: noqa: E402
import os

os.environ["MPLBACKEND"] = "Agg"  # matplotlibバックエンド設定

import json
import logging
import time

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.logging_utils import setup_logging
from ztb.utils.training_utils import display_training_complete

setup_logging()

if __name__ == "__main__":
    start_time = time.time()

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

    training_time = time.time() - start_time
    final_metrics = {
        "model_path": result.get('model_path') if result else None,
        "training_success": bool(result),
    }
    display_training_complete(final_metrics, training_time)
        print("\n❌ Training failed or was cancelled")
