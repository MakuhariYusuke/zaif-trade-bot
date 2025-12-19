"""
SAC v395d (Optimal) - Best of Both Worlds
v395aの低いLoss + v395b/cの安定したent_coef
"""
import logging
import time
from pathlib import Path

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.file_utils import safe_json_load
from ztb.utils.logging_utils import setup_logging
from ztb.utils.training_utils import display_training_complete

setup_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info("🎯 SAC v395d - Optimal Parameter Set")
    logger.info("=" * 80)

    start_time = time.time()

    config_path = "configs/sac_v395d_optimal.json"

    # 設定ファイル読み込み
    config = safe_json_load(Path(config_path))

    logger.info("📊 Parameter Selection Strategy:")
    logger.info("-" * 80)
    for param, choice in config["analysis_summary"]["parameter_choices"].items():
        logger.info(f"  • {param:20s}: {choice}")
    logger.info("")

    logger.info("🎯 Expected Outcomes:")
    print("-" * 80)
    for metric, target in config["analysis_summary"]["expected_outcomes"].items():
        print(f"  • {metric:20s}: {target}")
    print()

    logger.info("🚀 Starting 5k timesteps training...")
    logger.info("=" * 80)
    trainer = UnifiedTrainer(config)
    result = trainer.train()

    training_time = time.time() - start_time
    final_metrics = {
        "model_path": result.get('model_path', 'N/A') if result else None,
        "training_success": bool(result),
    }
    display_training_complete(final_metrics, training_time)
        print("  1. Run: python compare_three_sac_versions.py")
        print("  2. If successful, extend to 10k timesteps")
        print("  3. Then 50k → 100k for final evaluation")
    else:
        print("❌ Training failed")
    print("=" * 80)


if __name__ == "__main__":
    main()
