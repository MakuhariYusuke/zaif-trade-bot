"""
SAC v395d (Optimal) - Best of Both Worlds
v395aの低いLoss + v395b/cの安定したent_coef
"""
import time
from pathlib import Path

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.file_utils import safe_json_load
from ztb.utils.training_utils import display_training_complete


def main():
    print("🎯 SAC v395d - Optimal Parameter Set")
    print("=" * 80)

    start_time = time.time()

    config_path = "configs/sac_v395d_optimal.json"

    # 設定ファイル読み込み
    config = safe_json_load(Path(config_path))

    print("📊 Parameter Selection Strategy:")
    print("-" * 80)
    for param, choice in config["analysis_summary"]["parameter_choices"].items():
        print(f"  • {param:20s}: {choice}")
    print()

    print("🎯 Expected Outcomes:")
    print("-" * 80)
    for metric, target in config["analysis_summary"]["expected_outcomes"].items():
        print(f"  • {metric:20s}: {target}")
    print()

    print("🚀 Starting 5k timesteps training...")
    print("=" * 80)
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
