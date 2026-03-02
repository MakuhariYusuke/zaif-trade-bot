"""
SAC v395a Test - 1k timesteps quick iteration
連続行動空間での動作確認用
"""

from ztb.training.utils.training_main_template import create_simple_main_template
from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.unified_trainer.trainer import UnifiedTrainer

main = create_simple_main_template(
    UnifiedTrainer,
    "configs/sac_v395a_test_1k.json",
    "SAC v395a Test - 1k timesteps quick iteration"
)

if __name__ == "__main__":
    main()
