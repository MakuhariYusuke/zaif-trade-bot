"""
SAC v395b (Stable) - 5k timesteps
積極的なパラメータ調整でCritic Loss安定化
"""

from ztb.training.utils.training_main_template import create_simple_main_template
from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.unified_trainer.trainer import UnifiedTrainer

extra_info = """📊 Key Changes from v395a:
  • ent_coef: 1.0 -> 0.5 (more conservative exploration)
  • learning_rate: 3e-4 -> 1e-4 (slower learning)
  • buffer_size: 1e6 -> 5e5 (smaller replay buffer)
  • batch_size: 256 -> 128 (smaller batches)

🎯 Expected Improvements:
  • Critic Loss < 1e7 (was 4.34e7)
  • ent_coef stable in 0.5-2.0 range (was 4.03)
  • Smoother learning curve"""

main = create_simple_main_template(
    UnifiedTrainer,
    "configs/sac_v395b_stable.json",
    "SAC v395b Test - Stabilized Parameters",
    extra_info
)


if __name__ == "__main__":
    main()
