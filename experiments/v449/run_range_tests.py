import logging
import os

# Pre-import torch to avoid DLL errors
try:
    import torch
except ImportError:
    pass
import pandas as pd

from ztb.features.unified_feature import UnifiedFeatureEngineer
from ztb.training.unified_trainer.trainer import UnifiedTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def add_features(input_path, output_path):
    logger.info(f"Generating features for {input_path}...")
    df = pd.read_csv(input_path)

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    engineer = UnifiedFeatureEngineer()
    # Use 'sac' model type to get rich features compatible with HeavyTradingEnv
    df_featured = engineer.generate_features(df, model_type="sac")

    df_featured.to_csv(output_path, index=False)
    logger.info(f"Saved featured data to {output_path}")
    return output_path


def run_test(data_path, label):
    logger.info(f"🚀 Starting Test: {label}")

    config = {
        "model_name": f"test_{label}",
        "training": {
            "algorithm": "sac",
            "total_timesteps": 2000,
            "log_interval": 2000,  # Only log at end
            "environment": {
                "initial_portfolio_value": 1000000.0,
                "use_continuous_actions": True,
                # Adaptive Thresholding
                "adaptive_threshold_mode": True,
                "continuous_to_discrete_threshold": 0.02,
                "threshold_volatility_multiplier": 2.0,
                "min_action_threshold": 0.002,
                "max_action_threshold": 0.08,
                # Action Signal Guide
                "signal_guidance_enabled": True,
                "signal_guidance_mode": "partial",
                # Reward Settings (Smart Incentive)
                "reward_settings": {
                    "use_smart_incentive": True,
                    "smart_incentive_mode": "regime_adaptive",
                },
            },
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 10000,
                "learning_starts": 100,
                "batch_size": 256,
                "ent_coef": "auto",
            },
            "data_config": {"data_path": data_path},
        },
    }

    try:
        trainer = UnifiedTrainer(config=config)
        success = trainer.train()

        if success:
            logger.info(f"✅ Test {label} completed successfully")
        else:
            logger.error(f"❌ Test {label} failed")

    except Exception as e:
        logger.error(f"Test {label} crashed: {e}", exc_info=True)


if __name__ == "__main__":
    datasets = [
        ("data/range_tight.csv", "data/range_tight_featured.csv", "Tight Range"),
        ("data/range_wide.csv", "data/range_wide_featured.csv", "Wide Range"),
        ("data/range_choppy.csv", "data/range_choppy_featured.csv", "Choppy Range"),
    ]

    for raw_path, featured_path, label in datasets:
        if os.path.exists(raw_path):
            add_features(raw_path, featured_path)
            run_test(featured_path, label)
        else:
            logger.error(f"Raw data not found: {raw_path}")
