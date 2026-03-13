import logging
import os

try:
    import torch
except ImportError:
    pass
import pandas as pd

from ztb.features.unified_feature import UnifiedFeatureEngineer
from ztb.training.unified_trainer.trainer import UnifiedTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_features_v450(input_path, output_path):
    logger.info(f"Generating features for {input_path}...")
    df = pd.read_csv(input_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    engineer = UnifiedFeatureEngineer()
    df_featured = engineer.generate_features(df, model_type="sac")
    df_featured.to_csv(output_path, index=False)
    logger.info(f"Saved featured data to {output_path}")
    return output_path


def run_test_v450(data_path, label):
    logger.info(f"🚀 v450 Range Test: {label}")
    config = {
        "training": {
            "algorithm": "sac",
            "total_timesteps": 3000,
            "log_interval": 2000,
            "environment": {
                "initial_portfolio_value": 200000.0,
                "use_continuous_actions": True,
                "dynamic_threshold_mode": "z_score",
                "z_score_window": 50,
                "z_score_threshold": 3.0,
                "z_score_method": "mad",
                "min_action_threshold": 0.002,
                "max_action_threshold": 0.08,
                "curriculum_stage": "action_discovery",
                "regime_detection_config": {
                    "use_relative": True,
                    "reference_window": 1000,
                },
            },
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 10000,
                "batch_size": 256,
                "learning_starts": 100,
                "ent_coef": "auto",
            },
            "data_config": {"data_path": data_path},
        }
    }

    try:
        trainer = UnifiedTrainer(config=config)
        success = trainer.train()
        if success:
            logger.info(f"✅ v450 Test {label} completed successfully")
        else:
            logger.error(f"❌ v450 Test {label} failed")
    except Exception as e:
        logger.error(f"Test {label} crashed: {e}", exc_info=True)


if __name__ == "__main__":
    datasets = [
        ("data/range_tight.csv", "data/range_tight_featured.csv", "Tight Range"),
        ("data/range_medium.csv", "data/range_medium_featured.csv", "Medium Range"),
        ("data/range_wide.csv", "data/range_wide_featured.csv", "Wide Range"),
    ]
    for raw, featured, label in datasets:
        if os.path.exists(raw):
            add_features_v450(raw, featured)
            run_test_v450(featured, label)
        else:
            logger.error("Missing dataset: %s", raw)
