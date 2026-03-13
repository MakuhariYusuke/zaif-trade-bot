import logging
import os
import time

from ztb.training.unified_trainer.trainer import UnifiedTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_short_training():
    logger.info(
        "🚀 v450: Short Training run with dynamic thresholding and action_discovery stage"
    )

    # Use the v450 template config as baseline
    config = {
        "training": {
            "algorithm": "sac",
            "total_timesteps": 5000,
            "log_interval": 100,
            "environment": {
                "initial_portfolio_value": 200000.0,
                "use_continuous_actions": True,
                "domain_randomization": {
                    "enabled": True,
                    "intensity": 0.5,
                    "maker_fee_range": [0.001, 0.005],
                    "taker_fee_range": [0.002, 0.010],
                    "slippage_range": [0.01, 0.05],
                    "latency_range": [50.0, 500.0],
                },
                "dynamic_threshold_mode": "z_score",
                "z_score_window": 50,
                "z_score_threshold": 3.0,
                "z_score_method": "mad",
                "min_action_threshold": 0.002,
                "max_action_threshold": 0.08,
                "regime_detection_config": {
                    "use_relative": True,
                    "reference_window": 1000,
                    "percentile_threshold": 90,
                },
                "curriculum_stage": "action_discovery",
                "reward_settings": {
                    "action_discovery": {"enabled": True, "scale": 0.5},
                    "use_smart_incentive": True,
                },
            },
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 10000,
                "learning_starts": 100,
                "batch_size": 256,
                "ent_coef": "auto",
            },
            "data_config": {"data_path": "data/btc_jpy_featured_dataset.csv"},
        }
    }

    # Ensure data exists; if not, do minimal synthetic generation
    data_path = config["training"]["data_config"]["data_path"]
    if not os.path.exists(data_path):
        logger.info(
            "Data file not found. Generating minimal synthetic data for short run"
        )
        from ztb.data.synthetic_data_generator import generate_synthetic_market_data

        df = generate_synthetic_market_data(n_samples=6000)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info("Saved synthetic data: %s", data_path)

    try:
        trainer = UnifiedTrainer(config=config)
        start_time = time.time()
        success = trainer.train()
        dur = time.time() - start_time
        if success:
            logger.info("✅ v450 training completed: %.2fs", dur)
        else:
            logger.error("❌ v450 training failed")
    except Exception as exc:
        logger.exception("Training raised an exception: %s", exc)


if __name__ == "__main__":
    run_short_training()
