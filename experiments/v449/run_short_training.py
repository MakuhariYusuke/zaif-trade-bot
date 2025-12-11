import logging
import os
import time

from ztb.training.unified_trainer.trainer import UnifiedTrainer

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_short_training():
    logger.info("🚀 Starting Short Training for Validation...")

    # Config enabling Adaptive Thresholding AND Action Signal Guide
    config = {
        "training": {
            "algorithm": "sac",
            "total_timesteps": 5000,  # Short run
            "log_interval": 100,
            "environment": {
                "initial_portfolio_value": 200000.0,
                "use_continuous_actions": True,  # Required for SAC
                # Domain Randomization
                "exchange_profile": {
                    "name": "base_profile",
                    "maker_fee_rate": 0.0,
                    "taker_fee_rate": 0.0,
                    "slippage_rate": 0.0,
                    "latency_ms": 0.0,
                },
                "domain_randomization": {
                    "enabled": True,
                    "intensity": 0.5,  # Test intensity scaling
                    "maker_fee_range": [0.001, 0.005],
                    "taker_fee_range": [0.002, 0.010],
                    "slippage_range": [0.01, 0.05],
                    "latency_range": [50.0, 500.0],
                },
                # Adaptive Thresholding
                "adaptive_threshold_mode": True,
                "continuous_to_discrete_threshold": 0.02,  # Set explicit base threshold
                "threshold_volatility_multiplier": 2.0,
                "min_action_threshold": 0.002,
                "max_action_threshold": 0.08,
                # Action Signal Guide
                "signal_guidance_enabled": True,
                "signal_guidance_mode": "partial",  # or "full"
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
            "data_config": {
                # Use synthetic data generation if file doesn't exist or for testing
                # But UnifiedTrainer expects a file.
                # We'll point to the existing one or let it fail if missing (and then maybe generate synthetic)
                "data_path": "data/btc_jpy_featured_dataset.csv"
            },
        }
    }

    # Check if data exists, if not generate synthetic data
    data_path = config["training"]["data_config"]["data_path"]
    if not os.path.exists(data_path):
        logger.info("Data file not found. Generating synthetic data...")
        from ztb.data.synthetic_data_generator import generate_synthetic_market_data

        df = generate_synthetic_market_data(n_samples=5000)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Synthetic data saved to {data_path}")

    try:
        # Initialize Trainer
        trainer = UnifiedTrainer(config=config)

        # Run Training
        start_time = time.time()
        success = trainer.train()
        duration = time.time() - start_time

        if success:
            logger.info(f"✅ Training completed successfully in {duration:.2f}s")
        else:
            logger.error("❌ Training failed")

    except Exception as e:
        logger.error(f"Training crashed: {e}", exc_info=True)


if __name__ == "__main__":
    run_short_training()
