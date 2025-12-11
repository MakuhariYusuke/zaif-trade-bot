import logging

from ztb.training.unified_trainer.trainer import UnifiedTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_config(mode: str):
    base = {
        "training": {
            "algorithm": "sac",
            "total_timesteps": 2000,
            "model_name": f"sac_v450_ab_{mode}",
            "sac_hyperparameters": {
                "learning_rate": 0.0003,
                "buffer_size": 10000,
                "batch_size": 256,
                "learning_starts": 100,
            },
            "data_config": {"data_path": "data/btc_jpy_featured_dataset.csv"},
            "environment": {
                "use_continuous_actions": True,
                "curriculum_stage": "action_discovery",
            },
        }
    }

    # Mode-specific overrides
    if mode == "fixed":
        base["training"]["environment"]["dynamic_threshold_mode"] = "fixed"
        base["training"]["environment"]["min_action_threshold"] = 0.002
        base["training"]["environment"]["max_action_threshold"] = 0.08
    elif mode == "volatility":
        base["training"]["environment"]["dynamic_threshold_mode"] = "volatility"
        base["training"]["environment"]["threshold_volatility_multiplier"] = 2.0
    elif mode == "z_score":
        base["training"]["environment"]["dynamic_threshold_mode"] = "z_score"
        base["training"]["environment"]["z_score_window"] = 50
        base["training"]["environment"]["z_score_threshold"] = 3.0
        base["training"]["environment"]["z_score_method"] = "mad"

    base["training"]["environment"]["initial_balance"] = 200000.0
    # Duplicate algorithm-specific hyperparams to top-level for TrainingConfigManager compatibility
    if "sac_hyperparameters" in base["training"]:
        base["sac_hyperparameters"] = base["training"]["sac_hyperparameters"].copy()
    return base


def run_ab_test():
    logger.info(
        "Starting v450 A/B test for threshold modes (fixed / volatility / z_score)"
    )
    configs = [(m, create_config(m)) for m in ["fixed", "volatility", "z_score"]]

    results = []
    for mode, cfg in configs:
        logger.info("Running mode: %s", mode)
        try:
            # Ensure data exists - generate synthetic data if missing (for quick tests)
            data_path = cfg["training"]["data_config"]["data_path"]
            import os

            if not os.path.exists(data_path):
                logger.info(
                    "Data file not found. Generating synthetic data for quick AB test..."
                )
                try:
                    from ztb.data.synthetic_data_generator import (
                        generate_synthetic_market_data,
                    )

                    df = generate_synthetic_market_data(n_samples=2000)
                    os.makedirs(os.path.dirname(data_path), exist_ok=True)
                    df.to_csv(data_path, index=False)
                    logger.info(f"Synthetic data saved to {data_path}")
                except Exception:
                    logger.warning(
                        "Failed to generate synthetic data. Continuing and letting trainer validation catch missing data"
                    )

            trainer = UnifiedTrainer(config=cfg)
            success = trainer.train()
            results.append({"mode": mode, "success": bool(success)})
            logger.info("Completed mode %s: success=%s", mode, success)
        except Exception as e:
            logger.error("Mode %s failed: %s", mode, e)
            results.append({"mode": mode, "success": False, "error": str(e)})

    # Basic result summary
    for r in results:
        logger.info("Result: %s", r)


if __name__ == "__main__":
    run_ab_test()
