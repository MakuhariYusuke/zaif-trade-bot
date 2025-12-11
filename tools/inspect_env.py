import json
import sys
from pathlib import Path

import pandas as pd

proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.feature_mapping import map_trained_features
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    CONFIG = proj / "config" / "sac_v444_advanced_regime_adaptation_config.json"
    DATA_CSV = proj / "data" / "btc_jpy_real_dataset.csv"

    if not CONFIG.exists():
        logger.error("Config not found: %s", CONFIG)
        return 1
    if not DATA_CSV.exists():
        logger.error("Data CSV not found: %s", DATA_CSV)
        return 1

    with open(CONFIG, "r") as f:
        cfg = json.load(f)

    # replicate mapping logic from backtest_sac_v444
    trained_feature_names = [
        "Supertrend",
        "Supertrend_Direction",
        "BB_Upper",
        "BB_Lower",
        "BB_Middle",
        "BB_Width",
        "BB_Position",
        "OBV",
    ]

    env_config = cfg.get("environment", {}).get("config", {})
    if "feature_names" not in env_config:
        df = pd.read_csv(DATA_CSV)
        matched = map_trained_features(df, trained_feature_names)
        if matched:
            env_config["feature_names"] = matched
        else:
            logger.warning(
                "Could not fully map trained features, leaving to auto-discovery.",
            )

    # Initialize env and print diagnostics
    logger.info(
        "Initializing HeavyTradingEnv with env_config keys: %s",
        list(env_config.keys())[:20],
    )
    df = pd.read_csv(DATA_CSV)
    env = HeavyTradingEnv(df=df, config=env_config)
    logger.info("env.features (len): %d", len(env.features))
    logger.info("env.features sample: %s", env.features[:24])
    logger.info("_feature_matrix.shape: %s", getattr(env, "_feature_matrix").shape)
    logger.info("df columns tail: %s", list(env.df.columns[-20:]))
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
