import json
from pathlib import Path

import pandas as pd

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

config_path = Path("config/sac_v444_advanced_regime_adaptation_config.json")
config = json.loads(config_path.read_text())

data_config = config.get("training", {}).get("data_config", {})
csv_path = data_config.get("csv_path", "data/btc_jpy_featured_dataset.csv")

if not Path(csv_path).exists():
    logger.error("Data file not found: %s", csv_path)
else:
    df = pd.read_csv(csv_path)
    env_config = config.get("environment", {}).get("config", {})
    env = HeavyTradingEnv(df=df, config=env_config)
    logger.info("env.features len= %d", len(env.features))
    logger.info("env.features= %s", env.features)
    logger.info("obs space shape= %s", env.observation_space.shape)
