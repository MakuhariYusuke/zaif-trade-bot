import os

# Try to fix DLL load error
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
try:
    import torch
except ImportError:
    pass

import logging

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.config.unified_config import UnifiedConfig
from ztb.features.models.sac.sac_v427_feature_engineering import SACv427FeatureEngineer
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv

# Add project root to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




if __name__ == "__main__":
    analyze_bias()
