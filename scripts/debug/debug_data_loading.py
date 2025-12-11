import os

# Try to fix DLL load error
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
try:
    import torch
except ImportError:
    print("Could not import torch")

import logging
from pathlib import Path

import pandas as pd

from ztb.features.models.sac.sac_v427_feature_engineering import SACv427FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_loading():
    data_file = "data/btc_jpy_real_dataset.csv"
    if Path(data_file).exists():
        data = pd.read_csv(data_file)
        logger.info(f"DEBUG: Raw data loaded. Shape: {data.shape}")
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data.set_index("timestamp", inplace=True)
        logger.info(f"✅ Loaded {len(data)} rows from {data_file}")
    else:
        logger.error("Data file not found")
        return

    logger.info("Applying feature engineering...")
    try:
        feature_engineer = SACv427FeatureEngineer()
        featured_data = feature_engineer.generate_v427_features(
            data, skip_quality_filtering=False
        )

        if len(featured_data) == len(data):
            featured_data.index = data.index
        else:
            logger.warning(
                "Feature data length (%d) differs from raw data (%d)",
                len(featured_data),
                len(data),
            )

        logger.info(f"DEBUG: Featured data shape: {featured_data.shape}")

        # Check for zeros in close price if it exists
        if "close" in featured_data.columns:
            zeros = (featured_data["close"] == 0).sum()
            logger.info(f"Zeros in 'close' column: {zeros}")

            # Check NaNs
            nans = featured_data["close"].isna().sum()
            logger.info(f"NaNs in 'close' column: {nans}")

            # Check if fillna(0) would create zeros
            filled = featured_data.fillna(0)
            zeros_after = (filled["close"] == 0).sum()
            logger.info(f"Zeros in 'close' after fillna(0): {zeros_after}")

        else:
            logger.warning("'close' column missing from featured_data")

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_loading()
