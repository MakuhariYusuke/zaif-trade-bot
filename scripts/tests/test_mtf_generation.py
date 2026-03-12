import os
import sys

# Fix DLL load error for PyTorch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
try:
    import torch
except ImportError:
    pass

import logging

import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from ztb.features.models.sac.sac_v427_feature_engineering import SACv427FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mtf_features():
    # Load a small chunk of 1m data
    data_path = "data/yahoo_finance/btc_jpy_1m_converted.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    # Use a subset for speed
    df_subset = df.head(10000).copy()

    logger.info(f"Data loaded. Shape: {df_subset.shape}")

    # Initialize Feature Engineer
    fe = SACv427FeatureEngineer()

    # Generate features
    logger.info("Generating features...")
    featured_df = fe.generate_v427_features(df_subset)

    logger.info(f"Features generated. Shape: {featured_df.shape}")

    # Check for MTF columns
    mtf_keywords = ["_5m", "_15m", "_1h"]
    mtf_cols = [
        col for col in featured_df.columns if any(k in col for k in mtf_keywords)
    ]

    logger.info(f"Found {len(mtf_cols)} MTF features.")
    if len(mtf_cols) > 0:
        logger.info(f"Sample MTF columns: {mtf_cols[:10]}")

        # Check values
        sample_col = mtf_cols[0]
        logger.info(f"Sample values for {sample_col} (HEAD):")
        logger.info(featured_df[sample_col].head(20))
        logger.info(f"Sample values for {sample_col} (TAIL):")
        logger.info(featured_df[sample_col].tail(20))
    else:
        logger.warning("No MTF features found!")

    # Check for NaNs
    nans = featured_df.isna().sum().sum()
    logger.info(f"Total NaNs in featured data: {nans}")


if __name__ == "__main__":
    test_mtf_features()
