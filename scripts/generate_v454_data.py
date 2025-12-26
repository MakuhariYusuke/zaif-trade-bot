import sys
import argparse
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ztb.features.models.sac.sac_v454_feature_engineering import SACv454FeatureEngineer
from ztb.utils.data_utils import load_csv_data


def main():
    parser = argparse.ArgumentParser(description="Generate v454 features from market data.")
    parser.add_argument("--input", type=str, help="Path to input CSV file")
    parser.add_argument("--output", type=str, help="Path to output CSV file")
    args = parser.parse_args()

    # Default paths
    default_input = project_root / "data" / "btc_jpy_1m_dataset.csv"
    default_output = project_root / "data" / "btc_jpy_1m_v454.csv"

    input_path = Path(args.input) if args.input else default_input
    output_path = Path(args.output) if args.output else default_output

    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return

    print(f"Loading data from {input_path}")
    df = load_csv_data(input_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

    print("Initializing SACv454FeatureEngineer...")
    engineer = SACv454FeatureEngineer()

    print("Generating features...")
    features_df = engineer.generate_v454_features(df)

    print(f"Generated features shape: {features_df.shape}")
    print("Columns:", features_df.columns.tolist()[:20], "...")

    # Check for new features
    new_cols = ["vol_ema_14", "trend_dev_100", "noise_index"]
    for col in new_cols:
        if col in features_df.columns:
            print(f"Found {col}: OK")
        else:
            print(f"Missing {col}: FAIL")

    print(f"Saving to {output_path}")
    features_df.to_csv(output_path)
    print("Done.")


if __name__ == "__main__":
    main()
