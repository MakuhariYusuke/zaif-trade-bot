import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ztb.features.models.sac.sac_v451_feature_engineering import SACv451FeatureEngineer


def main():
    input_path = project_root / "data" / "btc_jpy_1m_dataset.csv"
    output_path = project_root / "data" / "btc_jpy_1m_v451.csv"

    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

    print("Initializing SACv451FeatureEngineer...")
    engineer = SACv451FeatureEngineer()

    print("Generating features...")
    # We use a subset for testing if needed, but here we want full dataset
    features_df = engineer.generate_v451_features(df)

    print(f"Generated features shape: {features_df.shape}")
    print("Columns:", features_df.columns.tolist()[:20], "...")

    # Check for new features
    new_cols = ["hour_sin", "regime_low", "vol_rank"]
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
