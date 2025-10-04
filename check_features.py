import pandas as pd

df = pd.read_csv("ml-dataset.csv", nrows=5)
exclude_cols = ["ts", "timestamp", "exchange", "pair", "episode_id", "side", "source"]
features = [c for c in df.columns if c not in exclude_cols]

print(f"All columns: {list(df.columns)}")
print(f"Excluded: {exclude_cols}")
print(f"Features used: {features}")
print(f"Feature count: {len(features)}")
print("Sample data:")
print(df[features].head())
