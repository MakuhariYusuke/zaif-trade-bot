import pandas as pd

df = pd.read_csv("data/btc_jpy_yahoo_real_20251021_featured.csv")
print("Total columns:", len(df.columns))
print("Numeric columns:", len(df.select_dtypes(include=["number"]).columns))
print(
    "Sample numeric columns:",
    df.select_dtypes(include=["number"]).columns[:10].tolist(),
)
print("Data shape:", df.shape)
print("First few rows:")
print(df.head())
