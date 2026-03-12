import pandas as pd

# Load data
df = pd.read_csv("data/btc_jpy_real_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("DataFrame columns:", df.columns.tolist())
print("DataFrame index:", df.index.name)
print("First row via iloc[0]:")
row = df.iloc[0]
print("Type:", type(row))
print("Index:", row.index.tolist())
print("Values:", row.values)
print("timestamp value:", row["timestamp"])
