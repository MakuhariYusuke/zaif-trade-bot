import pandas as pd

df = pd.read_csv("btc_jpy_real_dataset.csv")
print("Columns:", df.columns.tolist())
print(
    "Start:", df.iloc[0]["timestamp"] if "timestamp" in df.columns else "No timestamp"
)
print("End:", df.iloc[-1]["timestamp"] if "timestamp" in df.columns else "No timestamp")
print("Rows:", len(df))
