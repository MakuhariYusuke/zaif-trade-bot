import pandas as pd

df = pd.read_csv("data/btc_jpy_extended_dataset.csv")

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Filter to 2023 data
df_2023 = df[df["timestamp"].dt.year == 2023].copy()

print("2023 data shape:", df_2023.shape)
print(
    "2023 data date range:",
    df_2023["timestamp"].min(),
    "to",
    df_2023["timestamp"].max(),
)
print("NaN counts in 2023 data:")
print(df_2023.isnull().sum())

# Check if we have any valid technical indicators in 2023
valid_indicators = df_2023.dropna()
print("Rows with no NaN values in 2023:", len(valid_indicators))
