import pandas as pd

df = pd.read_csv("data/btc_jpy_extended_dataset.csv")
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("First few rows:")
print(df.head())
print("NaN counts per column:")
print(df.isnull().sum())
print("Data types:")
print(df.dtypes)
