import pandas as pd

df = pd.read_csv("data/btc_jpy_real_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("=== 分足データ詳細分析 ===")
print(f"開始日時: {df.timestamp.min()}")
print(f"終了日時: {df.timestamp.max()}")
print(
    f"総期間: {(df.timestamp.max() - df.timestamp.min()).total_seconds() / 3600:.1f}時間"
)
print(f"データ数: {len(df)}行")
print(
    f"時間間隔: {(df.timestamp.iloc[1] - df.timestamp.iloc[0]).total_seconds() / 60:.0f}分"
)
print()

print("=== 価格変動統計 ===")
print(f"始値: {df.close.iloc[0]:,.0f} JPY")
print(f"終値: {df.close.iloc[-1]:,.0f} JPY")
print(f"最高値: {df.high.max():,.0f} JPY")
print(f"最低値: {df.low.min():,.0f} JPY")
print(
    f"価格変動幅: {((df.close.iloc[-1] - df.close.iloc[0]) / df.close.iloc[0] * 100):+.2f}%"
)
print(
    f"最大ドローダウン: {((df.low.min() - df.high.max()) / df.high.max() * 100):+.2f}%"
)
