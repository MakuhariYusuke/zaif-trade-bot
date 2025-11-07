import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('data/quick_training_data.csv')

# 基本統計を表示
print('=== 学習データ分析 ===')
print(f'データ行数: {len(df)}')
print(f'カラム数: {len(df.columns)}')
print(f'カラム: {list(df.columns)}')

# RSIの分布
print(f'RSI平均: {df["RSI"].mean():.2f}')
print(f'RSI最小: {df["RSI"].min():.2f}')
print(f'RSI最大: {df["RSI"].max():.2f}')

# 価格変動の分析
df['returns'] = df['close'].pct_change()
print(f'平均リターン: {df["returns"].mean():.6f}')
print(f'リターン標準偏差: {df["returns"].std():.6f}')
print(f'最大ドローダウン: {((df["close"].cummax() - df["close"]) / df["close"].cummax()).max():.4f}')

# 市場レジーム分析（ログから）
print('\n=== 市場レジーム分析 ===')
print('Regime distribution over 200 steps: bull: 44.5%, bear: 50.5%, sideways: 5.0%')
print('- ベア相場がやや優勢')
print('- サイドウェイズ相場は少ない')