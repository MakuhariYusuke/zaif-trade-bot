import pandas as pd
import numpy as np
from ztb.analysis.walk_forward_analyzer import WalkForwardAnalyzer

# サンプルデータ作成
dates = pd.date_range('2023-01-01', periods=200, freq='D')
np.random.seed(42)
trend = np.linspace(0, 50, 200)
noise = np.random.randn(200) * 3
prices = 100 + trend + noise
data = pd.DataFrame({
    'open': prices,
    'high': prices + np.abs(np.random.randn(200)),
    'low': prices - np.abs(np.random.randn(200)),
    'close': prices + np.random.randn(200) * 0.5
}, index=dates)

print(f'データ期間: {data.index.min()} から {data.index.max()}')
print(f'総日数: {(data.index.max() - data.index.min()).days} 日')
print(f'データ行数: {len(data)}')

analyzer = WalkForwardAnalyzer()
try:
    windows = analyzer.create_sliding_windows(data, train_days=30, test_days=10, step_days=15, min_samples=10)
    print(f'作成されたウィンドウ数: {len(windows)}')
    for i, w in enumerate(windows[:3]):  # 最初の3つを表示
        print(f'ウィンドウ {i}: 訓練 {w.train_start.date()} - {w.train_end.date()}, テスト {w.test_start.date()} - {w.test_end.date()}')
except Exception as e:
    print(f'エラー: {e}')