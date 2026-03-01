"""
Phase 2 パフォーマンス分析スクリプト

Phase 1実装のパフォーマンスを分析し、最適化ポイントを特定します。
"""

import time

import numpy as np
import pandas as pd
import psutil

from ztb.trading.backtest.adapters import ActionSignalGuideAdapter
from ztb.utils.performance_profiler import PerformanceProfiler


def create_test_data(num_days: int = 1000) -> pd.DataFrame:
    """大規模テストデータ作成"""
    np.random.seed(42)

    dates = pd.date_range("2023-01-01", periods=num_days, freq="1H")

    # より現実的な価格データ生成
    base_price = 5000000  # 500万円
    prices = [base_price]
    returns = np.random.normal(0.0001, 0.02, num_days - 1)  # 時間足リターン

    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)

    # OHLCVデータ作成
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i - 1] if i > 0 else price
        volume = np.random.randint(1000000, 10000000)  # 100万-1000万

        data.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def analyze_signal_generation_performance():
    """シグナル生成のパフォーマンス分析"""
    print("=" * 60)
    print("シグナル生成パフォーマンス分析")
    print("=" * 60)

    profiler = PerformanceProfiler()

    # テストデータ作成
    data = create_test_data(500)  # 500時間分のデータ

    # アダプタ初期化
    with profiler.profile_context("Adapter Initialization"):
        adapter = ActionSignalGuideAdapter()

    # シグナル生成テスト
    signals = []
    batch_size = 50

    print(f"\nデータサイズ: {len(data)} rows")
    print(f"バッチサイズ: {batch_size}")

    with profiler.profile_context("Batch Signal Generation"):
        for i in range(0, len(data) - batch_size, batch_size):
            batch_data = data.iloc[i : i + batch_size]
            for j in range(len(batch_data)):
                current_data = batch_data.iloc[: j + 1]
                signal = adapter.generate_signal(current_data, 0)
                signals.append(signal)

    print(f"生成シグナル数: {len(signals)}")
    print(
        f"平均処理時間: {(profiler._profile_stats.total_tt if profiler._profile_stats else 0) / len(signals) * 1000:.2f}ms per signal"
    )


def analyze_memory_usage():
    """メモリ使用量分析"""
    print("\n" + "=" * 60)
    print("メモリ使用量分析")
    print("=" * 60)

    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    print(".1f")

    # アダプタ初期化時のメモリ使用量
    adapter = ActionSignalGuideAdapter()
    after_init_memory = process.memory_info().rss / 1024 / 1024
    print(".1f")

    # 大規模データ処理時のメモリ使用量
    data = create_test_data(1000)
    signals = []

    for i in range(100, len(data), 50):
        current_data = data.iloc[:i]
        signal = adapter.generate_signal(current_data, 0)
        signals.append(signal)

        if i % 200 == 0:
            current_memory = process.memory_info().rss / 1024 / 1024
            print(".1f")

    final_memory = process.memory_info().rss / 1024 / 1024
    print(".1f")
    print(".1f")


def analyze_risk_management_overhead():
    """リスク管理機能のオーバーヘッド分析"""
    print("\n" + "=" * 60)
    print("リスク管理機能オーバーヘッド分析")
    print("=" * 60)

    profiler = PerformanceProfiler()
    data = create_test_data(200)

    # リスク管理なしの場合
    adapter_no_risk = ActionSignalGuideAdapter()
    # 一時的にリスク管理を無効化（モック）
    adapter_no_risk.risk_manager = None

    # リスク管理ありの場合
    adapter_with_risk = ActionSignalGuideAdapter()

    # 比較テスト
    signals_no_risk = []
    signals_with_risk = []

    # リスク管理なし
    start_time = time.time()
    for i in range(50, len(data), 10):
        current_data = data.iloc[:i]
        try:
            signal = adapter_no_risk.generate_signal(current_data, 0)
            signals_no_risk.append(signal)
        except Exception:
            pass  # エラーが発生したらスキップ
    time_no_risk = time.time() - start_time

    # リスク管理あり
    start_time = time.time()
    for i in range(50, len(data), 10):
        current_data = data.iloc[:i]
        signal = adapter_with_risk.generate_signal(current_data, 0)
        signals_with_risk.append(signal)
    time_with_risk = time.time() - start_time

    print(f"リスク管理なし: {time_no_risk:.3f}s ({len(signals_no_risk)} signals)")
    print(f"リスク管理あり: {time_with_risk:.3f}s ({len(signals_with_risk)} signals)")
    print(".2f")
    print(".1f")


def identify_performance_bottlenecks():
    """パフォーマンスボトルネックの特定"""
    print("\n" + "=" * 60)
    print("パフォーマンスボトルネック分析")
    print("=" * 60)

    import cProfile
    import pstats
    from io import StringIO

    data = create_test_data(100)
    adapter = ActionSignalGuideAdapter()

    # プロファイリング実行
    profiler = cProfile.Profile()
    profiler.enable()

    # シグナル生成実行
    for i in range(20, len(data), 5):
        current_data = data.iloc[:i]
        signal = adapter.generate_signal(current_data, 0)

    profiler.disable()

    # 結果分析
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)  # 上位20件

    print("プロファイリング結果（上位20件）:")
    print(s.getvalue())


def main():
    """メイン分析実行"""
    print("Phase 2 パフォーマンス分析")
    print("=" * 80)

    try:
        analyze_signal_generation_performance()
        analyze_memory_usage()
        analyze_risk_management_overhead()
        identify_performance_bottlenecks()

        print("\n" + "=" * 80)
        print("分析完了 - Phase 2最適化の指針:")
        print("1. シグナル生成のバッチ処理最適化")
        print("2. メモリ使用量の監視と解放")
        print("3. リスク計算のキャッシュ化")
        print("4. ATR計算の効率化")
        print("=" * 80)

    except Exception as e:
        print(f"分析中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
